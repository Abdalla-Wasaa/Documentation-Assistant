import os
from pathlib import Path
from typing import Any,Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Resolve from this file so IDE debugging and different working directories work.
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError(
        f"Set OPENROUTER_API_KEY in {env_path}, or in your process environment. "
        "Also check that your debugger does not set it to an empty value."
    )

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    chunk_size=50,
    retry_min_seconds=5,
    show_progress_bar=False,
)

vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"),
                                   embedding=embeddings)

# Initialize chat model
model = init_chat_model(
    "openai/gpt-5.2",
    model_provider="openai",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vector_store.as_retriever().invoke(query,k=4)

    #Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of restrieved documents
    """
    # create the agent with retrieval tool

    system_prompt = (
        "You are a helpful AI assistant that answers questions about Langchain documentation."
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent = create_agent(model,tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"] [-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return{
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)
