import os
import uuid
import tempfile
import asyncio
import random
import httpx
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="HackRx RAG API",
    description="An API to answer questions from a PDF document using a remote embedding service.",
    version="2.0.0"
)

# --- Constants and API Key Management ---

# Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Main API Key for securing this FastAPI service
API_KEY = os.getenv("API_KEY")

# Groq API Keys (for load balancing/failover)
GROQ_API_KEYS = [key for key in [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 4)] if key]
if not GROQ_API_KEYS:
    raise ValueError("At least one Groq API key (e.g., GROQ_API_KEY_1) must be set.")

# --- NEW: Configuration for the remote embedding service ---
EMBEDDING_API_URL = "https://rahulbro123-mymodel.hf.space/embed"
EMBEDDING_API_BATCH_SIZE = 32  # Number of texts to send in a single API call

KNOWN_PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
KNOWN_INDEX_NAME = "hackrx-index-f8939881"

# --- NEW: Custom Embeddings Client ---

class RemoteEmbeddingClient(Embeddings):
    """A custom LangChain embedding class to call a remote API."""
    def __init__(self, api_url: str, batch_size: int):
        self.api_url = api_url
        self.batch_size = batch_size
        # Use a single httpx.AsyncClient for connection pooling and performance
        self.client = httpx.AsyncClient(timeout=60.0)

    async def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Helper to make a single API call."""
        try:
            response = await self.client.post(self.api_url, json={"texts": texts})
            response.raise_for_status()
            data = response.json()
            if "embeddings" not in data or not isinstance(data["embeddings"], list):
                raise ValueError("Invalid response format from embedding API")
            return data["embeddings"]
        except httpx.RequestError as e:
            print(f"HTTP request to embedding API failed: {e}")
            raise HTTPException(status_code=503, detail=f"Embedding service is unavailable: {e}")
        except Exception as e:
            print(f"An error occurred while calling embedding API: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process embeddings: {e}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of documents using parallel batch requests."""
        all_embeddings = []
        # Create batches of texts
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Create a task for each batch
            tasks.append(self._call_embedding_api(batch))
        
        # Run all batch requests in parallel
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results from all batches
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query."""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    # Sync versions are required by the base class, but we primarily use async
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        return asyncio.run(self.aembed_query(text))


# --- Global Instances (loaded once on startup) ---
embeddings = RemoteEmbeddingClient(api_url=EMBEDDING_API_URL, batch_size=EMBEDDING_API_BATCH_SIZE)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


# --- Pydantic Models ---

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to be processed.")
    questions: List[str] = Field(..., description="List of questions to ask about the document.")

class AnswerResponse(BaseModel):
    answers: List[str]


# --- Asynchronous Pinecone and Document Processing Functions ---

async def create_index_if_not_exists_async(index_name: str):
    """Checks for and creates a Pinecone index if it doesn't exist."""
    def _create_sync():
        if index_name not in pinecone_client.list_indexes().names():
            print(f"Index '{index_name}' does not exist. Creating it...")
            pinecone_client.create_index(
                name=index_name,
                dimension=768,  # Dimension of all-mpnet-base-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"Index '{index_name}' already exists.")
    await asyncio.to_thread(_create_sync)

def _process_pdf_sync(pdf_content: bytes) -> List[str]:
    """Synchronous, CPU-bound helper for loading and chunking a PDF."""
    print("Starting PDF processing and chunking...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content)
        file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(docs)
        print("PDF chunking complete.")
        return [doc.page_content for doc in chunked_docs]
    finally:
        os.remove(file_path)

async def process_and_embed_pdf_async(pdf_url: str, index_name: str):
    """Asynchronously downloads, processes, embeds, and upserts a PDF."""
    # 1. Download PDF asynchronously
    async with httpx.AsyncClient() as client:
        try:
            print(f"Downloading PDF from {pdf_url}...")
            response = await client.get(pdf_url, follow_redirects=True, timeout=30)
            response.raise_for_status()
            pdf_content = response.content
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    # 2. Process and chunk the PDF in a thread to avoid blocking
    text_chunks = await asyncio.to_thread(_process_pdf_sync, pdf_content)
    
    # 3. Embed all text chunks in parallel using the remote API
    print(f"Embedding {len(text_chunks)} text chunks using remote API...")
    vectors = await embeddings.aembed_documents(text_chunks)
    print("Embedding complete.")

    # 4. Prepare vectors and metadata for Pinecone upsert
    to_upsert = []
    for i, (text, vec) in enumerate(zip(text_chunks, vectors)):
        to_upsert.append({
            "id": f"doc_chunk_{i}",
            "values": vec,
            "metadata": {"text": text}
        })
    
    # 5. Upsert to Pinecone in a thread
    def _upsert_sync():
        index = pinecone_client.Index(index_name)
        print(f"Upserting {len(to_upsert)} vectors to Pinecone index '{index_name}'...")
        index.upsert(vectors=to_upsert, batch_size=100) # Pinecone recommends batching
        print("Upserting to Pinecone complete.")
        
    await asyncio.to_thread(_upsert_sync)


async def answer_one_question(question: str, query_vector: list, index_name: str) -> str:
    """Generates an answer for one question, retrying with different Groq keys."""
    # 1. Search for context in Pinecone
    def _search_sync():
        try:
            index = pinecone_client.Index(index_name)
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)
            return "\n\n---\n\n".join(
                [match['metadata'].get('text', '') for match in results.get("matches", [])]
            )
        except Exception as e:
            print(f"Error during Pinecone search: {e}")
            return ""
            
    context = await asyncio.to_thread(_search_sync)

    if not context.strip():
        return "This information is not specified in the policy document."

    # 2. Generate answer with LLM, with failover for API keys
    prompt = f"""You are a helpful assistant. Only answer based on the context below. If the answer is not in the context, say "I could not find the answer for the specified document.". Be formal and concise. Start directly with the answer.
CONTEXT:
---
{context}
---
QUESTION: {question}
ANSWER:"""

    shuffled_keys = random.sample(GROQ_API_KEYS, len(GROQ_API_KEYS))
    for i, key in enumerate(shuffled_keys):
        try:
            llm = ChatGroq(temperature=0, groq_api_key=key, model_name="llama3-8b-8192")
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error with Groq Key #{i+1} for question '{question[:30]}...': {e}. Trying next...")
    
    return "The answer could not be generated due to an external service error."


# --- API Endpoint ---

@app.post("/hackrx/run", response_model=AnswerResponse, tags=["RAG"])
async def hackrx_run(request: QueryRequest, authorization: Optional[str] = Header(None)):
    """
    Main endpoint to process a PDF and answer questions about it.
    """
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key.")

    print("\n--- New Request Received ---")
    
    try:
        # Determine if we need to process a new document or use the existing one
        if request.documents == KNOWN_PDF_URL:
            index_name = KNOWN_INDEX_NAME
            print(f"Using known document and index: {index_name}")
        else:
            index_name = f"hackrx-index-{uuid.uuid4().hex[:8]}"
            print(f"Processing new document. Generated index: {index_name}")
            # This workflow creates the index and populates it
            await create_index_if_not_exists_async(index_name)
            await process_and_embed_pdf_async(request.documents, index_name)

        # Embed all user questions in a single parallel batch call
        print("Embedding questions using remote API...")
        query_vectors = await embeddings.aembed_documents(request.questions)

        # Create parallel tasks to answer each question
        print("Generating answers in parallel...")
        tasks = [
            answer_one_question(q, v, index_name)
            for q, v in zip(request.questions, query_vectors)
        ]
        answers = await asyncio.gather(*tasks)

        print("--- Request Processing Done ---\n")
        return AnswerResponse(answers=answers)

    except Exception as e:
        print(f"An unexpected error occurred in the main endpoint: {e}")
        # Re-raise as HTTPException for proper client response
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")