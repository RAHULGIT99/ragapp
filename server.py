import os
import uuid
import tempfile
import asyncio
import random
import httpx
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- Constants and API Key Management ---

# Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Main API Key for securing this FastAPI service
API_KEY = os.getenv("API_KEY")

# Groq API Keys
GROQ_API_KEYS = [
    key for key in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if key is not None
]
if not GROQ_API_KEYS:
    raise ValueError("At least one Groq API key (e.g., GROQ_API_KEY_1) must be set in the .env file.")

KNOWN_PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
KNOWN_INDEX_NAME = "hackrx-index-f8939881"

# --- Global Instances (loaded once on startup) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# --- Asynchronous Pinecone and Document Processing Functions ---

def _create_index_sync(index_name: str):
    """Synchronous helper function to create a Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating it...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists.")

async def create_index_if_not_exists_async(index_name: str):
    """Runs the synchronous index creation in a separate thread to avoid blocking."""
    await asyncio.to_thread(_create_index_sync, index_name)

def _load_and_embed_sync(pdf_content: bytes, index_name: str):
    """Synchronous helper for CPU/IO-bound PDF processing and embedding."""
    print(f"Starting PDF processing and embedding for index '{index_name}'...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content)
        file_path = tmp_file.name

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunked_docs = text_splitter.split_documents(docs)
        PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=PINECONE_API_KEY
        )
        print("Embedding and upserting to Pinecone complete.")
    finally:
        os.remove(file_path)

async def load_and_embed_pdf_async(pdf_url: str, index_name: str):
    """Asynchronously downloads and then embeds a PDF in a non-blocking way."""
    async with httpx.AsyncClient() as client:
        try:
            print(f"Asynchronously downloading PDF from {pdf_url}...")
            response = await client.get(pdf_url, follow_redirects=True, timeout=30)
            response.raise_for_status()
            pdf_content = response.content
            print("Download complete.")
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")

    await asyncio.to_thread(_load_and_embed_sync, pdf_content, index_name)

async def search_similar_documents_async(query_vector: list, index_name: str) -> str:
    """Asynchronously searches Pinecone using its native async client."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        context_chunks = [match['metadata'].get('text', '') for match in results.get("matches", [])]
        return "\n\n---\n\n".join(context_chunks)
    except Exception as e:
        print(f"An error occurred during Pinecone search: {e}")
        return ""

async def answer_one_question_with_retry(
    question: str,
    query_vector: list,
    index_name: str,
    available_groq_keys: List[str]
) -> str:
    """
    Generates an answer for one question, retrying with different Groq keys on failure.
    """
    context = await search_similar_documents_async(query_vector, index_name)

    if not context.strip():
        return "This information is not specified in the policy document."

    prompt = f"""
    You are a helpful assistant with access to relevant document information.
    You must only answer questions based on the context below.
    If you cannot find an answer, respond with:
    "I could not find the answer in the provided document."

    Instructions:
    - Start directly with the answer — do NOT use phrases like "According to the document", "Based on the provided text", or "As per the document".
    - Be formal, concise, and accurate.
    - If there are multiple conditions, limits, or exceptions, structure them using bullets or short paragraphs.
    - If the answer is not found in the context, respond with:
    **"I could not find the answer for the specified document."**
    - Always avoid vague or hesitant phrases like “It seems”, “I think”, etc.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {question}

    ANSWER:
    """

    # Shuffle keys for this specific question
    shuffled_keys = random.sample(available_groq_keys, len(available_groq_keys))
    
    for i, key in enumerate(shuffled_keys):
        try:
            print(f"Attempting to answer '{question[:30]}...' with Groq Key #{i+1}")
            llm = ChatGroq(temperature=0, groq_api_key=key, model_name="llama3-8b-8192")
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error with Groq Key #{i+1} for question '{question}': {e}. Trying next key...")
            continue

    print(f"Error: All {len(available_groq_keys)} Groq API keys failed for question '{question}'.")
    return "The answer could not be generated due to an external service error."

@app.post("/hackrx/run", response_model=AnswerResponse)
async def hackrx_run(request: QueryRequest, authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key.")

    print("\n--- New Request Received ---")
    print(request)
    print()

    try:
        if request.documents == KNOWN_PDF_URL:
            index_name = KNOWN_INDEX_NAME
            print(f"Using known index: {index_name}")
        else:
            index_name = f"hackrx-index-{uuid.uuid4().hex[:8]}"
            print(f"Processing new document. Generated new index: {index_name}")
            await create_index_if_not_exists_async(index_name)
            await load_and_embed_pdf_async(request.documents, index_name)

        print("Asynchronously embedding questions...")
        query_vectors = await embeddings.aembed_documents(request.questions)

        print("Generating answers in parallel with API key rotation...")
        
        tasks = [
            answer_one_question_with_retry(q, v, index_name, GROQ_API_KEYS)
            for q, v in zip(request.questions, query_vectors)
        ]
        
        answers = await asyncio.gather(*tasks)

        print("--- Request Processing Done ---")
        return AnswerResponse(answers=answers)

    except Exception as e:
        print(f"An exception occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")