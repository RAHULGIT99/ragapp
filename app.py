import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

# Load the same model you were using locally
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbedResponse)
def embed_texts(req: EmbedRequest):
    try:
        embeddings = model.encode(req.texts, convert_to_numpy=False)
        return {"embeddings": [embedding.tolist() for embedding in embeddings]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
