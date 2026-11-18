# ============================================================
# rag.py
# Purpose: answer a question using the stored chunks
# Flow: question -> embed -> retrieve top-k chunks -> prompt -> LLM answer
# ============================================================

import os
import chromadb
import ollama                                # talk to the local Ollama model
from sentence_transformers import SentenceTransformer

# Same folder logic as ingest.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# IMPORTANT: use the SAME embedding model as in ingest.py,
# otherwise the question and the chunks won't be comparable.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen2.5:3b"                     # must match a name in `ollama list`

# Load these once when the file starts (loading is slow, so don't repeat it)
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model loaded")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("notes")
print(f"✅ Database connected ({collection.count()} chunks stored)")


# ------------------------------------------------------------
# Function 1: retrieve  (Step 4)
# Input : the user's question, and k = how many chunks to return
# Output: a list of (chunk_text, source_filename) pairs
# How   : the question is turned into numbers, then ChromaDB finds
#         the k stored chunks whose numbers are closest in meaning.
# ------------------------------------------------------------
def retrieve(question, k=3):
    query_embedding = embedder.encode([question]).tolist()   # question -> embedding
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,                                          # top-k closest chunks
    )
    docs = results["documents"][0]                            # chunk texts
    metas = results["metadatas"][0]                           # their source files
    return [(doc, meta["source"]) for doc, meta in zip(docs, metas)]


# ------------------------------------------------------------
# Function 2: build_prompt
# Input : the question and the retrieved chunks
# Output: one text prompt that tells the LLM to answer ONLY from the context
# ------------------------------------------------------------
def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source: {source}]\n{text}" for text, source in chunks
    )
    return (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context below. If the answer is not in the context, say "
        "\"I could not find this in the documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ------------------------------------------------------------
# Function 3: generate_answer  (Step 5)
# Input : the prompt
# Output: the LLM's answer as text
# ------------------------------------------------------------
def generate_answer(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ------------------------------------------------------------
# Function 4: ask
# Runs the full pipeline and prints a status message after each stage
# Output: (answer, chunks) so the UI can also show the sources
# ------------------------------------------------------------
def ask(question, k=3):
    chunks = retrieve(question, k)
    print(f"✅ Retrieval done: found {len(chunks)} relevant chunks")

    prompt = build_prompt(question, chunks)
    print("✅ Prompt built")

    print(f"Generating answer with {LLM_MODEL} (may take a few seconds on CPU)...")
    answer = generate_answer(prompt)
    print("✅ Generation done")

    return answer, chunks


# ------------------------------------------------------------
# Startup check: make sure everything is ready before asking questions
# Returns True if the pipeline can work, False otherwise
# ------------------------------------------------------------
def check_ready():
    # 1) The database must contain chunks
    if collection.count() == 0:
        print("❌ The database is empty. Run ingest.py first.")
        return False

    # 2) Ollama must be running and the model must exist
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        print("❌ Cannot reach Ollama. Open the Ollama app and try again.")
        return False

    if LLM_MODEL not in installed:
        print(f"❌ Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}")
        return False

    print(f"✅ Ollama is running and '{LLM_MODEL}' is available")
    return True


# ------------------------------------------------------------
# Quick command-line test: python rag.py
# ------------------------------------------------------------
if __name__ == "__main__":
    if check_ready():
        print("\n🎉 RAG is ready!")

        while True:
            q = input("\nAsk a question (or type 'exit'): ")
            if q.strip().lower() == "exit":
                break

            answer, chunks = ask(q)

            print("\nAnswer:\n", answer)
            print("\nSources:", sorted({source for _, source in chunks}))
            print("\n✅ RAG completed successfully")

        print("Goodbye!")