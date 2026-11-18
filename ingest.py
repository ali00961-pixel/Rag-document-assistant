# ============================================================
# ingest.py
# Purpose: read files -> split into chunks -> embed -> store
# Run it once, and again whenever you add new files to data/
# ============================================================

import os                                   # work with folders and file paths
import pymupdf as fitz                      # PyMuPDF: read PDF files
import chromadb                             # vector database
from sentence_transformers import SentenceTransformer  # text -> embeddings

# Folders are resolved relative to this file, so it works
# no matter where the terminal is opened.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")           # put your .pdf/.txt files here
DB_DIR = os.path.join(BASE_DIR, "chroma_db")        # the database is saved here


# ------------------------------------------------------------
# Function 1: load_text
# Input : path to a file
# Output: the full text of the file, or None if the type is unsupported
# ------------------------------------------------------------
def load_text(path):
    # lower() so that ".PDF" is accepted as well as ".pdf"
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)               # open the PDF
        # extract the text of every page and join them with newlines
        return "\n".join(page.get_text() for page in doc)

    if path.lower().endswith(".txt"):
        # errors="ignore" skips odd characters instead of crashing
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()                 # read the whole text file

    return None                             # unsupported type (image, docx, ...)


# ------------------------------------------------------------
# Function 2: split_text
# Input : full text, chunk size, overlap size
# Output: a list of small chunks
# Why split?   The model only needs the parts relevant to the question.
# Why overlap? So meaning isn't cut in half between two chunks.
# ------------------------------------------------------------
def split_text(text, size=500, overlap=100):
    chunks, start = [], 0                   # list of chunks + starting position
    while start < len(text):                # repeat until the end of the text
        chunk = text[start:start + size].strip()   # take 500 characters, trim spaces
        if chunk:                           # skip empty chunks
            chunks.append(chunk)
        start += size - overlap             # move forward 400 (500 - 100), so the
                                            # last 100 characters repeat in the next chunk
    return chunks


# ------------------------------------------------------------
# Main program: runs only when you execute this file directly
# ------------------------------------------------------------
if __name__ == "__main__":

    # Debug info: shows where the code looks and what it finds
    print("Looking in:", DATA_DIR)
    print("Files found:", os.listdir(DATA_DIR))

    # 1) Load the embedding model (turns text into numbers that capture meaning).
    #    It supports Arabic and English. Downloaded only the first time.
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 2) Create (or open) the ChromaDB database saved on disk
    client = chromadb.PersistentClient(path=DB_DIR)

    # 3) A "collection" is like a table. Created if it doesn't exist yet.
    col = client.get_or_create_collection("notes")

    total = 0                               # counter for all stored chunks

    # 4) Loop over every file inside the data folder
    for filename in os.listdir(DATA_DIR):

        # read the file text (os.path.join builds the full path correctly)
        text = load_text(os.path.join(DATA_DIR, filename))

        if not text:                        # unsupported type or empty file
            print(f"Skipped: {filename}")
            continue                        # go to the next file

        chunks = split_text(text)           # split the text into chunks

        # convert each chunk to an embedding (a list of numbers)
        # tolist() because ChromaDB needs plain lists, not numpy arrays
        embeddings = model.encode(chunks).tolist()

        # store in the database
        # upsert = add if new, update if it already exists (no duplicate errors)
        col.upsert(
            ids=[f"{filename}_{i}" for i in range(len(chunks))],  # unique ID per chunk
            documents=chunks,                                      # the original text
            embeddings=embeddings,                                 # the meaning as numbers
            metadatas=[{"source": filename}] * len(chunks),        # which file it came from
        )

        print(f"{filename}: {len(chunks)} chunks")
        total += len(chunks)

    print(f"Stored in ChromaDB. Total: {total} chunks")