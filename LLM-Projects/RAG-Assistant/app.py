# ============================================================
# app.py
# Purpose: a simple web interface (Gradio) for the RAG assistant
# Run with: python app.py  -> opens at http://127.0.0.1:7860
# ============================================================

import gradio as gr
from rag import ask, check_ready      # reuse the pipeline from rag.py


# ------------------------------------------------------------
# Function: answer_question
# Input : the question typed by the user
# Output: (answer text, sources text) shown in the two output boxes
# ------------------------------------------------------------
def answer_question(question):
    if not question.strip():                      # ignore empty input
        return "Please type a question.", ""

    answer, chunks = ask(question)                # run the full RAG pipeline

    # Show each source file and a short preview of the chunk that was used
    sources = "\n\n".join(
        f"📄 {source}\n{text[:200]}..." for text, source in chunks
    )
    return answer, sources


# ------------------------------------------------------------
# Build the interface
# ------------------------------------------------------------
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Your question",
        placeholder="e.g. What is a neural network?",
        lines=2,
    ),
    outputs=[
        gr.Textbox(label="Answer", lines=8),
        gr.Textbox(label="Sources (retrieved chunks)", lines=10),
    ],
    title="📚 RAG Assistant",
    description="Ask questions about your documents. Answers come only from the files in the data folder.",
    examples=[
        ["What is deep learning?"],
        ["What is a neural network?"],
    ],
    flagging_mode="never",                        # hide the Flag button
)

# ------------------------------------------------------------
# Start the app (only when this file is run directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    if check_ready():                             # make sure DB and Ollama are OK
        print("🚀 Starting the web app...")
        demo.launch()