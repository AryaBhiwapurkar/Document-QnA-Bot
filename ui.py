import gradio as gr

from ingest import ingest_pdf
from chat import load_chain
from llm import get_embeddings

rag_chain = None


def upload_and_ingest(file):
    global rag_chain

    if file is None:
        return "No file uploaded."

    try:
        embeddings = get_embeddings()  # load once
        success = ingest_pdf(file.name, embeddings)
        if success:
            rag_chain = load_chain(embeddings)  # reuse
            return "Document ingested successfully! You can now ask questions."
        else:
            return "Failed to ingest document. Please try again."
    except Exception as e:
        return f"Error during ingestion: {str(e)}"
    
    

def respond(message):
    global rag_chain

    if rag_chain is None:
        return "Please upload a PDF document first."

    try:
        return rag_chain.invoke(message)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")  # log to console
        return "Something went wrong while generating the answer. Please try again."


with gr.Blocks() as ui:
    gr.Markdown("# Document Q&A Bot")
    gr.Markdown("Upload a PDF document and ask questions about it.")

    with gr.Row():
        file_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"]
        )
        ingest_status = gr.Textbox(
            label="Status",
            interactive=False
        )

    file_input.change(
        fn=upload_and_ingest,
        inputs=file_input,
        outputs=ingest_status
    )

    with gr.Row():
        question_input = gr.Textbox(
            placeholder="Ask your question...",
            label="Your Question"
        )
        answer_output = gr.Textbox(
            label="Answer",
            interactive=False
        )

    gr.Button("Ask").click(
        fn=respond,
        inputs=question_input,
        outputs=answer_output
    )


if __name__ == "__main__":
    ui.launch()