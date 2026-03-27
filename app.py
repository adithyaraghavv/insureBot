import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def ask_question(pdf_file, question, history):
    if pdf_file is None:
        history.append(("", "⚠️ Please upload an insurance PDF first!"))
        return history
    try:
        vectorstore = process_pdf(pdf_file)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            api_key=os.environ["GROQ_API_KEY"]
        )

        prompt = f"""You are InsureBot, an expert AI assistant for insurance policies.
Use the context below to answer the question clearly and simply.
If the answer is not in the document, say "This information is not found in the uploaded policy."

Context: {context}

Question: {question}

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        history.append((question, answer))
        return history

    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))
        return history

with gr.Blocks(theme=gr.themes.Soft(), title="InsureBot") as app:

    gr.Markdown("""
    # 🏥 InsureBot — AI Insurance Policy Assistant
    **Upload any insurance policy PDF → Ask questions → Get instant AI answers**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(
                label="📄 Upload Insurance Policy PDF",
                file_types=[".pdf"]
            )
            gr.Markdown("""
            **Try asking:**
            - What is covered under this policy?
            - What is the claim process?
            - What are the exclusions?
            - What is the sum insured?
            - What is the waiting period?
            """)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="InsureBot 🤖", height=400)
            question_box = gr.Textbox(
                placeholder="Ask anything about your policy...",
                label="Your Question"
            )
            with gr.Row():
                submit_btn = gr.Button("Ask 🤖", variant="primary")
                clear_btn = gr.Button("Clear 🗑️")

    submit_btn.click(
        fn=ask_question,
        inputs=[pdf_upload, question_box, chatbot],
        outputs=[chatbot]
    )
    question_box.submit(
        fn=ask_question,
        inputs=[pdf_upload, question_box, chatbot],
        outputs=[chatbot]
    )
    clear_btn.click(lambda: [], outputs=[chatbot])

app.launch()
