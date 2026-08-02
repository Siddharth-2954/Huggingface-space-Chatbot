from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

st.set_page_config(page_title="Conversation RAG with PDF + Message History GROQ")

st.title("Conversation RAG with PDF + Message History GROQ")

api_key = st.text_input("Enter your GROQ API Key", type="password")

if api_key:

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if uploaded_file:

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Loading PDF...")

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        st.success(f"Loaded {len(documents)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        st.success(f"Created {len(chunks)} text chunks")

        st.info("Loading Embedding Model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.success("Embeddings Loaded")

        st.info("Creating Vector Database...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        st.success("Vector Database Created")

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant.

Answer the question only from the provided context.

If the answer is not available in the context, say:

"I don't know based on the uploaded PDF."

Context:
{context}

Question:
{input}
"""
        )

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        question = st.text_input("Ask a question from the PDF")

        if question:

            with st.spinner("Generating Answer..."):

                response = retrieval_chain.invoke(
                    {"input": question}
                )

            st.subheader("Answer")

            st.write(response["answer"])

            with st.expander("Retrieved Context"):

                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")

    else:
        st.info("Please upload a PDF.")

else:
    st.warning("Please enter your GROQ API Key.")