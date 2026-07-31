from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chains import (create_retrieval_chain, create_history_aware_retriever)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoTokenizer

load_dotenv()

st.title("Conversation RAG with PDF + Message History GROQ")

api_key = st.text_input("Provide with the GROQ API Key", type='password')

if api_key:
    model = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=api_key)

    session_id = st.text_input("Please Provide Your Session ID", value='default-session')

    if 'store' not in st.session_state:
        st.session_state.store={}
        
    uploaded_file = st.file_uploader("Upload Your PDFs: ", type='pdf')

    if uploaded_file:
        temp_pdf="./temporary.pdf"
        with open(temp_pdf, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Load PDF
        loader = PyPDFLoader(temp_pdf)
        st.write("Loading PDF...")
        documents = loader.load()
        st.success(f"Loaded {len(documents)} pages")

    
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

        st.write("Splitting...")
        splits = splitter.split_documents(documents)
        st.success(f"Created {len(splits)} chunks")

        st.write("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.success("Embeddings loaded")

        st.write("Creating Chroma DB...")
        vector = Chroma.from_documents(
            splits,
            embedding=embeddings
        )

        st.success("Vector DB created")
        
        retriever = vector.as_retriever()

        # Code for storing history

        contextualize_system_prompt= '''Given a chat history and the latest user question, reforumlate the question to make it standalone. Do not answer only rephrase.'''

        contextualize_prompt = ChatPromptTemplate.from_messages([('system', contextualize_system_prompt), MessagesPlaceholder('chat_history'), ('human', '{input}')])

        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_prompt)

        system_prompt = '''You are an helpful AI assistant. For Question Answer from the provided  context only. If unsure, Say I don't know context 
        {context}
        '''

        qa_prompt = ChatPromptTemplate.from_messages([('system', system_prompt), MessagesPlaceholder('chat_history'), ('human', '{input}')])

        document_chain = create_stuff_documents_chain(model, qa_prompt)

        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        MAX_TOKENS = 1200

        MAX_MESSAGES = 6

        def trim_history_by_tokens(history):

            total_tokens = 0
            trimmed_history = []

            for message in reversed(history.messages):
                tokens = len(tokenizer.encode(message.content))
                if total_tokens + tokens > MAX_TOKENS:
                    break

                total_tokens += tokens
                trimmed_history.insert(0, message)

            history.messages = trimmed_history

            return total_tokens


        def get_session_history(session_id):

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            history = st.session_state.store[session_id]

            total_tokens = trim_history_by_tokens(history)

            st.sidebar.write(f"Current Tokens : {total_tokens}")

            return history
        
        conversational_rag_chain = RunnableWithMessageHistory(retrieval_chain, get_session_history, input_messages_key='input', history_messages_key='chat_history', output_messages_key='answer')



        user_input = st.text_input("Ask a question about you Pdf: ")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({'input':user_input}, config={'configurable': {'session_id': session_id}})

            with st.expander("Retrieved Context"):
                if "context" in response:
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("----------------")

            st.subheader("Assistant Answer: ")

            st.write(response['answer'])

            with st.expander("Chat History"):
                st.write(f"Stored messages: {len(session_history.messages)} / {MAX_MESSAGES}")
                st.write(session_history.messages)

    else:
        st.info("Please upload a PDF.")

else:
    st.warning("Please Enter Your API Key to continue")    