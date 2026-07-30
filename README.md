---
title: Groq Chatbot Query App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.48.1"
python_version: "3.11"
app_file: rag_pdf.py
pinned: false
---

# Groq Chatbot Query App

A Streamlit chatbot powered by Groq and LangChain.

# PART 1 — PDF Ingestion & Preprocessing

# File Structure
- `file.ipynb` and `rag_pdf.py` consists of all Tasks.

## Task 1: Load PDF Documents

- Loaded the PDF documents using a LangChain PDF loader.

- Counted and displayed the total number of pages in the PDF.

- Printed a sample of the text from one page to verify successful loading.

- Confirmed that the document content is ready for further preprocessing and analysis.

## Task 2: Text Splitting

- Used RecursiveCharacterTextSplitter to divide the PDF text into smaller chunks.

- Configured the chunk_size and chunk_overlap values for better text segmentation.

- Split the document into multiple overlapping chunks to preserve context.

- Verified that the generated text chunks are ready for embedding and retrieval.

# PART 2 — Embeddings & Vector Store

## Task 3: Create Embeddings

- Selected an embedding model (OpenAI, HuggingFace, or Ollama) to convert text into vector representations.

- Generated embeddings for each text chunk created from the PDF documents.

- Stored the embeddings so that similar text can be searched efficiently.

- Verified that all PDF chunks were successfully converted into embeddings for use in the vector database.

## Task 4: Vector Store Setup

- Stored the generated embeddings in a vector database such as FAISS or ChromaDB.

- Indexed all document chunks to enable fast and efficient similarity searches.

- Created a retriever from the vector store for retrieving the most relevant document chunks.

- Verified that the retriever successfully returns relevant chunks for user queries.

# PART 3 — Conversational Prompt with Message History

## Task 5: RAG Prompt Template

- Created a ChatPromptTemplate with a system message, chat history placeholder, and the current user question.

- Configured the prompt to answer questions using only the retrieved PDF context.

- Added a MessagesPlaceholder to maintain conversation history for follow-up questions.

# PART 4 — Conversational RAG Chain

## Task 6: Build Conversational RAG Chain

- Built a conversational RAG pipeline that starts with the user’s question.

- Used the retriever to fetch the most relevant PDF chunks from the vector store.

- Passed the retrieved PDF context and chat history into the prompt template.

- Sent the final prompt to the LLM and generated an answer based only on the retrieved PDF content.

## Task 7: Maintain Message History

- Stored both user questions and AI responses after each conversation turn.

- Updated the chat history continuously to preserve the conversation context.

- Injected the message history into the prompt using MessagesPlaceholder.

- Enabled the chatbot to answer follow-up questions using previous conversation context.

## Task 8: Trimming Chat History

- Limited the conversation history by setting a maximum number of messages or token length.

- Monitored the chat history before sending it to the language model.

- Removed the oldest user and AI messages when the history exceeded the configured limit.

# PART 5 — Multi-Turn Conversation Testing

## Task 9: Follow-Up Q&A Testing

- Asked follow-up and clarification questions to verify that the conversation history was used correctly.

- Confirmed that the chatbot preserved context across multiple conversation turns.

- Verified that all responses remained grounded in the retrieved PDF content and returned "I don't know" when the information was not available in the document.

- ![Chatbot UI](images/image.png)
- ![Chatbot UI](images/image1.png)
- ![Chatbot UI](images/image2.png)
- ![Chatbot UI](images/image3.png)
- ![Chatbot UI](images/image4.png)
- ![Chatbot UI](images/image5.png)
- ![Chatbot UI](images/image6.png)
- ![Chatbot UI](images/image12.png)
- ![Chatbot UI](images/image13.png)
- ![Chatbot UI](images/image8.png)
- ![Chatbot UI](images/image10.png)

# PART 6 — Mini Project: Conversational PDF Chatbot

## Task 10: Build Final Chatbot Application

- Built a conversational PDF chatbot that accepts user questions through a simple chat interface.

- Retrieved the most relevant PDF chunks using a vector store before generating each response.

- Maintained conversation history to understand follow-up and context-based questions.

- Generated accurate answers using only the retrieved PDF content and replied with "I don't know" when the information was not found.

## Task 11: Observations & Insights

- Questions and Answers.


## 🚀 How to Open and Run the Notebook

### Step 1: Open the Notebook

Using VS Code
1. Open VS Code
2. Click on `file.ipynb` to open it 
3. Create a virtual environment using command prompt `conda create --name venv python=3.10`.
4. Open Command Prompt and run command `pip install -r requirements.txt`.
5. Select a Python interpreter.

### Step 2: Run All Cells

#### Run All Cells at Once
1. Click the **"Run All"** button in the toolbar (or use keyboard shortcut `Ctrl+Shift+Enter`)
2. Wait for all cells to execute

#### Run Cells Sequentially
1. Click the first cell (imports cell)
2. Press `Shift+Enter` to run the cell and move to the next
3. Continue pressing `Shift+Enter` for each cell

#### Option C: Run Specific Task
1. Locate the task you want to run
2. Click on any cell within that task
3. Press `Shift+Enter` to run the cell
4. Continue running dependent cells in sequence.