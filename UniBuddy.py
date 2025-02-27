import time
import requests
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API token securely
hf_api_token = os.getenv("HF_API_TOKEN")

# Hugging Face model details
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model, api_token=hf_api_token)

# Embedding model details
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "docs"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Load FAISS vector store
vector_db = FAISS.load_local("docs", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Prompt template
template = """You are a chatbot answering questions based on context and chat history.

Previous conversation:
{chat_history}

Context:
{context}

New human question: {question}
Response:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})

##### Streamlit App #####

st.title("My.Unibuddy_Germany: Your Guide to Studying and Living in Germany")

st.sidebar.title("Popular Questions")
st.sidebar.markdown("""
- What are the requirements to study in Germany?
- How do I apply for a student visa for Germany?
- What is the cost of living for students in Germany?
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("How may I help you?"):

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Fetching answer..."):
        try:
            answer = chain.invoke({"question": user_input, "chat_history": st.session_state.messages})
            response = answer["answer"] if answer else "Sorry, I couldn't find an answer."

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error: {e}")



