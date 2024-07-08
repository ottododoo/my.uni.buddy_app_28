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

# Load environment variables from the '.env' file
load_dotenv(".env")

# Set your API token directly as an environment variable
os.environ["HF_API_TOKEN"] = "your_hugging_face_api_token_here"

# Function to handle Hugging Face API requests with retry logic
def make_hf_request(llm, prompt, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = llm(prompt)
            if response:
                return response
            else:
                time.sleep(5)  # Sleep for 5 seconds before retrying
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit reached. Waiting for 10 seconds before retrying...")
                time.sleep(10)
            else:
                print(f"HTTP error occurred: {e}")
                raise  # Rethrow the exception for other HTTP errors
        except Exception as e:
            print(f"An error occurred: {e}")
            raise  # Rethrow any other unexpected exceptions

        retries += 1

    # If retries exceed max_retries without success, handle it accordingly
    print(f"Failed to get a valid response after {max_retries} retries.")
    return None

# Hugging Face model details
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model, api_token=os.getenv("HF_API_TOKEN"))

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
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})

##### Streamlit App #####

st.title("My.Unibuddy_Germany: Your Guide to Studying and living in Germany")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("Design ohne Titel (2).png")

with col2:
    st.image("Design ohne Titel (1).png")

with col3:
    st.image("Design ohne Titel.png")

st.sidebar.title("Popular Questions")
st.sidebar.markdown("""
- What are the requirements to study in Germany?
- How do I apply for a student visa for Germany?
- What is the cost of living for students in Germany?
- Are there scholarships available for international students in Germany?
- How can I find accommodation in Germany?
- What are the best universities in Germany for engineering?
- What is the application process for German universities?
- Can I work while studying in Germany?
- What are the career prospects after studying in Germany?
- How can I learn German effectively before moving?
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How may I help you?"):

    # Display user message in chat message container
    st.chat_messag
