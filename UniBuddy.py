import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
hf_api_key = os.getenv('SECRET_NAME')

# Initialize Hugging Face endpoint with token in model_kwargs
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# Initialize embeddings model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "docs"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Load FAISS vector store
vector_db = FAISS.load_local("docs", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Define prompt template
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""
prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt_template})

# Streamlit App
st.title("My.Unibuddy_Germany: Your Guide to Studying and Living in Germany")

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
if user_input := st.chat_input("How may I help you?"):

    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Begin spinner before answering question
    with st.spinner("Uncovering knowledge about studying in Germany..."):

        try:
            # Send question to chain to get answer
            answer = chain(user_input)

            # Extract answer from dictionary returned by chain
            response = answer["answer"]

            # Display chatbot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            # Handle and log errors
            st.error(f"An error occurred: {e}")
            if hasattr(e, 'response'):
                st.error(f"Response: {e.response}")



