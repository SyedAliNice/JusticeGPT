import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
# load_dotenv()

# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Access secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]
groq_api = st.secrets["GROQ_API_KEY"]

# Configure APIs
genai.configure(api_key=google_api_key)
llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api)



# Must be the first Streamlit command
st.set_page_config(page_title="JusticeBot", page_icon="⚖️", layout="centered")

# Center the title using HTML
st.markdown("<h1 style='text-align: center; color: black;'>⚖️ JusticeBot</h1>", unsafe_allow_html=True)

# Background image via CSS (URL version)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t3.ftcdn.net/jpg/04/86/36/64/360_F_486366405_noZdxkmgs12lIkxydkIv57aTMfAb5sfe.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("<h4 style='text-align: center; color: #F8F8F8;'>Your Virtual Legal Assistant — Fair, Fast, and Smart</h4>", unsafe_allow_html=True)
st.markdown("---")

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily (if PyMuPDFLoader requires a file path)
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and split the document
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documents = text_splitter.split_documents(docs)
    
    
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()
    
    # Initialize LLM

    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api)
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant of a judge which will help me understand the partition and you will answers questions based on the provided partition document context and if you will not able to answer the question from file, look for the answer again and if you are still not able to answer the question, tell them the information is not provided in the document. Just must be accurate"),
        ("user", "Context: {context}\nQuestion: {question}")
    ])
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "verbose": True
        }
    )
    
    # User inputs their question
    input_text = st.text_input("Enter your question:")
    
    if input_text:
        with st.spinner("Searching for answer..."):
            result = qa_chain({"query": input_text})
        
        st.subheader("Answer:")
        st.write(result["result"])
