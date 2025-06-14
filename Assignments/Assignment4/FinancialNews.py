from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import operator
from typing import List
from pydantic import BaseModel , Field
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,END
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model and embeddings
model = ChatGroq(model="deepseek-r1-distill-llama-70b")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Loading the documents
loader = PyMuPDFLoader(r"Assignments/Assignment4/tsla_Q1.pdf")
docs = loader.load()

# split the documents into smaller chunks
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=75
)

chunks = text_splitter.split_documents(documents=docs)

# Create a vector store from the chunks
vector_store = Chroma.from_documents(
    documents=chunks
    ,embedding=embeddings
    ,persist_directory="Assignments/Assignment4/financial_news_vector_store"
)

retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 30}
)



