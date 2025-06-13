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
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import re

def clean_text_list(docs):
    cleaned_list = []
    for doc in docs:
        # Extract text content from Document object
        text = doc.page_content  # or doc.content depending on your Document class
        # Now clean the raw text string
        text = text.replace('\\t', ' ')
        text = text.replace('\\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        cleaned_list.append(text)
    return cleaned_list

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model and embeddings
model = ChatGroq(model="deepseek-r1-distill-llama-70b")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Loading the documents
loader=DirectoryLoader("C:\Projects-Agentic\Agentic AI\Agentic-AI\Assignments\Assignment4",glob="*.pdf",loader_cls=PyPDFLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")
# print(docs)
print(type(docs))

clean_text = clean_text_list(docs)
for item in clean_text:
    print(item) 

# split the documents into smaller chunks
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

