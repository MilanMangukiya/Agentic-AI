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
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

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

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]

# Define the Pydantic model for the output parser 
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="reasoning behind topic selection")

parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)

def supervisor_node(state:AgentState):
    
    question=state["messages"][-1]

    print("Question received:", question)
    
    template="""
    Your task is to classify the given user query into one of the following topic:    
    [Financial report of Tesla Q1 earning, real-time updates, Not Related].

    Use the following guidelines to classify the topic for the query:
    1) If information is related to Tesla's Q1 earnings report, classify it under "Financial report of Tesla Q1 earning". 
    2) If the information is about real-time updates or recent information or online search-based or latest information classify it under "real-time updates".
    3) If the information is not related to Tesla's Q1 earnings report or real-time updates, classify it under "Not Related".

    Respond ONLY in valid JSON with the following keys:
    - Topic: the category name
    - Reasoning: the reasoning behind the topic selection

    User query: {question}
    {format_instructions}
    """
    
    prompt= PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain= prompt | model | parser
    
    response = chain.invoke({"question":question})
    
    print("Response from supervisor node:", response)

    print("Selected Topic:", response.Topic)
    return {"messages: ": [response.Topic]}


def router(state: AgentState):
    last_message = state["messages"][0]
    print("last_message:", last_message)
    if last_message == "Financial report of Tesla Q1 earning":
        print("Routing to RAG node")
        return "RAG"
    elif last_message == "real-time updates":
        return "Web"
    else:
        return "LLM"
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Function
def RAG(state:AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][0]
    
    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        
        input_variables=['context', 'question']
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return  {"messages": [result]}

# LLM Function
def LLM(state:AgentState):
    print("-> LLM Call ->")
    question = state["messages"][0]
    
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [response.content]}
    
def WEB(state: AgentState):
    print("-> WEB Call ->")
    query = state["messages"][0]
    print(f"state: {state}")
    tool = TavilySearchResults()
    response = tool.invoke({"query": query})
    print(f"response from web: {response}")
    return {"messages": [response[0].get("content")]}

def validator_node(state: AgentState):
    print("-> Validator Node ->")

    question = state['messages'][0]
    response = state['messages'][-1]

    prompt = PromptTemplate.from_template(
            template="""
            Your task is to evaluate whether the provided response is suitable for the given user question.
            If it is suitable, respond with Yes; otherwise, No. Follow the instrustions while giving response 
            just give what required in instructions dont give additional info
             
            Question: {question}
            Response: {response}
            instructions:{format_instructions}
            """,
            input_variable=["question", "response"],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )

    # Combine the chain
    chain = prompt | model | parser

    # Invoke with format instructions passed
    res = chain.invoke({
        'question': question,
        'response': response,
    })
    print(res)
    return {"messages": [response]}

# def retry_router(state: AgentState):
#     print(f"state: {state}")
#     if state["messages"][-1] == "__RETRY__":
#         return "Supervisor"
#     return "Final"

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Router", router)
workflow.add_node("RAG", RAG)
workflow.add_node("LLM", LLM)
workflow.add_node("Web", WEB)
workflow.add_node("Validate", validator_node)
# workflow.add_node("Final", retry_router)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor", 
    router, 
    {"RAG": "RAG", 
     "LLM": "LLM", 
     "Web": "Web"}
)

workflow.add_edge("RAG", "Validate")
workflow.add_edge("LLM", "Validate")
workflow.add_edge("Web", "Validate")

# workflow.add_conditional_edges(
#     "Validate", retry_router, {"Supervisor": "Supervisor", "Final": "Final"}
# )

workflow.set_finish_point("Validate")

app = workflow.compile()


# state={"messages":["What is AI?"]}
state={"messages":["Tell me about Ahmedabad plane crash as of 12th July 2025?"]}
# state={"messages":["What are some key highlights of Tesla Q1 2025?"]}
supervisor_node(state)

result = app.invoke(state)
print("Final Result:", result["messages"][-1])
