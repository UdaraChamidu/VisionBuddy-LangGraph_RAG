from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain.tools import BaseTool
from duckduckgo_search import DDGS
import requests

load_dotenv()  

llm = ChatOpenAI(
    model="gpt-4o", temperature=0
)  # minimize hallucination

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

pdf_path = "Kanski’s clinical ophthalmology _ a systematic approach.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

pages_split = text_splitter.split_documents(pages)

persist_directory = os.path.join("C:\\VisionBuddy-LangGraph_RAG", "chroma_db")
collection_name = "vectorstore"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# --- VECTORSTORE CREATION / LOADING ---

def vectorstore_exists(persist_dir: str) -> bool:
    # This checks if any file/folder exists in the persist_directory (better than just os.listdir)
    expected_files = ["chroma-collections.parquet", "chroma-embeddings.parquet", "index"]
    return all(os.path.exists(os.path.join(persist_dir, f)) for f in expected_files)

if vectorstore_exists(persist_directory):
    print("Loading existing Chroma vector store...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma vector store from documents...")
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Saving vector store to disk...")
    vectorstore.persist()

print("Vector store ready!")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the clinical ophthalmology textbook.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Eye Disease document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

@tool
def search_tool(query: str) -> str:
    """
    Searches the internet for the user's query using DuckDuckGo.
    Returns a detailed summary of the top results.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=8)
            if not results:
                return "No internet search results found."

            summaries = []
            for i, res in enumerate(results, 1):
                summaries.append(
                    f"Result {i}:\nTitle: {res['title']}\nSnippet: {res['body']}\nURL: {res['href']}"
                )
            return "\n\n".join(summaries)
    except Exception as e:
        return f"Error during DuckDuckGo search: {str(e)}"

tools = [retriever_tool, search_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about clinical ophthalmology based primarily on the PDF document.
- First, attempt to get information from the PDF using retriever_tool.
  If you do, answer with: "According to the book, ..." and cite the PDF text.
- If the PDF has no relevant info, call the internet search tool and then answer with: "From the internet, ..." and cite the search results.
Always call the appropriate tool when needed.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}  
   
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT with Memory ===")
    
    messages = []  # persistent message history
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("\n👋 Exiting conversation. See you!")
            break

        if user_input.lower() == "/save":
            with open("chat_history.txt", "w", encoding="utf-8") as f:
                for msg in messages:
                    role = "You" if isinstance(msg, HumanMessage) else "AI"
                    f.write(f"{role}: {msg.content}\n")
            print("💾 Chat saved to chat_history.txt")
            continue  # go to next input without running LLM

        messages.append(HumanMessage(content=user_input))  # add user input

        result = rag_agent.invoke({"messages": messages})

        ai_reply = result['messages'][-1]
        print(f"\nAI: {ai_reply.content}")

        messages.append(ai_reply)  # add AI reply

running_agent()




