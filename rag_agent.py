# rag_agent.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
#from vector_store import load_or_create_vectorstore
from tools_setup import create_tools
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

# Vector Store Load
persist_directory = os.path.join("C:\\VisionBuddy-LangGraph_RAG", "chroma_db")
collection_name = "vectorstore"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM and tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = create_tools(retriever)
llm = llm.bind_tools(tools)
tools_dict = {tool.name: tool for tool in tools} 

# Agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an AI assistant for clinical ophthalmology.
- Use retriever_tool to check the PDF first.
- If not found, use search_tool.
Always cite your source.
"""

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    return {'messages': [response]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        name = t['name']
        args = t['args'].get('query', '')
        print(f"⚙️ Calling tool: {name} with query: {args}")
        if name in tools_dict:
            result = tools_dict[name].invoke(args)
        else:
            result = "Tool not found."
        results.append(ToolMessage(tool_call_id=t['id'], name=name, content=str(result)))

    return {'messages': results}

# Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "tool_agent", False: END})
graph.add_edge("tool_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# Conversation loop
def running_agent():
    print("\n=== 🧠 RAG Agent with Memory ===")
    messages = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "/save":
            with open("chat_history.txt", "w", encoding="utf-8") as f:
                for m in messages:
                    role = "You" if isinstance(m, HumanMessage) else "AI"
                    f.write(f"{role}: {m.content}\n")
            print("💾 Chat saved to chat_history.txt")
            continue

        messages.append(HumanMessage(content=user_input))
        result = rag_agent.invoke({"messages": messages})
        reply = result['messages'][-1]
        print(f"\nAI: {reply.content}")
        messages.append(reply)

if __name__ == "__main__":
    running_agent()
