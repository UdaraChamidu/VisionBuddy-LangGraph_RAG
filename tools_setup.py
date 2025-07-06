# tools_setup.py

from langchain_core.tools import tool
from duckduckgo_search import DDGS

def create_tools(retriever):
    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns the information from the ophthalmology PDF document using vector similarity.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the Eye Disease document."
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    @tool
    def search_tool(query: str) -> str:
        """
        Searches the internet for the user's query using DuckDuckGo and returns summaries of top results.
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=8)
                if not results:
                    return "No internet search results found."
                return "\n\n".join(
                    f"Result {i}:\nTitle: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}"
                    for i, r in enumerate(results, 1)
                )
        except Exception as e:
            return f"Error during DuckDuckGo search: {str(e)}"

    return [retriever_tool, search_tool]
