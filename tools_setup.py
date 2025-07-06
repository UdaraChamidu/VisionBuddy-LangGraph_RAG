from langchain_core.tools import tool
from duckduckgo_search import DDGS
import feedparser
import datetime
import requests

def create_tools(retriever):
    
    # 📘 Tool 1: PDF Retriever (already included)
    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns the information from the ophthalmology PDF document using vector similarity.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the Eye Disease document."
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    # 🌐 Tool 2: DuckDuckGo Internet Search
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

    # 📰 Tool 3: Medical News Tool
    @tool
    def medical_news_tool(topic: str) -> str:
        """
        Fetches the latest medical news headlines related to a topic (e.g., eye disease, glaucoma) from RSS feeds.
        """
        feeds = [
            "https://www.medscape.com/rss/public",
            "https://www.webmd.com/rss/news_breaking.xml",
            "https://www.sciencedaily.com/rss/health_medicine.xml"
        ]
        results = []
        for url in feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if topic.lower() in entry.title.lower() or topic.lower() in entry.summary.lower():
                    results.append(f"📰 {entry.title}\n{entry.link}")
                if len(results) >= 5:
                    break
            if len(results) >= 5:
                break
        return "\n\n".join(results) if results else "No recent medical news found for this topic."

    # 🧠 Tool 4: WHO Disease Info Tool (from open WHO APIs)
    @tool
    def who_disease_info(disease: str) -> str:
        """
        Provides general disease info from the World Health Organization. Best for known conditions like cataract, glaucoma.
        """
        try:
            url = f"https://www.who.int/api/search?q={disease}"
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                return "WHO API could not be reached."
            data = res.json()
            entries = data.get("results", [])[:3]
            if not entries:
                return "No WHO information found."
            return "\n\n".join([f"🔹 {e['title']}\n{e['link']}" for e in entries])
        except Exception as e:
            return f"Error accessing WHO data: {e}"

    # 🧰 Final tool list
    return [
        retriever_tool,
        search_tool,
        medical_news_tool,
        who_disease_info,
    ]

    
