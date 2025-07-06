# tools_setup.py

from langchain_core.tools import tool
from duckduckgo_search import DDGS
import feedparser
import requests
from datetime import datetime

# 🔍 Tool 1: Search Tool (Internet)
@tool
def search_tool(query: str) -> str:
    """Searches the internet for general queries using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            if not results:
                return "No results found."
            return "\n\n".join(
                f"🔎 {r['title']}\n{r['body']}\n{r['href']}" for r in results
            )
    except Exception as e:
        return f"Error during search: {e}"

# 📰 Tool 2: Medical News Tool
@tool
def medical_news_tool(topic: str) -> str:
    """Fetches recent medical news headlines for a given topic."""
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
    return "\n\n".join(results) if results else "No recent medical news found."

# 🧠 Tool 3: WHO Disease Info
@tool
def who_disease_info(disease: str) -> str:
    """Provides disease information from the World Health Organization."""
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

# 📅 Tool 4: Today’s Date
@tool
def today_tool(_: str) -> str:
    """Returns today’s date."""
    return datetime.now().strftime("Today is %A, %d %B %Y")

# ✅ Custom Retriever Tool
def create_tools(retriever):
    @tool
    def retriever_tool(query: str) -> str:
        """Searches the PDF document for eye disease-related information."""
        docs = retriever.invoke(query)
        if not docs:
            return "No information found in the PDF."
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    # Return retriever + global tools
    return [
        retriever_tool,
        search_tool,
        medical_news_tool,
        who_disease_info,
        today_tool
    ]
