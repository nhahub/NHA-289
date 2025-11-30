from langchain.tools import Tool
from ddgs import DDGS

def search_images(query: str, max_results: int = 5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.images(query):
            results.append({"title": r["title"], "url": r["image"]})
            if len(results) >= max_results:
                break
    return results

image_search_tool = Tool(
    name="Image Search",
    func=search_images,
    description="Search images on DuckDuckGo and return top 5 results"
)
