import asyncio
from pydantic import BaseModel, Field
from agentory import Agent
from agentory.tools import Tools
from agentory.tools.di import Inject


class NotionClient:
    async def search_pages(self, query: str) -> list[dict]:
        return [
            {
                "id": "abc123",
                "title": "Meeting Notes",
                "url": "https://notion.so/abc123",
            },
            {
                "id": "def456",
                "title": "Project Roadmap",
                "url": "https://notion.so/def456",
            },
        ]

    async def get_page(self, page_id: str) -> dict:
        return {
            "id": page_id,
            "content": "# Meeting Notes\n\nTodo: Fix bug, Deploy service",
        }

    async def create_page(self, title: str, content: str) -> dict:
        return {"id": "new123", "title": title, "url": "https://notion.so/new123"}


class SearchPagesParams(BaseModel):
    query: str = Field(description="Search query to find pages in Notion")


class GetPageParams(BaseModel):
    page_id: str = Field(description="The Notion page ID to retrieve")


class CreatePageParams(BaseModel):
    title: str = Field(description="Title of the new page")
    content: str = Field(description="Markdown content of the new page")


notion_tools = Tools()


@notion_tools.action("Search for pages in Notion", params=SearchPagesParams)
async def search_pages(
    params: SearchPagesParams,
    client: Inject[NotionClient],
) -> str:
    results = await client.search_pages(params.query)
    return "\n".join(f"- [{r['title']}]({r['url']}) (id: {r['id']})" for r in results)


@notion_tools.action("Get the content of a Notion page by ID", params=GetPageParams)
async def get_page(
    params: GetPageParams,
    client: Inject[NotionClient],
) -> str:
    page = await client.get_page(params.page_id)
    return page["content"]


@notion_tools.action("Create a new Notion page", params=CreatePageParams)
async def create_page(
    params: CreatePageParams,
    client: Inject[NotionClient],
) -> str:
    page = await client.create_page(params.title, params.content)
    return f"Created page '{page['title']}' → {page['url']}"


# --- Agent ---


async def main():
    from llmify import ChatOpenAI  # o.ä. – je nach llmify-Impl.
    from dotenv import load_dotenv

    load_dotenv()

    agent = Agent(
        instructions="You are a Notion assistant. Help the user manage their Notion workspace.",
        llm=ChatOpenAI(model="gpt-5.4-mini"),
        tools=notion_tools,
        injectables=[NotionClient()],
    )

    async for event in agent.run(
        "Find pages about meetings and summarize what's in the first one"
    ):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
