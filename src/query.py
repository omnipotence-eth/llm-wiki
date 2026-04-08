"""LangGraph query pipeline: search → retrieve → synthesize → optionally persist."""

from __future__ import annotations

import logging
from datetime import date
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.llm import complete_structured
from src.models import QueryAnswer, WikiFrontmatter, WikiPage
from src.search import WikiIndex
from src.wiki import read_all_pages, title_to_path, write_page

logger = logging.getLogger(__name__)

# ── State ────────────────────────────────────────────────────────────────────


class QueryState(TypedDict, total=False):
    question: str
    search_results: list[WikiPage]
    retrieved_context: str
    answer: str
    citations: list[str]
    follow_up_queries: list[str]
    should_persist: bool
    persisted_page: str
    errors: list[str]


# ── Node functions ───────────────────────────────────────────────────────────


async def search_wiki_node(state: QueryState) -> QueryState:
    """Search the wiki using BM25."""
    settings = get_settings()
    pages = read_all_pages(settings.wiki_dir)

    if not pages:
        return {"search_results": [], "errors": ["wiki is empty — ingest sources first"]}

    index = WikiIndex(pages)
    results = index.search(state["question"], top_k=5)

    logger.info("search results=%d for query=%r", len(results), state["question"])
    return {"search_results": results}


async def retrieve_pages_node(state: QueryState) -> QueryState:
    """Build context string from search results."""
    results = state.get("search_results", [])
    if not results:
        return {"retrieved_context": "", "errors": state.get("errors", []) + ["no search results"]}

    context_parts = []
    for page in results:
        context_parts.append(
            f"## {page.frontmatter.title}\n"
            f"Type: {page.frontmatter.page_type.value} | "
            f"Confidence: {page.frontmatter.confidence.value}\n\n"
            f"{page.body}"
        )

    context = "\n\n---\n\n".join(context_parts)
    logger.info("retrieved context pages=%d", len(results))
    return {"retrieved_context": context}


async def synthesize_node(state: QueryState) -> QueryState:
    """Call LLM to synthesize an answer from retrieved context."""
    context = state.get("retrieved_context", "")
    if not context:
        return {"errors": state.get("errors", []) + ["no context for synthesis"]}

    settings = get_settings()
    question = state["question"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge wiki assistant. Answer questions using the provided "
                "wiki pages as context. Cite the wiki page titles you use. If your answer "
                "synthesizes information across multiple pages in a novel way, set "
                "should_persist=true and provide a synthesis_page."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nWiki context:\n\n{context}",
        },
    ]

    try:
        result = await complete_structured(
            messages=messages,
            response_model=QueryAnswer,
            temperature=settings.query_temperature,
        )

        return {
            "answer": result.answer,
            "citations": result.citations,
            "follow_up_queries": result.follow_up_queries,
            "should_persist": result.should_persist,
        }
    except Exception as exc:
        logger.error("synthesis failed", exc_info=True)
        return {"errors": state.get("errors", []) + [f"synthesis failed: {exc}"]}


async def persist_synthesis_node(state: QueryState) -> QueryState:
    """Persist the synthesis as a new wiki page if LLM decided it's worth saving."""
    settings = get_settings()
    question = state["question"]
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    today = date.today()

    # Create a synthesis page from the answer
    title = f"Synthesis: {question[:60]}"
    fm = WikiFrontmatter(
        title=title,
        page_type="synthesis",
        sources=[],
        tags=["synthesis", "query-generated"],
        created=today,
        updated=today,
        confidence="medium",
        related=[title_to_path(c) for c in citations],
    )

    body = f"# {title}\n\n{answer}\n\n## Sources\n"
    for cite in citations:
        body += f"- [[{cite}]]\n"

    path = write_page(fm, body, settings.wiki_dir)
    logger.info("persisted synthesis path=%s", path)
    return {"persisted_page": path}


# ── Graph ────────────────────────────────────────────────────────────────────


def _should_retrieve(state: QueryState) -> str:
    if state.get("errors"):
        return END
    return "retrieve_pages"


def _should_synthesize(state: QueryState) -> str:
    if state.get("errors"):
        return END
    return "synthesize"


def _should_persist(state: QueryState) -> str:
    """Conditional edge: persist only if LLM decided the synthesis is novel."""
    if state.get("errors"):
        return END
    if state.get("should_persist", False):
        return "persist_synthesis"
    return END


def build_query_graph() -> StateGraph:
    """Build the LangGraph query pipeline."""
    graph = StateGraph(QueryState)

    graph.add_node("search_wiki", search_wiki_node)
    graph.add_node("retrieve_pages", retrieve_pages_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("persist_synthesis", persist_synthesis_node)

    graph.set_entry_point("search_wiki")
    graph.add_conditional_edges("search_wiki", _should_retrieve)
    graph.add_conditional_edges("retrieve_pages", _should_synthesize)
    graph.add_conditional_edges("synthesize", _should_persist)
    graph.add_edge("persist_synthesis", END)

    return graph


async def run_query(question: str) -> QueryState:
    """Run the full query pipeline."""
    graph = build_query_graph()
    app = graph.compile()

    initial_state: QueryState = {
        "question": question,
        "errors": [],
    }

    result = await app.ainvoke(initial_state)
    return result
