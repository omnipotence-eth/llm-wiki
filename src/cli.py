"""Click CLI entry point: wiki ingest, query, lint, stats."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from src.config import get_settings

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


@click.group()
def main() -> None:
    """LLM Wiki — Git-backed knowledge base maintained by LLM."""
    _setup_logging()


@main.command()
@click.argument("source")
@click.option("--url", is_flag=True, help="Treat SOURCE as a URL instead of file path.")
def ingest(source: str, url: bool) -> None:
    """Ingest a source file or URL into the wiki."""
    from src.ingest import run_ingest

    result = asyncio.run(run_ingest(source))
    errors = result.get("errors", [])
    written = result.get("written_paths", [])

    if errors:
        click.secho(f"Errors: {len(errors)}", fg="red")
        for err in errors:
            click.echo(f"  - {err}")
        raise SystemExit(1)

    click.secho(f"Created {len(written)} pages:", fg="green")
    for p in written:
        click.echo(f"  wiki/{p}")


@main.command(name="ingest-all")
@click.option("--glob", "pattern", default="*", help="Glob pattern to match source files.")
def ingest_all(pattern: str) -> None:
    """Ingest all matching files from the sources directory."""

    from src.config import get_settings
    from src.ingest import run_ingest

    settings = get_settings()
    sources = sorted(settings.sources_dir.glob(pattern))
    sources = [s for s in sources if s.is_file()]

    if not sources:
        click.secho(f"No files matching '{pattern}' in {settings.sources_dir}/", fg="yellow")
        return

    click.echo(f"Found {len(sources)} source(s) to ingest.")

    total_pages = 0
    total_errors = 0

    for src_path in sources:
        click.echo(f"\nIngesting: {src_path.name}")
        result = asyncio.run(run_ingest(str(src_path)))
        written = result.get("written_paths", [])
        errors = result.get("errors", [])

        if errors:
            click.secho(f"  Errors: {len(errors)}", fg="red")
            for err in errors:
                click.echo(f"    - {err}")
            total_errors += len(errors)
        else:
            click.secho(f"  Created {len(written)} pages", fg="green")
            total_pages += len(written)

    click.echo()
    click.secho(
        f"Done: {total_pages} pages created, {total_errors} errors",
        fg="green" if not total_errors else "yellow",
    )


@main.command()
@click.argument("question")
def query(question: str) -> None:
    """Query the wiki and get a synthesized answer."""
    from src.query import run_query

    result = asyncio.run(run_query(question))
    errors = result.get("errors", [])

    if errors:
        click.secho(f"Errors: {len(errors)}", fg="red")
        for err in errors:
            click.echo(f"  - {err}")
        raise SystemExit(1)

    click.secho("Answer:", fg="green", bold=True)
    click.echo(result.get("answer", "No answer generated."))

    citations = result.get("citations", [])
    if citations:
        click.echo()
        click.secho("Citations:", fg="cyan")
        for c in citations:
            click.echo(f"  - {c}")

    follow_ups = result.get("follow_up_queries", [])
    if follow_ups:
        click.echo()
        click.secho("Follow-up questions:", fg="yellow")
        for q in follow_ups:
            click.echo(f"  - {q}")

    if result.get("persisted_page"):
        click.echo()
        click.secho(f"Persisted as: wiki/{result['persisted_page']}", fg="green")


@main.command()
def lint() -> None:
    """Check wiki health: orphans, broken links, missing fields."""
    from src.lint import lint_wiki

    issues = lint_wiki()

    if not issues:
        click.secho("No issues found.", fg="green")
        return

    severity_colors = {"error": "red", "warning": "yellow", "info": "cyan"}

    for issue in issues:
        color = severity_colors.get(issue.severity, "white")
        click.secho(f"[{issue.severity.upper()}] {issue.page}: {issue.message}", fg=color)
        if issue.suggestion:
            click.echo(f"  → {issue.suggestion}")

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    click.echo()
    click.echo(f"Total: {len(issues)} issues ({errors} errors, {warnings} warnings)")

    if errors:
        raise SystemExit(1)


@main.command()
def stats() -> None:
    """Show wiki statistics."""
    from src.wiki import extract_wikilinks, list_pages, read_all_pages

    settings = get_settings()
    paths = list_pages(settings.wiki_dir)
    pages = read_all_pages(settings.wiki_dir)

    if not pages:
        click.echo("Wiki is empty. Run 'wiki ingest' to add content.")
        return

    # Count types
    type_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    total_links = 0

    for page in pages:
        ptype = page.frontmatter.page_type.value
        type_counts[ptype] = type_counts.get(ptype, 0) + 1

        for tag in page.frontmatter.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        total_links += len(extract_wikilinks(page.body))

    click.secho("Wiki Statistics", fg="green", bold=True)
    click.echo(f"  Pages: {len(paths)}")
    click.echo(f"  Links: {total_links}")
    click.echo(f"  Link density: {total_links / len(pages):.1f} per page")

    click.echo()
    click.secho("Page types:", fg="cyan")
    for ptype, count in sorted(type_counts.items()):
        click.echo(f"  {ptype}: {count}")

    click.echo()
    click.secho("Top tags:", fg="cyan")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        click.echo(f"  {tag}: {count}")
