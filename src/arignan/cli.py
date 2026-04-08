from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from arignan.application import ArignanApp, AskResult, LoadResult, format_citation
from arignan.config import load_config


class ArignanHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30, width=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arignan",
        usage="arignan [--app-home PATH] [--settings PATH] [--pid PID] <command> [<args>]",
        description="Local-first knowledge base CLI for loading content, retrieval, and session management.",
        formatter_class=ArignanHelpFormatter,
        epilog=(
            "Examples:\n"
            "  arignan load notes.md\n"
            "  arignan ask \"What is JEPA?\"\n"
            "  arignan delete <load_id>"
        ),
    )
    parser.add_argument("--app-home", type=Path, default=None, help="Override the Arignan app-home directory.")
    parser.add_argument("--settings", type=Path, default=None, help="Use a specific settings.json file.")
    parser.add_argument("--pid", type=int, default=None, help="Override the terminal PID used for session state.")

    subparsers = parser.add_subparsers(dest="command", required=True, title="commands", metavar="<command>")

    load_parser = subparsers.add_parser(
        "load",
        help="Ingest a PDF, markdown file, folder, or URL into the knowledge base.",
        description="Ingest a PDF, markdown file, folder, or URL into the knowledge base.",
        formatter_class=ArignanHelpFormatter,
    )
    load_parser.add_argument("input_ref", help="Path or URL to ingest.")
    load_parser.add_argument("--hat", default="auto", help="Target hat. Defaults to auto.")
    load_parser.add_argument("--debug", action="store_true", help="Print grouping and segmentation details.")

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question against the local knowledge base.",
        description="Ask a question against the local knowledge base.",
        formatter_class=ArignanHelpFormatter,
    )
    ask_parser.add_argument("question", help="Question to answer.")
    ask_parser.add_argument("--hat", default="auto", help="Restrict retrieval to a hat. Defaults to auto.")
    ask_parser.add_argument("--debug", action="store_true", help="Print retrieval and reranking context details.")

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete one or more past ingestions by load_id.",
        description="Delete one or more past ingestions by load_id.",
        formatter_class=ArignanHelpFormatter,
    )
    delete_parser.add_argument("load_ids", nargs="*", help="One or more load IDs. Omit to list ingestions.")

    save_parser = subparsers.add_parser(
        "save-session",
        help="Save the current active session to disk.",
        description="Save the current active session to disk.",
        formatter_class=ArignanHelpFormatter,
    )
    save_parser.add_argument("destination", nargs="?", help="Optional destination JSON path.")

    load_session_parser = subparsers.add_parser(
        "load-session",
        help="Load a saved session JSON into the active session slot.",
        description="Load a saved session JSON into the active session slot.",
        formatter_class=ArignanHelpFormatter,
    )
    load_session_parser.add_argument("source", help="Path to a saved session JSON file.")

    subparsers.add_parser(
        "reset-session",
        help="Reset the active session for the current terminal PID.",
        description="Reset the active session for the current terminal PID.",
        formatter_class=ArignanHelpFormatter,
    )
    subparsers.add_parser(
        "list-loads",
        help="List ingestion log events, including deletes.",
        description="List ingestion log events, including deletes.",
        formatter_class=ArignanHelpFormatter,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(settings_path=args.settings, app_home=args.app_home)
    app = ArignanApp(config)

    if args.command == "load":
        result = app.load(args.input_ref, hat=args.hat)
        print(_format_load_summary(result))
        if args.debug:
            print()
            print(_format_load_debug(result))
        return 0

    if args.command == "ask":
        result = app.ask(args.question, hat=args.hat, terminal_pid=args.pid)
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for citation in result.citations:
                print(f"- {citation}")
        if args.debug:
            print()
            print(_format_ask_debug(result))
        return 0

    if args.command == "delete":
        if not args.load_ids:
            events = app.list_ingestions()
            if not events:
                print("No ingestions found.")
                return 0
            for event in events:
                print(f"{event.load_id}\t{event.hat}\t{', '.join(event.source_items)}")
            return 0
        result = app.delete(args.load_ids)
        print(
            f"Deleted loads: {', '.join(result.deleted_load_ids) or 'none'}. "
            f"Missing: {', '.join(result.missing_load_ids) or 'none'}."
        )
        return 0

    if args.command == "save-session":
        destination = Path(args.destination) if args.destination else None
        path = app.save_session(terminal_pid=args.pid, destination=destination)
        print(path)
        return 0

    if args.command == "load-session":
        session = app.load_session(Path(args.source), terminal_pid=args.pid)
        print(f"Loaded session {session.session_id} into pid {session.terminal_pid}.")
        return 0

    if args.command == "reset-session":
        session = app.reset_session(terminal_pid=args.pid)
        print(f"Reset session. New session_id: {session.session_id}")
        return 0

    if args.command == "list-loads":
        events = app.list_events()
        if not events:
            print("No log events found.")
            return 0
        for event in events:
            print(f"{event.created_at}\t{event.operation.value}\t{event.load_id}\t{event.hat}\t{', '.join(event.source_items)}")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


def _format_load_summary(result: LoadResult) -> str:
    return (
        f"Loaded {result.document_count} document(s) into hat '{result.hat}' "
        f"with load_id {result.load_id}. Topics: {', '.join(result.topic_folders) or 'none'}. "
        f"Chunks: {result.total_chunks}. Markdown segments: {result.total_markdown_segments}."
    )


def _format_load_debug(result: LoadResult) -> str:
    lines = ["Debug: load details"]
    for index, trace in enumerate(result.traces, start=1):
        lines.extend(
            [
                f"{index}. {trace.title}",
                f"   Source: {trace.source_uri}",
                f"   Topic folder: {trace.topic_folder}",
                f"   Grouping decision: {trace.grouping_decision}",
                f"   Chunks: {trace.chunk_count}",
                f"   Markdown segments: {trace.markdown_segment_count}",
            ]
        )
        if trace.segment_titles:
            lines.append(f"   Segment titles: {', '.join(trace.segment_titles)}")
        if trace.rationale:
            lines.append("   Rationale:")
            for item in trace.rationale:
                lines.append(f"   - {item}")
    return "\n".join(lines)


def _format_ask_debug(result: AskResult) -> str:
    debug = result.debug
    lines = [
        "Debug: ask retrieval",
        f"Selected hat: {debug.selected_hat}",
        f"Expanded query: {debug.expanded_query}",
        (
            "Hit counts: "
            f"dense={len(debug.dense_hits)}, lexical={len(debug.lexical_hits)}, "
            f"map={len(debug.map_hits)}, fused={len(debug.fused_hits)}, reranked={len(debug.reranked_hits)}"
        ),
        "",
    ]
    lines.extend(_format_hit_group("Dense hits", debug.dense_hits))
    lines.extend(_format_hit_group("Lexical hits", debug.lexical_hits))
    lines.extend(_format_hit_group("Map hits", debug.map_hits))
    lines.extend(_format_hit_group("Fused hits", debug.fused_hits))
    lines.extend(_format_hit_group("Reranked hits", debug.reranked_hits))
    return "\n".join(lines).rstrip()


def _format_hit_group(title: str, hits) -> list[str]:
    lines = [f"{title} ({len(hits)}):"]
    if not hits:
        lines.append("  none")
        lines.append("")
        return lines
    for index, hit in enumerate(hits[:5], start=1):
        citation = format_citation(hit)
        snippet = " ".join(hit.text.split())
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        score = hit.extras.get("rerank_score", hit.extras.get("rrf_score", hit.score))
        lines.append(f"  {index}. [{score:.3f}] {citation}")
        lines.append(f"     {snippet}")
    lines.append("")
    return lines
