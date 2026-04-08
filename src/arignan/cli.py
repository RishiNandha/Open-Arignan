from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence, TextIO

from arignan.application import ArignanApp, AskResult, LoadResult, format_citation
from arignan.config import load_config


class ArignanHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30, width=100)


class LineProgressReporter:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, message: str) -> None:
        print(f"[arignan] {message}", file=self.stream, flush=True)

    def finish(self) -> None:
        return None


class AskStatusReporter:
    _spinner_frames = "|/-\\"

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self._spinner_index = 0
        self._last_length = 0
        self._active = False

    def emit(self, message: str) -> None:
        if message.startswith("Local LLM unavailable"):
            self._emit_line(message)
            return
        status = self._map_status(message)
        if status is None:
            return
        frame = self._spinner_frames[self._spinner_index % len(self._spinner_frames)]
        self._spinner_index += 1
        payload = f"[arignan] {frame} {status}"
        padding = " " * max(self._last_length - len(payload), 0)
        print(f"\r{payload}{padding}", end="", file=self.stream, flush=True)
        self._last_length = len(payload)
        self._active = True

    def finish(self) -> None:
        if not self._active:
            return
        clear = " " * self._last_length
        print(f"\r{clear}\r", end="", file=self.stream, flush=True)
        self._last_length = 0
        self._active = False

    def _emit_line(self, message: str) -> None:
        self.finish()
        print(f"[arignan] {message}", file=self.stream, flush=True)

    @staticmethod
    def _map_status(message: str) -> str | None:
        if message.startswith("Hat chosen:"):
            return message
        if any(
            token in message
            for token in (
                "Running retrieval pipeline",
                "Expanding query",
                "Selecting hat",
                "Searching dense index",
                "Searching lexical index",
                "Searching map context",
                "Fusing retrieval candidates",
            )
        ):
            return "Retrieval in progress"
        if "Reranking" in message or "rerank" in message.lower():
            return "Reranking"
        if "Hitting local LLM" in message:
            return "Hitting LLM"
        if "Composing raw retrieval output" in message or "Composing retrieval synthesis answer" in message:
            return "Composing answer"
        return None


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
            "  arignan delete <load_id>\n"
            "  arignan delete --hat psychology"
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
    ask_parser.add_argument(
        "--answer-mode",
        choices=["default", "light", "none", "raw"],
        default="default",
        help="Choose how the final answer is produced: default LLM, light LLM, deterministic summary, or raw reranked context.",
    )
    ask_parser.add_argument("--debug", action="store_true", help="Print retrieval and reranking context details.")

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete one or more past ingestions by load_id.",
        description="Delete one or more past ingestions by load_id.",
        formatter_class=ArignanHelpFormatter,
    )
    delete_parser.add_argument("--hat", help="Delete an entire hat after confirmation.")
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
    resolved_pid = args.pid or os.getppid() or os.getpid()
    config = load_config(settings_path=args.settings, app_home=args.app_home)
    reporter = _build_progress_reporter(command=args.command, debug=getattr(args, "debug", False))
    app = ArignanApp(config, progress_sink=reporter.emit, terminal_pid=resolved_pid)

    try:
        if args.command == "load":
            result = app.load(args.input_ref, hat=args.hat)
            reporter.finish()
            _print_output_block(_format_load_summary(result))
            if args.debug:
                _print_output_block(_format_load_debug(result))
            return 0

        if args.command == "ask":
            result = app.ask(args.question, hat=args.hat, terminal_pid=resolved_pid, answer_mode=args.answer_mode)
            reporter.finish()
            ask_lines = [result.answer]
            if result.citations:
                ask_lines.append("")
                ask_lines.append("Citations:")
                for citation in result.citations:
                    ask_lines.append(f"- {citation}")
            _print_output_block("\n".join(ask_lines))
            if args.debug:
                _print_output_block(_format_ask_debug(result))
            return 0

        if args.command == "delete":
            if args.hat and args.load_ids:
                parser.error("delete accepts either one or more load_ids or --hat, not both")
            if args.hat:
                confirmation = input(
                    f"Delete entire hat '{args.hat}'? This will remove all indexes, summaries, and source copies for that hat. [y/N]: "
                ).strip()
                if confirmation.lower() not in {"y", "yes"}:
                    _print_output_block("Cancelled hat deletion.")
                    return 0
                result = app.delete_hat(args.hat)
                if not result.existed:
                    reporter.finish()
                    _print_output_block(f"Hat '{args.hat}' was not found.")
                    return 0
                reporter.finish()
                _print_output_block(
                    f"Deleted hat '{result.hat}'. "
                    f"Loads removed: {len(result.deleted_load_ids)}. "
                    f"Topics removed: {len(result.deleted_topics)}."
                )
                return 0
            if not args.load_ids:
                events = app.list_ingestions()
                reporter.finish()
                if not events:
                    _print_output_block("No ingestions found.")
                    return 0
                _print_output_block(
                    "\n".join(f"{event.load_id}\t{event.hat}\t{', '.join(event.source_items)}" for event in events)
                )
                return 0
            result = app.delete(args.load_ids)
            reporter.finish()
            _print_output_block(
                f"Deleted loads: {', '.join(result.deleted_load_ids) or 'none'}. "
                f"Missing: {', '.join(result.missing_load_ids) or 'none'}."
            )
            return 0

        if args.command == "save-session":
            destination = Path(args.destination) if args.destination else None
            path = app.save_session(terminal_pid=resolved_pid, destination=destination)
            reporter.finish()
            _print_output_block(str(path))
            return 0

        if args.command == "load-session":
            session = app.load_session(Path(args.source), terminal_pid=resolved_pid)
            reporter.finish()
            _print_output_block(f"Loaded session {session.session_id} into pid {session.terminal_pid}.")
            return 0

        if args.command == "reset-session":
            session = app.reset_session(terminal_pid=resolved_pid)
            reporter.finish()
            _print_output_block(f"Reset session. New session_id: {session.session_id}")
            return 0

        if args.command == "list-loads":
            events = app.list_events()
            reporter.finish()
            if not events:
                _print_output_block("No log events found.")
                return 0
            _print_output_block(
                "\n".join(
                    f"{event.created_at}\t{event.operation.value}\t{event.load_id}\t{event.hat}\t{', '.join(event.source_items)}"
                    for event in events
                )
            )
            return 0

        parser.error(f"unsupported command: {args.command}")
        return 2
    except Exception as exc:
        reporter.finish()
        log_path = app.log_exception(
            component="cli",
            task=f"{args.command} command",
            exc=exc,
            context={"argv": list(argv) if argv is not None else None},
        )
        print(f"[arignan] Full traceback logged to {log_path}", file=sys.stderr)
        raise


def _build_progress_reporter(*, command: str, debug: bool):
    if command == "ask" and not debug:
        return AskStatusReporter()
    return LineProgressReporter()


def _print_output_block(text: str) -> None:
    print()
    if text:
        print(text)
    print()


def _format_load_summary(result: LoadResult) -> str:
    lines = [
        f"Loaded {result.document_count} document(s) into hat '{result.hat}' "
        f"with load_id {result.load_id}. Topics: {', '.join(result.topic_folders) or 'none'}. "
        f"Chunks: {result.total_chunks}. Markdown segments: {result.total_markdown_segments}. "
        f"Failed files: {len(result.failures)}."
    ]
    if result.failures:
        lines.append("")
        lines.append("Failed files:")
        for failure in result.failures:
            lines.append(f"- {failure.source_uri}")
    return "\n".join(lines)


def _format_load_debug(result: LoadResult) -> str:
    lines = ["Debug: load details"]
    lines.extend(_format_model_calls(result.model_calls))
    if result.failures:
        lines.append(f"Failures ({len(result.failures)}):")
        for index, failure in enumerate(result.failures, start=1):
            lines.append(f"  {index}. {failure.source_uri}")
            lines.append(f"     {failure.message}")
        lines.append("")
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
        f"Answer mode: {debug.answer_mode}",
        f"Selected hat: {debug.selected_hat}",
        f"Expanded query: {debug.expanded_query}",
        (
            "Hit counts: "
            f"dense={len(debug.dense_hits)}, lexical={len(debug.lexical_hits)}, "
            f"map={len(debug.map_hits)}, fused={len(debug.fused_hits)}, reranked={len(debug.reranked_hits)}"
        ),
        "",
    ]
    lines.extend(_format_model_calls(debug.model_calls))
    lines.extend(_format_hit_group("Dense hits", debug.dense_hits))
    lines.extend(_format_hit_group("Lexical hits", debug.lexical_hits))
    lines.extend(_format_hit_group("Map hits", debug.map_hits))
    lines.extend(_format_hit_group("Fused hits", debug.fused_hits))
    lines.extend(_format_hit_group("Reranked hits", debug.reranked_hits))
    return "\n".join(lines).rstrip()


def _format_model_calls(calls) -> list[str]:
    lines = [f"Model calls ({len(calls)}):"]
    if not calls:
        lines.append("  none")
        lines.append("")
        return lines
    for index, call in enumerate(calls, start=1):
        detail_parts = [call.task]
        if call.item_count is not None:
            detail_parts.append(f"items={call.item_count}")
        if call.detail:
            detail_parts.append(call.detail)
        details = " | ".join(detail_parts)
        lines.append(
            f"  {index}. [{call.status}] {call.component} | {call.backend} | {call.model_name}"
        )
        lines.append(f"     {details}")
    lines.append("")
    return lines


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
