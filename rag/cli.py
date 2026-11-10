"""Command-line interface: `python -m rag.cli ingest|ask|chat|search|list|delete|clear|serve`."""

from __future__ import annotations

import argparse
import logging
import sys


def _pipeline():
    from rag.pipeline import RAGPipeline

    return RAGPipeline()


def _print_answer_stream(rag, question: str, history: list[dict], show_sources: bool) -> str:
    answer = []
    for event in rag.ask_stream(question, history):
        if event["type"] == "sources" and show_sources:
            if event["sources"]:
                print("Sources:")
                for s in event["sources"]:
                    print(f"  [{s['n']}] {s['source']} (chunk {s['chunk']})")
                print()
            else:
                print("(no matching documents)\n")
        elif event["type"] == "text":
            answer.append(event["text"])
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print()
            if event.get("notice"):
                print(f"\n[{event['notice']}]")
        elif event["type"] == "error":
            print(f"\nError: {event['message']}", file=sys.stderr)
    return "".join(answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Retrieval-augmented Q&A over your documents with Claude.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="Index files or directories")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask", help="Ask a single question")
    p.add_argument("question")
    p.add_argument("--no-sources", action="store_true")
    sub.add_parser("chat", help="Interactive multi-turn chat")
    p = sub.add_parser("search", help="Show retrieved chunks without calling Claude")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    sub.add_parser("list", help="List indexed documents")
    p = sub.add_parser("delete", help="Remove a document by id")
    p.add_argument("doc_id")
    sub.add_parser("clear", help="Remove all documents from the index")
    sub.add_parser("set-key", help="Enter, validate and save your Anthropic API key to .env")
    p = sub.add_parser("serve", help="Run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    rag = _pipeline()
    if args.cmd in ("ask", "chat") and not rag.has_api_key:
        if sys.stdin.isatty():
            print("No Anthropic API key is configured.")
            if main(["set-key"]) != 0:
                return 1
            rag = _pipeline()
        else:
            from rag.pipeline import MISSING_KEY_MESSAGE

            print(f"Error: {MISSING_KEY_MESSAGE}", file=sys.stderr)
            return 1
    if args.cmd == "ingest":
        for r in rag.ingest_paths(args.paths):
            if "error" in r:
                print(f"  ✗ {r['source']}: {r['error']}")
            elif r["skipped"]:
                print(f"  = {r['source']} (unchanged)")
            else:
                print(f"  ✓ {r['source']}: {r['chunks']} chunks")
        print(f"Index: {len(rag.store.documents())} documents, {len(rag.store.chunks)} chunks ({rag.store.mode}).")
    elif args.cmd == "ask":
        _print_answer_stream(rag, args.question, [], not args.no_sources)
    elif args.cmd == "chat":
        history: list[dict] = []
        print(f"Chatting over {len(rag.store.documents())} documents. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            print("claude> ", end="")
            answer = _print_answer_stream(rag, q, history, show_sources=False)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
            history = history[-20:]
            print()
    elif args.cmd == "search":
        for n, r in enumerate(rag.store.search(args.query, top_k=args.k), start=1):
            preview = r.chunk.text[:200].replace("\n", " ")
            print(f"[{n}] {r.chunk.source} #{r.chunk.index + 1}  score={r.score:.4f}\n    {preview}…\n")
    elif args.cmd == "list":
        docs = rag.store.documents()
        for d in docs:
            print(f"{d['doc_id']}  {d['chunks']:>4} chunks  {d['source']}")
        print(f"{len(docs)} documents ({rag.store.mode})")
    elif args.cmd == "delete":
        removed = rag.store.delete_document(args.doc_id)
        print(f"Removed {removed} chunks." if removed else "No such document.")
    elif args.cmd == "set-key":
        import getpass

        import anthropic

        key = getpass.getpass("Anthropic API key (input hidden): ")
        try:
            rag.set_api_key(key)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except anthropic.AuthenticationError:
            print("Error: Anthropic rejected this key.", file=sys.stderr)
            return 1
        except anthropic.APIError as exc:
            print(f"Error: could not verify the key ({exc}).", file=sys.stderr)
            return 1
        print("API key verified and saved to .env.")
    elif args.cmd == "clear":
        rag.store.clear()
        print("Index cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
