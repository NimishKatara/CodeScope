"""
cli_demo.py
Command-line demo of the Codebase Understanding Agent.
Run: python cli_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from agent import CodebaseAgent

SAMPLE_REPO = os.path.join(os.path.dirname(__file__), "sample_repo")

DEMO_QUESTIONS = [
    "Where is authentication handled?",
    "How does data flow through the system?",
    "How are passwords hashed?",
    "What database operations exist?",
]

def separator(label=""):
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}\n")
    else:
        print("\n" + "─" * width + "\n")


def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        CodeScope — Codebase Understanding Agent      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print(f"Indexing repository: {SAMPLE_REPO}")
    agent = CodebaseAgent(SAMPLE_REPO)
    stats = agent.index()

    separator("INDEX STATS")
    print(f"  Total chunks  : {stats['total_chunks']}")
    print(f"  Python files  : {stats['total_files']}")
    print(f"  Functions     : {stats['functions']}")
    print(f"  Classes       : {stats['classes']}")
    print(f"  Files         : {', '.join(stats['files'])}")

    # Interactive or demo mode
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEMO_QUESTIONS[:2]  # Run 2 demo questions

    for question in questions:
        separator(f"QUERY")
        print(f"  Q: {question}\n")

        result = agent.query(question)
        baseline = agent.keyword_search_baseline(question)

        print("  [Agent] Retrieved chunks:")
        for c in result["retrieved_chunks"]:
            print(f"    • {c['name']} ({c['type']}) — {c['file']} {c['lines']}")

        print(f"\n  [Keyword Baseline]:")
        for r in baseline["results"][:3]:
            print(f"    • {r['name']} — {r['file']} ({r['hits']} hits)")

        separator("ANSWER")
        # Print answer with simple word-wrap
        answer = result["answer"]
        for line in answer.split("\n"):
            print(f"  {line}")

    separator()
    print("  Open frontend/dashboard.html in a browser for the full interactive UI.\n")


if __name__ == "__main__":
    main()
