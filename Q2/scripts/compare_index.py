"""
compare_configs.py
Compares cosine vs dot product similarity metrics across test queries.
Run once to justify metric choice in the Google Doc.

Usage:
    python scripts/compare_configs.py
"""

import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT)

CORPUS_PATH = os.path.join(ROOT, "data", "corpus.json")

from rich.console import Console
from rich.table import Table
from rag_pipeline import build_index, retrieve

QUERIES = [
    "What is the minimum deposit to open a savings account?",
    "Do I need an Absher account to open an account?",
    "Can expatriates open an Al Rajhi account online?",
    "What happens if I don't make any transactions for a year?",
]

K = 3

console = Console()


def compare(query: str, index):
    cosine_results = retrieve(query, index, k=K, metric="cosine")
    dot_results    = retrieve(query, index, k=K, metric="dot")

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    table = Table(show_lines=True)
    table.add_column("Rank",       width=6,  style="dim")
    table.add_column("Cosine ID",  width=10)
    table.add_column("Score",      width=8,  style="green")
    table.add_column("Dot ID",     width=10)
    table.add_column("Score",      width=8,  style="yellow")

    for i in range(K):
        c = cosine_results[i]
        d = dot_results[i]
        same = "✅" if c["id"] == d["id"] else "⚠️"
        table.add_row(
            f"{i+1}",
            f"{c['id']} {same if i == 0 else ''}",
            str(c["score"]),
            f"{d['id']} {same if i == 0 else ''}",
            str(d["score"]),
        )

    console.print(table)

    # agreement check
    cosine_top = cosine_results[0]["id"]
    dot_top    = dot_results[0]["id"]
    if cosine_top == dot_top:
        console.print("[green]✅ Top snippet agrees across both metrics[/green]")
    else:
        console.print("[red]⚠️  Top snippet differs between metrics[/red]")

    # score range
    console.print(f"[dim]Cosine score range: {cosine_results[-1]['score']} – {cosine_results[0]['score']} (bounded 0–1)[/dim]")
    console.print(f"[dim]Dot score range:    {dot_results[-1]['score']} – {dot_results[0]['score']} (unbounded)[/dim]")


def main():
    console.print("\n[bold]Index Config Comparison — Cosine vs Dot Product[/bold]")
    console.print(f"[dim]k={K} | corpus: data/corpus.json[/dim]\n")

    index = build_index(CORPUS_PATH)

    for query in QUERIES:
        compare(query, index)



if __name__ == "__main__":
    main()