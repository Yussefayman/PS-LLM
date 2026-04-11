import json
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
LLM_MODEL       = os.getenv("LLM_MODEL",       "llama3-8b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
THRESHOLD       = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))

SYSTEM_PROMPT = """You are a banking assistant for Al Rajhi Bank.
Answer using ONLY the knowledge base below. If the answer is not here, say you don't have that information.

- To open a savings account you need a valid National ID or Iqama, an active Absher account, a valid National Address Registration, and a mobile number registered in your name.
- There is no minimum balance requirement. You only need to deposit SAR 1 within 90 days of opening. Failure to deposit may result in account closure.
- Account opening is fully digital via the Al Rajhi mobile app. No branch visit needed.
- An active Absher account is mandatory. It verifies your identity and national address digitally.
- Expatriates with a valid Iqama or Visitor ID can open an account online without visiting a branch.
- New customers get 1,000 Mokafaa loyalty points as a welcome bonus upon opening via the app.
- Your registered mobile number must be active and in your name for OTP and security alerts.
- Family Account service lets the main holder open sub-accounts for family members via the app.
- Accounts with no transactions for 12 consecutive months are classified as dormant and charged SAR 25 per quarter.
- Upon opening you receive a free Mada debit card printable instantly at any branch kiosk."""

client   = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)
console  = Console()


def llm_service(prompt: str) -> str:
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
    )
    return res.choices[0].message.content.strip()


def semantic_score(a: str, b: str) -> float:
    ea, eb = embedder.encode(a), embedder.encode(b)
    return float(dot(ea, eb) / (norm(ea) * norm(eb)))


def prompt_eval():
    with open("test_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    results = []

    console.print(f"\n[bold cyan]Eval Harness — Al Rajhi Bank[/bold cyan]")
    console.print(f"[dim]LLM: {LLM_MODEL} | Embedder: {EMBEDDING_MODEL} | Threshold: {THRESHOLD}[/dim]\n")

    for case in track(cases, description="Running..."):
        response = llm_service(case["prompt"])
        score    = semantic_score(response, case["reference"])
        results.append({**case, "score": round(score, 3), "passed": score >= THRESHOLD, "response": response})

    table = Table(show_lines=True)
    table.add_column("ID",           style="dim",    width=4)
    table.add_column("Label",        width=22)
    table.add_column("Failure Mode", style="yellow", width=26)
    table.add_column("Score",        width=6)
    table.add_column("Result",       width=8)

    for r in results:
        status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"
        table.add_row(r["id"], r["label"], r["failure_mode"], str(r["score"]), status)

    console.print(table)

    total     = len(results)
    passed    = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / total
    failures  = [r for r in results if not r["passed"]]

    console.print(f"\n[bold]Pass rate:[/bold] {passed}/{total} ({passed/total*100:.0f}%)")
    console.print(f"[bold]Avg score:[/bold] {avg_score:.3f}\n")

    if failures:
        console.print("[bold red]Failed:[/bold red]")
        for f in failures:
            console.print(f"  [{f['id']}] {f['label']} — {f['score']}")
            console.print(f"  [dim]Response: {f['response'][:120]}[/dim]\n")


if __name__ == "__main__":
    prompt_eval()