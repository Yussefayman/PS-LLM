import json
import re

CHUNK_SIZE = 3
OVERLAP    = 1
INPUT      = "raw_doc.txt"
OUTPUT     = "corpus.json"


def split_sentences(text: str) -> list[dict]:
    lines = text.splitlines()
    sentences = []
    for line_no, line in enumerate(lines, start=1):
        raw = re.split(r'(?<=[.!?])\s+', line.strip())
        for s in raw:
            s = s.strip()
            if len(s) > 20:
                sentences.append({"text": s, "line": line_no})
    return sentences


def chunk_sentences(sentences: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    step = size - overlap
    for i in range(0, len(sentences), step):
        group = sentences[i:i + size]
        if not group:
            continue
        chunks.append({
            "text":     " ".join(s["text"] for s in group),
            "metadata": {
                "source":     INPUT,
                "start_line": group[0]["line"],
                "end_line":   group[-1]["line"],
                "sentences":  len(group),
            },
        })
    return chunks


def main():
    with open(INPUT, encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)
    chunks    = chunk_sentences(sentences, CHUNK_SIZE, OVERLAP)
    corpus    = [{"id": i + 1, **chunk} for i, chunk in enumerate(chunks)]

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"Chunked {len(sentences)} sentences into {len(corpus)} chunks → {OUTPUT}")


if __name__ == "__main__":
    main()