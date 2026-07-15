from rag_pipeline import RAGPipeline

TOP_K = 5

# (query, expected_class) - extend this list; 3-5 queries per class is a good start
EVAL_QUERIES = [
    ("experience with account reconciliation and financial reporting", "ACCOUNTANT"),
    ("candidate who managed accounts payable teams", "ACCOUNTANT"),
    ("knowledge of general ledger and fiscal year-end close", "ACCOUNTANT"),
]


def evaluate(rag, queries, top_k=TOP_K):
    rows = []
    for query, expected in queries:
        matches = rag.retrieve(query, top_k=top_k)
        classes = [m["metadata"]["class"] for m in matches]

        hit = expected in classes
        precision = classes.count(expected) / len(classes) if classes else 0.0
        rr = 0.0
        for rank, c in enumerate(classes, start=1):
            if c == expected:
                rr = 1.0 / rank
                break

        rows.append({
            "query": query,
            "expected": expected,
            "hit": hit,
            "precision": precision,
            "rr": rr,
            "retrieved": classes,
        })
    return rows


def report(rows, top_k=TOP_K):
    n = len(rows)
    print(f"\n=== Retrieval evaluation (top_k={top_k}, {n} queries) ===\n")
    for r in rows:
        status = "PASS" if r["hit"] else "FAIL"
        print(f"[{status}] {r['query'][:60]}")
        print(f"       expected={r['expected']}  "
              f"precision@{top_k}={r['precision']:.2f}  rr={r['rr']:.2f}")
        if not r["hit"]:
            print(f"       got instead: {r['retrieved']}")
        print()

    hit_rate = sum(r["hit"] for r in rows) / n
    avg_precision = sum(r["precision"] for r in rows) / n
    mrr = sum(r["rr"] for r in rows) / n
    print(f"hit@{top_k}:       {hit_rate:.2%}")
    print(f"precision@{top_k}: {avg_precision:.2%}")
    print(f"MRR:          {mrr:.3f}")


if __name__ == "__main__":
    rag = RAGPipeline(index_name="resumes")
    rows = evaluate(rag, EVAL_QUERIES)
    report(rows)
