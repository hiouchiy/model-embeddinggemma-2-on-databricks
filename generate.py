#!/usr/bin/env python
"""Embed a JSONL corpus with EmbeddingGemma 2 (text, images, video, audio) to a Parquet file, and optionally rank it.

The entry point of the AI Runtime batch job (air_job/generate.yaml). Any machine with the runtime from
scripts/install.sh can run it too (a GPU is used when there is one). The request handling is
embeddinggemma2_core.Embedder, shared with the notebook and the serving model, so each line's `params` takes any
`params_json` key of the endpoint (docs/SERVING-API.md); media come in as `image_path` / `video_path` / `audio_path`
(local, /Volumes or http(s)) or `*_b64`.

    python generate.py --input examples/corpus.jsonl --out /tmp/embeddinggemma2_out/corpus.parquet \\
        --query "How do I keep vectors small?" --query-prompt SearchQuery

One JSONL line: {"id": "...", "text": "...", "params": {"prompt_name": "Document", "image_path": "..."}}.
"""
import argparse
import json
import time
from pathlib import Path

import embeddinggemma2_core as core


def main() -> None:
    ap = argparse.ArgumentParser(description="EmbeddingGemma 2 batch embedding (JSONL -> Parquet)")
    ap.add_argument("--input", required=True, help="JSONL: one {id, text, params} object per line")
    ap.add_argument("--out", required=True, help="Output .parquet path (local disk; copy to a Volume afterwards)")
    ap.add_argument("--modalities", default=None, choices=sorted(core.MODALITIES),
                    help="encoders to load (default: $EMBEDDINGGEMMA_MODALITIES, else all); README 'Choosing the encoders'")
    ap.add_argument("--params", default="{}", help="JSON object merged under every line's params (e.g. truncate_dim)")
    ap.add_argument("--batch", type=int, default=64, help="rows per handle_rows call")
    ap.add_argument("--query", default=None, help="Optional query text: print the corpus ranked by cosine similarity")
    ap.add_argument("--query-prompt", default="SearchQuery", help="prompt_name for --query")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    import pandas as pd
    base = json.loads(args.params)
    lines = [json.loads(s) for s in Path(args.input).read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = [(r.get("text", ""), {**base, **(r.get("params") or {})}) for r in lines]

    emb = core.Embedder(args.modalities)
    core.log(f"loaded {emb.modalities} on {emb.device}/{emb.precision}; embedding {len(rows)} rows")
    res, t0 = [], time.perf_counter()
    for i in range(0, len(rows), args.batch):
        res += emb.handle_rows(rows[i:i + args.batch])
    secs = time.perf_counter() - t0
    df = pd.DataFrame([{"id": str(r.get("id", n)), "text": r.get("text", ""), **o} for n, (r, o) in enumerate(zip(lines, res))])
    bad = df[df["error"] != ""]
    for _, b in bad.iterrows():
        core.log(f"row {b['id']}: {b['error']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    import torch
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[embeddinggemma2] {len(df) - len(bad)}/{len(df)} rows embedded ({df['modality'].value_counts().to_dict()}) "
          f"in {secs:.2f} s (decode + encode), peak GPU {peak:.2f} GiB -> {out}")

    if args.query:
        dim = int(base.get("truncate_dim") or core.DIMS[0])
        q = emb.handle(args.query, {"prompt_name": args.query_prompt, "truncate_dim": dim})
        if q["error"]:
            raise SystemExit(f"[embeddinggemma2] query failed: {q['error']}")
        import numpy as np
        ok = df[df["error"] == ""].copy()
        ok["score"] = np.stack(ok["embedding"].to_numpy()) @ np.array(q["embedding"])
        print(f"[embeddinggemma2] top {args.top} for {args.query!r}:")
        for _, r in ok.sort_values("score", ascending=False).head(args.top).iterrows():
            print(f"  {r['score']:.4f}  {r['modality']:<11} {r['id']}: {' '.join(str(r['text']).split())[:70]}")
    if len(bad):
        raise SystemExit(f"[embeddinggemma2] {len(bad)} row(s) failed")


if __name__ == "__main__":
    main()
