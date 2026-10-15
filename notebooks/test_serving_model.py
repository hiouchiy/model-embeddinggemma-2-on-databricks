# Databricks notebook source
# MAGIC %md
# MAGIC # EmbeddingGemma 2 — test the serving model before registering it
# MAGIC
# MAGIC Runs `serving/embeddinggemma2_pyfunc.py` (the MLflow PyFunc the endpoint serves) in-process on **serverless GPU,
# MAGIC A10**: `load_context` + one `predict` with a row per input kind (text query, document, image, text + image,
# MAGIC audio, video, a 256-d Matryoshka row), `__list_params__` and two bad rows, and checks the response shape and
# MAGIC that the vectors are unit length. Use it after changing `serving/embeddinggemma2_pyfunc.py` or
# MAGIC `embeddinggemma2_core.py`, before `notebooks/register_serving.py`. It does not touch any endpoint.

# COMMAND ----------

import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import base64
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath("..")
sys.path[:0] = [REPO, os.path.join(REPO, "serving"), os.path.join(REPO, "scripts")]
import embeddinggemma2_core as core  # noqa: E402
import sample_media as sm  # noqa: E402
dbutils.widgets.dropdown("modalities", core.DEFAULT_MODALITIES, sorted(core.MODALITIES), "modalities (encoders)")
os.environ["EMBEDDINGGEMMA_MODALITIES"] = core.modalities_name(dbutils.widgets.get("modalities"))
from embeddinggemma2_pyfunc import EmbeddingGemma2Model  # noqa: E402

m = EmbeddingGemma2Model()
m.load_context(None)   # the encoders: $EMBEDDINGGEMMA_MODALITIES, set from the widget above
inputs = core.MODALITIES[os.environ["EMBEDDINGGEMMA_MODALITIES"]]["inputs"]
P = json.dumps
cases = [("query", "What causes the northern lights?", P({"prompt_name": "SearchQuery"})),
         ("doc", "The northern lights are caused by charged particles from the sun.", P({"prompt_name": "Document"})),
         ("other", "今日の昼ごはんはカレーでした。", P({"prompt_name": "Document"})),
         ("dim256", "short query", P({"prompt_name": "SearchQuery", "truncate_dim": 256}))]
if "image" in inputs:
    cases += [("image", "", P({"image_b64": sm.image_b64("red")})),
              ("text+image", "A red circle <|image|>", P({"image_b64": sm.image_b64("red")}))]
if "audio" in inputs:
    cases += [("audio", "", P({"audio_b64": sm.audio_b64(440)}))]
if "video" in inputs:
    cases += [("video", "", P({"video_b64": sm.video_b64((255, 0, 0))}))]
cases += [("list_params", "", P({"__list_params__": True})),
          ("bad_dim", "x", P({"truncate_dim": 300})),
          ("bad_image", "", P({"image_b64": base64.b64encode(b"not an image").decode()}))]
res = dict(zip([c[0] for c in cases], m.predict(None, pd.DataFrame([{"text": t, "params_json": p} for _, t, p in cases])).to_dict(orient="records")))

for name, r in res.items():
    if name.startswith(("bad", "list")):
        continue
    assert r["error"] == "", (name, r["error"])
    want = 256 if name == "dim256" else 768
    norm = float(np.linalg.norm(r["embedding"]))
    assert r["dim"] == want and abs(norm - 1) < 1e-3, (name, r["dim"], norm, r["embedding"][:4])
    assert not np.isnan(r["embedding"]).any(), name
assert res["bad_dim"]["error"] and res["bad_image"]["error"] and len(res["bad_image"]["embedding"]) == 0
assert set(json.loads(res["list_params"]["messages"])["inputs"]) == set(inputs)
cos = lambda a, b: float(np.dot(res[a]["embedding"], res[b]["embedding"]))
sims = {"query~doc": cos("query", "doc"), "query~other": cos("query", "other")}
assert sims["query~doc"] > sims["query~other"], sims
print(json.dumps({k: {f: v for f, v in r.items() if f != "embedding"} for k, r in res.items()}, ensure_ascii=False, indent=1)[:4000])
print(sims)

# COMMAND ----------

import torch  # noqa: E402
dbutils.notebook.exit(json.dumps({"modalities": os.environ["EMBEDDINGGEMMA_MODALITIES"], "rows_ok": [k for k in res if not k.startswith(("bad", "list"))],
                                  "similarity": sims, "embed_seconds": res["query"]["embed_seconds"],
                                  "bad_dim_error": res["bad_dim"]["error"][:200],
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, ensure_ascii=False))
