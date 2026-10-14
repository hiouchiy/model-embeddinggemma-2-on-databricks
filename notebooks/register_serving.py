# Databricks notebook source
# MAGIC %md
# MAGIC # EmbeddingGemma 2 — register for Model Serving (GPU Express Deployment, env_pack)
# MAGIC
# MAGIC Run on **serverless GPU, `GPU_1xA10`, environment version 4** (a serverless client image is required: on classic
# MAGIC compute env_pack fails with `is_client_image=False`). Open it from a Git folder (or a synced copy) of this repo.
# MAGIC
# MAGIC The model needs transformers 5.19 / sentence-transformers 6.1 on torch 2.10 / cu128. Standard GPU Model Serving
# MAGIC builds its own container and replaces torch with its own build. **`env_pack="databricks_model_serving"`** instead
# MAGIC packs THIS notebook's environment and restores it at serving verbatim, so there is no container rebuild.
# MAGIC
# MAGIC 1. Install the pinned runtime (`scripts/install.sh`; the pins and why they are what they are: `embeddinggemma2_core.py`).
# MAGIC 2. **Smoke-test on the A10** with the exact serving PyFunc.
# MAGIC 3. Log the PyFunc (`serving/embeddinggemma2_pyfunc.py`, code_paths = `embeddinggemma2_core.py`), then register it
# MAGIC    to Unity Catalog with env_pack and `install_dependencies=False` (pack and restore without re-resolving).
# MAGIC
# MAGIC The HF checkpoint is not bundled; the endpoint downloads it (≈ 1.5 GB) at load, so the serving environment needs
# MAGIC Hub egress.
# MAGIC
# MAGIC `modalities` picks the encoders the endpoint loads (`embeddinggemma2_core.MODALITIES`): `all` (740M), `text`
# MAGIC (270M), `text+vision` or `text+audio`. Register each set as its own UC model (README "Choosing the encoders").

# COMMAND ----------

import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil
import sys
import time

REPO = os.path.abspath("..")
sys.path[:0] = [REPO, os.path.join(REPO, "serving")]
import embeddinggemma2_core as core  # noqa: E402

dbutils.widgets.dropdown("modalities", core.DEFAULT_MODALITIES, sorted(core.MODALITIES), "modalities (encoders)")
dbutils.widgets.text("uc_model", "<CATALOG>.<SCHEMA>.embeddinggemma2", "UC model (catalog.schema.name)")
MODALITIES = core.modalities_name(dbutils.widgets.get("modalities"))
UC_MODEL = dbutils.widgets.get("uc_model")
assert "<" not in UC_MODEL, "set uc_model to <catalog>.<schema>.<model>"
os.environ["EMBEDDINGGEMMA_MODALITIES"] = MODALITIES   # the smoke test's load_context(None) picks it up

WORK = "/tmp/embeddinggemma2"
import torch  # noqa: E402
core.log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} {core.HF_MODEL} @ {core.HF_REVISION[:7]} modalities {MODALITIES}")

# COMMAND ----------

# MAGIC %md ## Smoke test with the exact serving PyFunc on this A10

# COMMAND ----------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from embeddinggemma2_pyfunc import EmbeddingGemma2Model  # noqa: E402

m = EmbeddingGemma2Model()
m.load_context(None)
q, d = m.predict(None, pd.DataFrame([
    {"text": "What causes the northern lights?", "params_json": json.dumps({"prompt_name": "SearchQuery"})},
    {"text": "The northern lights are caused by charged particles from the sun.", "params_json": json.dumps({"prompt_name": "Document"})},
])).to_dict(orient="records")
assert not q["error"] and not d["error"], (q["error"], d["error"])
SMOKE = {"modalities": MODALITIES, "dim": q["dim"], "query~doc": round(float(np.dot(q["embedding"], d["embedding"])), 4),
         "embed_seconds": q["embed_seconds"], "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
core.log(f"smoke ok: {SMOKE}")
del m; torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## Log, then register with env_pack (install_dependencies=False)

# COMMAND ----------

import mlflow  # noqa: E402
from mlflow.utils.env_pack import EnvPackConfig  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
shutil.copy(os.path.join(REPO, "embeddinggemma2_core.py"), WORK)
pyfunc = f"{WORK}/embeddinggemma2_pyfunc.py"
shutil.copy(os.path.join(REPO, "serving", "embeddinggemma2_pyfunc.py"), pyfunc)

# Two stable string columns: `text` + `params_json` (opaque JSON: prompt, truncate_dim, media).
input_example = pd.DataFrame([{"text": "What causes the northern lights?",
                               "params_json": json.dumps({"prompt_name": "SearchQuery", "truncate_dim": 768})}])
t0 = time.time()
info = mlflow.pyfunc.log_model(
    name="embeddinggemma2",
    python_model=pyfunc,
    code_paths=[f"{WORK}/embeddinggemma2_core.py"],
    model_config={"modalities": MODALITIES, "hf_model": core.HF_MODEL, "revision": core.HF_REVISION, "precision": "bf16"},
    input_example=input_example,
    extra_pip_requirements=core.SERVING_PIP_REQUIREMENTS,
)
core.log(f"logged {info.model_uri}")
mv = mlflow.register_model(model_uri=info.model_uri, name=UC_MODEL,
                           env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))
core.log(f"registered {UC_MODEL} version {mv.version} in {time.time() - t0:.0f}s")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "modalities": MODALITIES, "version": mv.version, "smoke": SMOKE}))
