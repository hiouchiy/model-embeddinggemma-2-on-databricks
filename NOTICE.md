# NOTICE

This repo is a **thin wrapper, procedures and tools** to run and serve **EmbeddingGemma 2** (by Google DeepMind) on
Databricks. It does not include the model: the notebooks, the AIR job and the endpoint download the weights from the
Hugging Face Hub at a pinned revision, and the model code comes from the `transformers` and `sentence-transformers`
releases pinned in `embeddinggemma2_core.RUNTIME_REQUIREMENTS`.

- Model: https://huggingface.co/google/embeddinggemma-2 @ `914f7f89142e33e77833254d9c9b90c3cef7303b`
  (`embeddinggemma2_core.HF_REVISION`); not gated.
- Licence: **Apache License 2.0** (the model card links it as the Gemma 4 licence:
  https://ai.google.dev/gemma/docs/gemma_4_license).
- The model card adds: "Deployments must adhere to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."

Check the upstream terms yourself before use. Code written in this repo is provided under the Apache License 2.0, the
same as the model.
