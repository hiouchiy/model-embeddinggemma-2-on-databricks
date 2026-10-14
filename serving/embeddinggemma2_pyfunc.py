"""EmbeddingGemma 2 serving PyFunc (MLflow models-from-code): the whole endpoint API, docs/SERVING-API.md.

Request (dataframe_records): each row = {"text": <str>, "params_json": <str>}. `params_json` is a JSON object with the
task prefix (`prompt_name` / `prompt` / `title`), the Matryoshka size (`truncate_dim`), and the media to embed with
or instead of the text (`image_b64` / `video_b64` / `audio_b64`, one item or a list). One opaque JSON string is used,
rather than typed columns, so MLflow schema enforcement never drops a key. `{"__list_params__": true}` returns the
supported names, the model's prompts and the loaded encoders without embedding.

One row is one embedding. Rows that share truncate_dim / normalize / processing_kwargs are encoded as one batch.

The request handling lives in embeddinggemma2_core.Embedder, shipped next to this file as a code_path by
notebooks/register_serving.py.
"""
import mlflow.pyfunc
import pandas as pd


class EmbeddingGemma2Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import embeddinggemma2_core as core
        cfg = (getattr(context, "model_config", None) or {}) if context is not None else {}
        self._embedder = core.Embedder(cfg.get("modalities"), cfg.get("precision"))

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = list(model_input)
        return pd.DataFrame(self._embedder.handle_rows([(row.get("text", ""), row.get("params_json")) for row in rows]))


mlflow.models.set_model(EmbeddingGemma2Model())
