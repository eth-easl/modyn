from collections.abc import Callable
from functools import partial

from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.models.pytorch import TransformerEmbedding
from pydantic import Field
from transformers import AutoTokenizer

from modyn.config.schema.base_model import ModynBaseModel


class AlibiDetectNLPreprocessor(ModynBaseModel):
    tokenizer_model: str = Field(description="AutoTokenizer pretrained model name. E.g. bert-base-cased")
    n_layers: int = Field(8)
    max_len: int = Field(..., description="Maximum length of input token sequences.")
    batch_size: int = Field(32, description="Batch size for tokenization.")

    def gen_preprocess_fn(self, device: str | None) -> Callable:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        emb_type = "hidden_state"
        layers = [-_ for _ in range(1, self.n_layers + 1)]

        embedding = TransformerEmbedding(self.tokenizer_model, emb_type, layers)
        if device:
            embedding = embedding.to(device)
        embedding = embedding.eval()

        return partial(
            preprocess_drift, model=embedding, tokenizer=tokenizer, max_len=self.max_len, batch_size=self.batch_size
        )
