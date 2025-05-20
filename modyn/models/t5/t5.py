from typing import Any
import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from modyn.models.coreset_methods_support import CoresetSupportingModule


def _mem_summary(tag: str, device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(int(device.split(":")[1]))
        print(f"[{tag}] GPU {device} free {free // 2**20} MiB total {total // 2**20} MiB")
    print(f"[{tag}] torch.cuda.memory_allocated = {torch.cuda.memory_allocated() // 2**20} MiB")
    print(f"[{tag}] torch.cuda.memory_reserved  = {torch.cuda.memory_reserved()  // 2**20} MiB")

class T5:
    def __init__(self, model_config: dict[str, Any], device: str, amp: bool) -> None:
        print("=== T5 wrapper init ===")
        print(f"requested device = {device}  amp={amp}")
        self.model = T5Modyn(model_config, device)

class T5Modyn(CoresetSupportingModule):
    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        super().__init__()
        print("=== T5Modyn init ===")
        print("env ACCELERATE_DISABLE_BIG_MODEL_INFERENCE =", os.getenv("ACCELERATE_DISABLE_BIG_MODEL_INFERENCE"))
        print("incoming model_config =", json.dumps(model_config, indent=2))

        # Debug: print relevant environment variables
        
        
        self.model_name = model_config.get("model_name_or_path", "t5-base")
        self.dtype = model_config.get("dtype", torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        _mem_summary("before load", device)
        print("calling from_pretrained …")

        # Load model and immediately move to device
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(device)

        print("from_pretrained done")
        _mem_summary("after load", device)

        # Detailed parameter inspection
        meta_count = 0
        for n, p in self.model.named_parameters():
            is_meta = p.is_meta
            print(f"PARAM {n} device={p.device} dtype={p.dtype} is_meta={is_meta}")
            if is_meta:
                print(f"META PARAM  {n}  shape={tuple(p.shape)}")
                meta_count += 1
        print(f"total meta params = {meta_count}")

        if meta_count:
            raise RuntimeError("meta tensors remain — aborting")

        print("model successfully loaded on", device)
        print("dtype distribution:",
              {p.dtype: sum(p.numel() for p in self.model.parameters() if p.dtype == d)
               for d in {p.dtype for p in self.model.parameters()}})

    def forward(self, data: torch.Tensor, labels: torch.Tensor | None = None, *, tag: str = "fwd"):
        input_ids, attention_mask = data[:, :, 0], data[:, :, 1]

        # ---- single call that *includes* the prompt hook ---------------------
        if labels is not None:
            
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,                 # may be None at inference
                output_hidden_states=True,     # so we can pool encoder reps
                return_dict=True,
            )
        else:
            encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            )
            last_hidden = encoder_outputs.last_hidden_state        # (batch, seq_len, dim)
            pooled = last_hidden.mean(dim=1)
            self.embedding_recorder(pooled)
            return pooled


        # encoder hidden states live at out.encoder_last_hidden_state
        enc_last = out.encoder_last_hidden_state          # (B, src_len+prompt, D)
        pooled   = enc_last.mean(dim=1)                   # simple global pool
        self.embedding_recorder(pooled)
        print(f"[{tag}] pooled grad={pooled.requires_grad}")
        return out.logits



    def get_last_layer(self) -> nn.Module:
        return self.model.lm_head

    def freeze_params(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_params(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def generate(self, data: torch.Tensor) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            early_stopping=True,
            do_sample=True,
        )
        return (
            F.pad(
                outputs,
                (0, self.max_length - outputs.shape[1]),
                value=self.tokenizer.pad_token_id,
            ) if outputs.shape[1] < self.max_length else outputs
        )
