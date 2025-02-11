import types
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# =============================================================================
# Custom Adapter Model (KAdapter) defined manually
# =============================================================================
# You must also have GPT2Block imported from your GPT-2 implementation.
# For this example, we assume it’s available as follows:
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class AdapterModel(nn.Module):
    def __init__(self, pretrained_config: GPT2Config | None = None) -> None:
        self.config = pretrained_config
        if self.config is None:
            self.config = GPT2Config.from_pretrained("gpt2-large")
        super().__init__()

        self.embed_dim = self.config.hidden_size
        # Define which layers to pull hidden states from
        self.adapter_list = [1, 11]  # For example, use layers 1 and 11
        self.adapter_num = len(self.adapter_list)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)
        # Create an adapter (here using GPT2Block) for each designated layer
        self.adapter = nn.ModuleList([GPT2Block(self.config) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs: CausalLMOutputWithCrossAttentions) -> torch.Tensor:
        # Assume pretrained_model_outputs is a ModelOutput with hidden_states and [0]=final hidden state.
        sequence_output = pretrained_model_outputs[0]
        hidden_states = pretrained_model_outputs.hidden_states
        # Determine device from sequence_output
        device = sequence_output.device
        hidden_states_last = torch.zeros(sequence_output.size(), device=device)

        for i, adapter_module in enumerate(self.adapter):
            # Get hidden state from the designated layer
            pretrained_hidden_state = hidden_states[self.adapter_list[i]]
            # Fuse with previously adapter-processed output

            fusion_state = pretrained_hidden_state + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)[0]

        scale_factor = 0.1
        # Fuse adapter output (after normalization/scaling) with the final hidden state
        outputs = (scale_factor * self.layer_norm(hidden_states_last)) + sequence_output
        return outputs


# =============================================================================
# LoRA layer & apply_lora remains unchanged
# =============================================================================


def apply_lora(
    model: nn.Module, target_modules: list[str] | None = None, adapter_dim: int = 16, adapter_alpha: int = 32
) -> nn.Module:
    # Use default target modules for GPT-2 if not provided.
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]

    # Create a LoRA configuration.
    # Here, `r` corresponds to the adapter (low-rank) dimension.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_dim,
        lora_alpha=adapter_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )

    # Wrap the model with the LoRA adapter.
    # This call freezes the base parameters and adds trainable LoRA parameters.
    model.model = get_peft_model(model.model, lora_config)
    return model


def apply_kadapter(
    model: nn.Module,
) -> nn.Module:
    # Freeze base parameters
    if hasattr(model, "freeze_params"):
        model.freeze_params()
    else:
        for param in model.parameters():
            param.requires_grad = False

    # Ensure the model has a 'config' attribute

    # (Optional) Print some config details for debugging

    # Attach the custom adapter; AdconfigapterModel should be defined to use model.config.hidden_size, etc.
    model.kadapter = AdapterModel(model.config)

    for param in model.kadapter.parameters():
        param.requires_grad = True

    # Define a new forward that integrates the adapter output.
    def forward_with_adapter(
        self: nn.Module,
        data: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool = True,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = data[:, :, 0]

        attention_mask = data[:, :, 1]
        model_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # If hidden states are returned, pass them through the adapter.
        if model_outputs.hidden_states is not None:
            hidden_states = self.kadapter(model_outputs)
        else:
            hidden_states = model_outputs[0]

        lm_logits = self.model.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            cross_attentions=model_outputs.cross_attentions,
        )

    # Replace the model's forward with the new forward
    model.forward = types.MethodType(forward_with_adapter, model)
    return model
