import types
from typing import Any

import torch
from peft import LoraConfig, PrefixTuningConfig, PromptTuningConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
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


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora(  # pylint: disable=dangerous-default-value
    model: nn.Module,
    target_modules: list[str] | None = ["c_attn", "c_proj"],
    adapter_dim: int = 16,
    adapter_alpha: int = 32,
) -> nn.Module:
    # Count and print trainable parameters before applying LoRA
    trainable_params_before = count_trainable_params(model)
    print(f"\n Trainable parameters BEFORE applying LoRA: {trainable_params_before}")

    # Create a LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_dim,
        lora_alpha=adapter_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )

    # Wrap the model with LoRA
    model.model = get_peft_model(model.model, lora_config)

    # Count and print trainable parameters after applying LoRA
    trainable_params_after = count_trainable_params(model)
    print(f"\n Trainable parameters AFTER applying LoRA: {trainable_params_after}")

    return model


# Recommended LoRA configuration based on the provided BERT-LoRA code:
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # or "SEQ_CLS", "TOKEN_CLS" based on use case
#     r=16,                          # Matches 'lora_r' from the BERT-LoRA code
#     lora_alpha=32,                 # Matches 'lora_alpha' from the BERT-LoRA code
#     lora_dropout=0.0,              # Default in the paper
#     target_modules=["query", "value"]  # Explicitly matches BERT paper code
# )


def apply_kadapter(
    model: nn.Module,
) -> nn.Module:
    # Freeze base parameters
    if hasattr(model.model, "freeze_params"):
        model.model.freeze_params()
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
        self: Any,
        input_ids: Any,
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

        model_outputs = self.model.transformer(
            input_ids=input_ids,
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
    model.model.forward = types.MethodType(forward_with_adapter, model)
    return model


def apply_prompt_tuning(
    model: nn.Module,
    num_virtual_tokens: int = 20,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> nn.Module:
    """Applies prompt tuning as an adapter to the model."""
    prompt_config = PromptTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
    )
    model.model = get_peft_model(model.model, prompt_config)
    print(f"\n Trainable parameters AFTER applying Prompt Tuning: {count_trainable_params(model)}")


def apply_prefix_tuning(
    model: nn.Module,
    prefix_length: int = 30,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> nn.Module:
    """Applies prefix tuning as an adapter to the model."""
    prefix_config = PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=prefix_length,
    )
    model.model = get_peft_model(model.model, prefix_config)
    print(f"\n Trainable parameters AFTER applying Prefix Tuning: {count_trainable_params(model)}")
    return model


def apply_adapters(model: nn.Module, adapters: list[str]) -> nn.Module:
    """Reads adapter names from a list and applies them to the model sequentially."""
    for adapter in adapters:
        adapter_lower = adapter.lower()
        if adapter_lower == "lora":
            model = apply_lora(model)
        elif adapter_lower == "kadapter":
            model = apply_kadapter(model)
        elif adapter_lower == "prompt_tuning":
            model = apply_prompt_tuning(model)
        elif adapter_lower == "prefix_tuning":
            model = apply_prefix_tuning(model)
        else:
            raise ValueError(f"Unknown adapter type: {adapter}")
    return model
