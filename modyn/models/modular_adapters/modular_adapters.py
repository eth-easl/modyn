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
    def __init__(self, pretrained_config: GPT2Config | None = None, adapter_layers: list[int] = [1, 11]) -> None:  # pylint: disable= dangerous-default-value
        self.config = pretrained_config or GPT2Config.from_pretrained("gpt2-large")
        super().__init__()

        self.embed_dim = self.config.hidden_size
        self.adapter_list = adapter_layers  # Adapter layers can now be customized
        self.adapter_num = len(self.adapter_list)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)
        self.adapter = nn.ModuleList([GPT2Block(self.config) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs: CausalLMOutputWithCrossAttentions) -> torch.Tensor:
        sequence_output = pretrained_model_outputs[0]
        hidden_states = pretrained_model_outputs.hidden_states
        device = sequence_output.device
        hidden_states_last = torch.zeros(sequence_output.size(), device=device)

        for i, adapter_module in enumerate(self.adapter):
            pretrained_hidden_state = hidden_states[self.adapter_list[i]]
            fusion_state = pretrained_hidden_state + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)[0]

        scale_factor = getattr(self, "scale_factor", 0.1)
        outputs = (scale_factor * self.layer_norm(hidden_states_last)) + sequence_output
        return outputs


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora(model: nn.Module, **kwargs: Any) -> nn.Module:
    """
    Applies LoRA to the model using the PEFT library.

    Kwargs:
        target_modules (list[str]): Default = ["c_attn", "c_proj"]
        adapter_dim (int): LoRA rank (r). Default = 16
        adapter_alpha (int): LoRA alpha. Default = 32
        dropout (float): LoRA dropout. Default = 0.0
    """
    target_modules = kwargs.get("target_modules", ["c_attn", "c_proj"])
    adapter_dim = kwargs.get("adapter_dim", 16)
    adapter_alpha = kwargs.get("adapter_alpha", 32)
    dropout = kwargs.get("dropout", 0.0)

    print(f"\n Trainable parameters BEFORE applying LoRA: {count_trainable_params(model)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_dim,
        lora_alpha=adapter_alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model.model = get_peft_model(model.model, lora_config)

    print(f"\n Trainable parameters AFTER applying LoRA: {count_trainable_params(model)}")
    return model


def apply_prompt_tuning(model: nn.Module, **kwargs: Any) -> nn.Module:
    """
    Applies Prompt Tuning as soft prompt tokens prepended to input.

    Kwargs:
        num_virtual_tokens (int): Default = 20
        task_type (TaskType): Default = TaskType.CAUSAL_LM
    """
    num_virtual_tokens = kwargs.get("num_virtual_tokens", 20)
    task_type = kwargs.get("task_type", TaskType.CAUSAL_LM)

    prompt_config = PromptTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
    )
    model.model = get_peft_model(model.model, prompt_config)

    print(f"\n Trainable parameters AFTER applying Prompt Tuning: {count_trainable_params(model)}")
    return model


def apply_prefix_tuning(model: nn.Module, **kwargs: Any) -> nn.Module:
    """
    Applies Prefix Tuning with trainable prefix tokens.

    Kwargs:
        prefix_length (int): Default = 30
        task_type (TaskType): Default = TaskType.CAUSAL_LM
    """
    prefix_length = kwargs.get("prefix_length", 30)
    task_type = kwargs.get("task_type", TaskType.CAUSAL_LM)

    prefix_config = PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=prefix_length,
    )
    model.model = get_peft_model(model.model, prefix_config)

    print(f"\n Trainable parameters AFTER applying Prefix Tuning: {count_trainable_params(model)}")
    return model


def apply_kadapter(model: nn.Module, **kwargs: Any) -> nn.Module:
    """
    Applies KaAdapter by injecting transformer blocks at specific layers.

    Kwargs:
        adapter_layers (list[int]): Layers to inject adapters. Default = [1, 11]
        scale_factor (float): Scale for adapter output fusion. Default = 0.1
    """
    adapter_layers = kwargs.get("adapter_layers", [1, 11])
    scale_factor = kwargs.get("scale_factor", 0.1)

    if hasattr(model.model, "freeze_params"):
        model.model.freeze_params()
    else:
        for param in model.parameters():
            param.requires_grad = False

    model.kadapter = AdapterModel(model.config, adapter_layers=adapter_layers)
    model.kadapter.scale_factor = scale_factor

    for param in model.kadapter.parameters():
        param.requires_grad = True

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

    model.model.forward = types.MethodType(forward_with_adapter, model)
    return model


def apply_adapters(model: nn.Module, adapters: list[str], adapter_args: dict[str, dict[str, Any]]) -> nn.Module:
    """
    Applies all specified adapters with their arguments to the model.

    Args:
        model: The model to apply adapters to.
        adapters: List of adapter names (lora, kadapter, etc.)
        adapter_args: Dict mapping adapter names to kwargs.
    """
    for adapter in adapters:
        adapter_lower = adapter.lower()
        args = adapter_args.get(adapter_lower, {})

        if adapter_lower == "lora":
            model = apply_lora(model, **args)
        elif adapter_lower == "kadapter":
            model = apply_kadapter(model, **args)
        elif adapter_lower == "prompt_tuning":
            model = apply_prompt_tuning(model, **args)
        elif adapter_lower == "prefix_tuning":
            model = apply_prefix_tuning(model, **args)

    return model
