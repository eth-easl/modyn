import types
from typing import Any

import torch
from peft import LoraConfig, PrefixTuningConfig, PromptTuningConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class ModularAdapter(nn.Module):
    """A generic, adapter module.

    It projects the hidden states down to a lower-dimensional space,
    applies an activation,projects back up to the original dimension,
    and adds the result via a residual connection.
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.activation = activation()
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_out = self.down_proj(hidden_states)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.up_proj(adapter_out)
        adapter_out = self.dropout(adapter_out)
        return hidden_states + adapter_out


class AdapterModel(nn.Module):
    """A model-agnostic adapter container that applies generic adapters at
    specified layers.

    The adapter modules are applied to the hidden states from the
    corresponding layers. The final output is the original sequence
    output augmented by a scaled, normalized adapter fusion.
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        hidden_size: int,
        adapter_layers: list[int] = [1, 11],
        adapter_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_layers = adapter_layers  # Layers at which to apply adapters
        self.num_adapters = len(adapter_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Create a list of generic adapters (one per specified layer)
        self.adapters = nn.ModuleList(
            [ModularAdapter(hidden_size, adapter_size=adapter_size, dropout=dropout) for _ in range(self.num_adapters)]
        )
        self.scale_factor = 0.1

    def forward(self, model_outputs: Any) -> torch.Tensor:
        """Applies the adapters to the provided model outputs.
        
        Expects model_outputs to be a CausalLMOutputWithCrossAttentions
        like object, where hidden_states is provided.
        """
        sequence_output = model_outputs[0]
        hidden_states = model_outputs.hidden_states
        fusion_state = torch.zeros_like(sequence_output)
        for i, adapter in enumerate(self.adapters):
            layer_idx = self.adapter_layers[i]
            adapter_input = hidden_states[layer_idx] + fusion_state
            fusion_state = adapter(adapter_input)
        # Normalize fusion state and add a scaled residual to the original sequence output.
        return sequence_output + self.scale_factor * self.layer_norm(fusion_state)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Applies LoRA to the model using the PEFT library.

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

    print(f"\nTrainable parameters BEFORE applying LoRA: {count_trainable_params(model)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_dim,
        lora_alpha=adapter_alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    # Apply LoRA to the inner model. This replaces some of the original layers with LoRA layers.
    model.model = get_peft_model(model.model, lora_config)

    print(f"\nTrainable parameters AFTER applying LoRA: {count_trainable_params(model)}")
    return model


def apply_prompt_tuning(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Applies Prompt Tuning with soft prompt tokens prepended to the input.

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

    print(f"\nTrainable parameters AFTER applying Prompt Tuning: {count_trainable_params(model)}")
    return model


def apply_prefix_tuning(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Applies Prefix Tuning with trainable prefix tokens.

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

    print(f"\nTrainable parameters AFTER applying Prefix Tuning: {count_trainable_params(model)}")
    return model


def apply_kadapter(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Applies a modular adapter (KaAdapter) by injecting generic adapter
    modules at specified layers.

    Kwargs:
        adapter_layers (list[int]): Layers to inject adapters. Default = [1, 11]
        scale_factor (float): Scale for adapter output fusion. Default = 0.1
        adapter_size (int): The bottleneck dimension in the adapter. Default = 64
        dropout (float): Dropout rate within the adapter. Default = 0.1
    """
    adapter_layers = kwargs.get("adapter_layers", [1, 11])
    scale_factor = kwargs.get("scale_factor", 0.1)
    adapter_size = kwargs.get("adapter_size", 64)
    dropout = kwargs.get("dropout", 0.1)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.model.config.hidden_size

    # Attach the adapter as a submodule of the inner wrapped model.
    model.model.kadapter = AdapterModel(
        hidden_size, adapter_layers=adapter_layers, adapter_size=adapter_size, dropout=dropout
    )
    model.model.kadapter.scale_factor = scale_factor

    # Ensure that the adapter’s parameters are trainable.
    for param in model.model.kadapter.parameters():
        param.requires_grad = True

    def forward_with_adapter(  # type: ignore[no-untyped-def]
        self,
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
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        model_outputs = self.transformer(
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

        adapted_hidden = self.kadapter(model_outputs)
        lm_logits = self.lm_head(adapted_hidden)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            cross_attentions=model_outputs.cross_attentions,
        )

    # Override the forward method on the inner model with our new method.
    model.model.forward = types.MethodType(forward_with_adapter, model.model)
    return model


def apply_adapters(model: nn.Module, adapters: list[str], adapter_args: dict[str, dict[str, Any]]) -> nn.Module:
    """Applies all specified adapters with their arguments to the model.

    Args:
        model: The model to apply adapters to.
        adapters: List of adapter names (e.g., "lora", "kadapter", "prompt_tuning",
          "prefix_tuning").
        adapter_args: Dictionary mapping each adapter name to its configuration kwargs.
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
