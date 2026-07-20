"""Baichuan sequence-classification wrapper for the v1 classification-head workflow."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BaichuanForSequenceClassification(PreTrainedModel):
    """Wrap Baichuan causal LM with a linear sequence-classification head.

    Baichuan2 does not provide an AutoModelForSequenceClassification mapping in
    Transformers. This wrapper keeps the Baichuan decoder backbone and adds a
    standard linear `score` head on top of the last non-padding token.
    """

    base_model_prefix = "baichuan"
    supports_gradient_checkpointing = True

    def __init__(self, baichuan_model: PreTrainedModel, num_labels: int, id2label=None, label2id=None):
        super().__init__(baichuan_model.config)
        self.baichuan = baichuan_model
        self.config = baichuan_model.config
        self.config.num_labels = num_labels
        self.num_labels = num_labels
        if id2label is not None:
            self.config.id2label = id2label
        if label2id is not None:
            self.config.label2id = label2id
        if getattr(self.config, "pad_token_id", None) is None and getattr(self.config, "eos_token_id", None) is not None:
            self.config.pad_token_id = self.config.eos_token_id

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Baichuan config does not expose hidden_size, cannot build classification head.")
        self.score = nn.Linear(int(hidden_size), num_labels, bias=False)
        device, dtype = self._infer_head_device_dtype()
        self.score.to(device=device, dtype=dtype)

    def _infer_head_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        for parameter in self.baichuan.parameters():
            if getattr(parameter, "is_meta", False):
                continue
            if parameter.is_floating_point():
                return parameter.device, parameter.dtype
        for parameter in self.baichuan.parameters():
            if not getattr(parameter, "is_meta", False):
                return parameter.device, torch.float32
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), torch.float32

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, num_labels: int, id2label=None, label2id=None, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(base_model, num_labels=num_labels, id2label=id2label, label2id=label2id)

    def get_input_embeddings(self):
        return self.baichuan.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.baichuan.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.baichuan, "gradient_checkpointing_enable"):
            try:
                return self.baichuan.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                return self.baichuan.gradient_checkpointing_enable()
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.baichuan, "gradient_checkpointing_disable"):
            return self.baichuan.gradient_checkpointing_disable()
        return super().gradient_checkpointing_disable()

    def _decoder_forward(self, input_ids=None, attention_mask=None, **kwargs):
        decoder = getattr(self.baichuan, "model", None)
        if decoder is not None:
            return decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=kwargs.pop("output_hidden_states", False),
                return_dict=True,
                **kwargs,
            )
        return self.baichuan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        sequence_lengths = attention_mask.to(hidden_states.device).sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths, :]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=True,
        **kwargs,
    ):
        kwargs.pop("use_cache", None)
        outputs = self._decoder_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=bool(output_hidden_states),
            output_attentions=output_attentions,
            use_cache=False,
            **kwargs,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            if getattr(outputs, "hidden_states", None) is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs[0]

        pooled = self._last_token_pool(hidden_states, attention_mask)
        logits = self.score(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (getattr(outputs, "hidden_states", None),)
            if output_attentions:
                output += (getattr(outputs, "attentions", None),)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
