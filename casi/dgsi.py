# Adapted from the supplied Apache-2.0 Hugging Face diffusers attention processor.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Defect-Guided Selective Interaction (DGSI), Section 2.3.

Pairs are interleaved [global_0, defect_0, global_1, defect_1, ...].
Legacy temporal_* tensor keys are retained solely for checkpoint compatibility.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError("Invalid LoRA rank")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        result = self.up(self.down(hidden_states.to(self.down.weight.dtype)))
        if self.network_alpha is not None:
            result = result * self.network_alpha / self.rank
        return result.to(hidden_states.dtype)


class DGSI(nn.Module):
    """Branch-specific LoRA with asymmetric attention and a defect-conditioned gate."""

    def __init__(self, hidden_size, cross_attention_dim=None, rank=32, enable_interaction=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.enable_interaction = enable_interaction
        for name, input_size in (
            ("to_q_lora", hidden_size),
            ("to_k_lora", cross_attention_dim or hidden_size),
            ("to_v_lora", cross_attention_dim or hidden_size),
            ("to_out_lora", hidden_size),
        ):
            setattr(self, name, nn.ModuleList(
                LoRALinearLayer(input_size, hidden_size, rank) for _ in range(2)
            ))
        if enable_interaction:
            self.temporal_i = nn.Linear(hidden_size, hidden_size)
            self.temporal_n = nn.LayerNorm(hidden_size, eps=1e-6)
            self.temporal_q = nn.Linear(hidden_size, hidden_size)
            self.temporal_k = nn.Linear(hidden_size, hidden_size)
            self.temporal_v = nn.Linear(hidden_size, hidden_size)
            self.temporal_o = nn.Linear(hidden_size, hidden_size)
            self.asym_gate = nn.Parameter(torch.tensor([0.1, 1.0]))
            self.spatial_gate_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.spatial_gate_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4), nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
            )
            nn.init.zeros_(self.spatial_gate_mlp[-1].weight)
            nn.init.constant_(self.spatial_gate_mlp[-1].bias, -2.0)

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        if hidden_states.ndim != 3 or hidden_states.shape[0] % 2:
            raise ValueError("DGSI expects [2 * batch, spatial_tokens, channels]")
        residual = hidden_states
        pair_count, sequence_length, channels = hidden_states.shape
        batch_size = pair_count // 2
        branches = hidden_states.reshape(batch_size, 2, sequence_length, channels)
        context = None
        if encoder_hidden_states is not None:
            context = encoder_hidden_states.reshape(batch_size, 2, *encoder_hidden_states.shape[1:])
        outputs = []
        for branch in range(2):
            hidden = branches[:, branch]
            encoder = context[:, branch] if context is not None else None
            if attn.group_norm is not None:
                hidden = attn.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden) + scale * self.to_q_lora[branch](hidden)
            if encoder is None:
                encoder = hidden
            elif attn.norm_cross:
                encoder = attn.norm_encoder_hidden_states(encoder)
            key = attn.to_k(encoder) + scale * self.to_k_lora[branch](encoder)
            value = attn.to_v(encoder) + scale * self.to_v_lora[branch](encoder)
            head_dim = query.shape[-1] // attn.heads
            query, key, value = (
                tensor.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                for tensor in (query, key, value)
            )
            mask = None
            if attention_mask is not None:
                branch_mask = attention_mask[branch::2] if attention_mask.shape[0] == pair_count else attention_mask
                mask = attn.prepare_attention_mask(branch_mask, encoder.shape[1], batch_size)
                mask = mask.reshape(batch_size, attn.heads, -1, mask.shape[-1])
            hidden = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
            hidden = hidden.transpose(1, 2).reshape(batch_size, -1, channels).to(query.dtype)
            hidden = attn.to_out[0](hidden) + scale * self.to_out_lora[branch](hidden)
            outputs.append(attn.to_out[1](hidden))
        hidden_states = torch.stack(outputs, dim=1).reshape(pair_count, sequence_length, channels)

        if self.enable_interaction:
            input_dtype = hidden_states.dtype
            features = hidden_states.to(self.temporal_n.weight.dtype)
            defect_features = features.reshape(batch_size, 2, sequence_length, channels)[:, 1]
            spatial_gate = torch.sigmoid(self.spatial_gate_mlp(self.spatial_gate_norm(defect_features)))
            spatial_gate = spatial_gate[:, None].expand(-1, 2, -1, -1).reshape(pair_count, sequence_length, 1)

            exchange = self.temporal_i(self.temporal_n(features))
            exchange = exchange.reshape(batch_size, 2, sequence_length, channels).permute(0, 2, 1, 3)
            exchange = exchange.reshape(batch_size * sequence_length, 2, channels)
            query, key, value = (projection(exchange) for projection in (
                self.temporal_q, self.temporal_k, self.temporal_v,
            ))
            direction_priors = torch.sigmoid(self.asym_gate)
            bias = exchange.new_zeros((1, 2, 2))
            bias[0, 0, 1] = torch.log(direction_priors[0] + 1e-8)  # defect -> global
            bias[0, 1, 0] = torch.log(direction_priors[1] + 1e-8)  # global -> defect
            exchange = F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.0)
            exchange = self.temporal_o(exchange)
            exchange = exchange.reshape(batch_size, sequence_length, 2, channels).permute(0, 2, 1, 3)
            exchange = exchange.reshape(pair_count, sequence_length, channels)
            hidden_states = hidden_states + (exchange * spatial_gate).to(input_dtype)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_dgsi(unet, rank=32):
    processors = {}
    for name in unet.attn_processors:
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden_size = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
        else:
            hidden_size = unet.config.block_out_channels[int(name.split(".")[1])]
        cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # Keep the final supplied configuration: no interaction in down_blocks 0/1.
        enabled = not name.startswith(("down_blocks.0.", "down_blocks.1."))
        processors[name] = DGSI(hidden_size, cross_dim, rank, enabled)
    unet.set_attn_processor(processors)
    return nn.ModuleList(list(unet.attn_processors.values()))


def save_dgsi(unet, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        f"unet.{name}.{key}": tensor.detach().cpu().clone()
        for name, processor in unet.attn_processors.items()
        for key, tensor in processor.state_dict().items()
    }
    torch.save(state, directory / "pytorch_lora_weights.bin")


def load_dgsi(unet, directory, replace=True):
    """Load both cleaned checkpoints and legacy scalar-direction/spatial-gate weights."""
    directory = Path(directory)
    if directory.is_file():
        path = directory
    else:
        paths = [directory / name for name in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")]
        path = next((path for path in paths if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No DGSI/LoRA weights in {directory}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("unet."): value for key, value in state.items()}
    remaining = set(state)
    processors = {}
    for name in unet.attn_processors:
        prefix = name + "."
        values = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not values:
            raise ValueError(f"Missing processor: {name}")
        down = values["to_q_lora.0.down.weight"]
        cross_dim = values["to_k_lora.0.down.weight"].shape[1] if name.endswith("attn2.processor") else None
        enabled = "asym_gate" in values
        processor = DGSI(down.shape[1], cross_dim, down.shape[0], enabled) if replace else unet.attn_processors[name]
        processor.load_state_dict(values, strict=True)
        processors[name] = processor
        remaining.difference_update(prefix + key for key in values)
    if remaining:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(remaining)[:5]}")
    if replace:
        device, dtype = next(unet.parameters()).device, next(unet.parameters()).dtype
        unet.set_attn_processor(processors)
        nn.ModuleList(list(unet.attn_processors.values())).to(device=device, dtype=dtype)
