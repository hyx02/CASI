"""Category-Shared Object-Defect Disentanglement (CODD), Section 2.2.

The object/defect adapter parameter names match the supplied checkpoints.
"""

import json
from itertools import combinations
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size=768, bottleneck=192, init_scale=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        return hidden_states + self.scale * self.up(self.act(self.down(hidden_states)))


def find_token_positions(tokenizer, prompt, keywords):
    tokens = tokenizer.tokenize(prompt)
    positions = set()
    for keyword in keywords:
        if not keyword.strip():
            continue
        keyword_tokens = tokenizer.tokenize(keyword.lower().replace("_", " "))
        for start in range(len(tokens) - len(keyword_tokens) + 1):
            if all(
                tokens[start + k].replace("</w>", "") == token.replace("</w>", "")
                for k, token in enumerate(keyword_tokens)
            ):
                positions.update(range(start + 1, start + len(keyword_tokens) + 1))
    return positions


def build_token_masks(tokenizer, prompt, object_keywords, defect_keywords, seq_len=77):
    masks = []
    for keywords in (object_keywords, defect_keywords):
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for position in find_token_positions(tokenizer, prompt, keywords):
            if position < seq_len - 1:  # Leave BOS/EOS and padding unchanged.
                mask[position] = True
        masks.append(mask)
    return tuple(masks)


def make_prompt(category, defect="", branch="global"):
    category = category.replace("_", " ")
    defect = defect.replace("_", " ")
    if branch == "defect":
        return f"a photo of a {defect}"
    if defect:
        return f"a photo of a {category} with a {defect}"
    return f"a photo of a {category}"


class CODD(nn.Module):
    """Frozen CLIP with token-routed object and defect adapters (Eq. 2)."""

    def __init__(self, text_encoder, num_adapter_layers=4, bottleneck=192):
        super().__init__()
        layers = text_encoder.text_model.encoder.layers
        if not 1 <= num_adapter_layers <= len(layers):
            raise ValueError("num_adapter_layers must fit the CLIP encoder")
        self.text_encoder = text_encoder.requires_grad_(False)
        self.num_adapter_layers = num_adapter_layers
        hidden_size = text_encoder.config.hidden_size
        self.object_adapters = nn.ModuleList(
            ResidualAdapter(hidden_size, bottleneck) for _ in range(num_adapter_layers)
        )
        self.defect_adapters = nn.ModuleList(
            ResidualAdapter(hidden_size, bottleneck) for _ in range(num_adapter_layers)
        )
        self._masks = None
        self._hooks = [
            layers[i].register_forward_hook(self._make_hook(i))
            for i in range(num_adapter_layers)
        ]

    def _make_hook(self, index):
        def hook(module, inputs, output):
            if self._masks is None:
                return output
            hidden = output[0]
            object_mask, defect_mask = (
                mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
                for mask in self._masks
            )
            object_delta = self.object_adapters[index](hidden) - hidden
            defect_delta = self.defect_adapters[index](hidden) - hidden
            hidden = hidden + object_mask * object_delta + defect_mask * defect_delta
            return (hidden,) + output[1:]
        return hook

    def adapter_parameters(self):
        return list(self.object_adapters.parameters()) + list(self.defect_adapters.parameters())

    def encode_texts(self, tokenizer, categories, defects, branch="global"):
        if len(categories) != len(defects) or not categories:
            raise ValueError("categories and defects must have the same nonzero length")
        if branch not in {"global", "defect"}:
            raise ValueError(f"Unknown branch: {branch}")
        prompts = [make_prompt(c, d, branch) for c, d in zip(categories, defects)]
        object_masks, defect_masks = [], []
        for prompt, category, defect in zip(prompts, categories, defects):
            object_mask, defect_mask = build_token_masks(
                tokenizer, prompt,
                [category] if branch == "global" else [],
                [defect] if defect else [],
                tokenizer.model_max_length,
            )
            if branch == "global" and not object_mask.any():
                raise ValueError(f"Object token not found in prompt: {prompt}")
            if defect and not defect_mask.any():
                raise ValueError(f"Defect token not found or truncated: {prompt}")
            object_masks.append(object_mask)
            defect_masks.append(defect_mask)
        ids = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.to(next(self.parameters()).device)
        self._masks = torch.stack(object_masks), torch.stack(defect_masks)
        try:
            return self.text_encoder(ids)[0]
        finally:
            self._masks = None

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({
            "object_adapters": self.object_adapters.state_dict(),
            "defect_adapters": self.defect_adapters.state_dict(),
        }, directory / "adapters.pt")
        config = {
            "num_adapter_layers": self.num_adapter_layers,
            "hidden_size": self.text_encoder.config.hidden_size,
            "bottleneck": self.object_adapters[0].down.out_features,
        }
        (directory / "codd_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, text_encoder, directory):
        directory = Path(directory)
        if (directory / "codd_config.json").exists():
            config = json.loads((directory / "codd_config.json").read_text(encoding="utf-8"))
        else:  # Read adapter checkpoints produced by the supplied code.
            config = torch.load(directory / "adapter_config.pt", map_location="cpu", weights_only=True)
        model = cls(text_encoder, config["num_adapter_layers"], config["bottleneck"])
        state = torch.load(directory / "adapters.pt", map_location="cpu", weights_only=True)
        model.object_adapters.load_state_dict(state["object_adapters"], strict=True)
        model.defect_adapters.load_state_dict(state["defect_adapters"], strict=True)
        return model


def semantic_losses(codd, tokenizer, category, all_defects, batch_defects, margin=0.5):
    """L_dis1, L_dis2 and L_obj; retain the supplied implementation's MSE reduction."""
    normal = codd.encode_texts(tokenizer, [category], [""]).mean(dim=1)
    present = sorted(set(batch_defects))
    anomalies = codd.encode_texts(tokenizer, [category] * len(present), present).mean(dim=1)
    loss_dis1 = F.cosine_similarity(normal.expand_as(anomalies), anomalies, dim=-1).abs().mean()

    hidden = codd.encode_texts(tokenizer, [category] * len(all_defects), all_defects)
    vectors = hidden.mean(dim=1)
    object_vectors = []
    for i, defect in enumerate(all_defects):
        object_mask, _ = build_token_masks(
            tokenizer, make_prompt(category, defect), [category], [], hidden.shape[1],
        )
        object_vectors.append(hidden[i, object_mask.to(hidden.device)].mean(dim=0))
    distance_losses, object_losses = [], []
    for i, j in combinations(range(len(all_defects)), 2):
        distance_losses.append(F.relu(margin - (vectors[i] - vectors[j]).norm(p=2)))
        object_losses.append(F.mse_loss(object_vectors[i], object_vectors[j]))
    zero = hidden.sum() * 0.0
    loss_dis2 = torch.stack(distance_losses).mean() if distance_losses else zero
    loss_obj = torch.stack(object_losses).mean() if object_losses else zero
    return loss_dis1, loss_dis2, loss_obj
