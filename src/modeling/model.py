import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Optional


class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        clip_emb_dim: int,
        visual_tokens_length: int,  # number of visual tokens from CLIP encoder (prefix)
        gpt_model: str = "gpt2",
    ) -> None:

        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # Projection from CLIP embedding to GPT-2 input space
        # [MEDIUM] : add Projection MLP params in config file
        self.projector = nn.Sequential(
            nn.Linear(clip_emb_dim, self.gpt_embedding_size * visual_tokens_length),
            nn.Tanh(),
        )

        self.visual_tokens_length = visual_tokens_length

    def forward(
        self,
        text_tokens: torch.Tensor,
        clip_embed: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Project CLIP embedding to GPT-2 embedding space
        visual_embeddings = self.projector(clip_embed).view(
            -1, self.visual_tokens_length, self.gpt_embedding_size
        )

        # Text token embeddings from GPT-2
        text_embeddings = self.gpt.transformer.wte(text_tokens)

        # Concatenate CLIP-projected visual tokens with text tokens
        gpt_input = torch.cat((visual_embeddings, text_embeddings), dim=1)

        # Adjust labels to match input size
        if labels is not None:
            dummy_visual_tokens = torch.full(
                (visual_embeddings.shape[0], self.visual_tokens_length),
                fill_value=-100,  # ignore prefix tokens for loss calculation
                dtype=torch.long,
                device=visual_embeddings.device,
            )
            labels = torch.cat((dummy_visual_tokens, text_tokens), dim=1)

        # Forward through GPT-2
        output = self.gpt(inputs_embeds=gpt_input, labels=labels)

        return output

    def generate(self, clip_embed: torch.Tensor, **generate_kwargs):
        visual_embeddings = self.projector(clip_embed).view(
            -1, self.visual_tokens_length, self.gpt_embedding_size
        )

        return self.gpt.generate(inputs_embeds=visual_embeddings, **generate_kwargs)
