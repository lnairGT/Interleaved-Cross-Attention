from collections import OrderedDict
import numpy as np
import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_cross: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_head, kdim=d_cross, vdim=d_cross, batch_first=True, dropout=0.1
        )
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.1)

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def cross_attention(self, x: torch.Tensor, x2: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.cross_attn(x, x2, x2, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, x2: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.cross_attention(self.ln_2(x), x2)
        x = x + self.mlp(self.ln_3(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.1)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        assert transformer_heads == vision_heads
        assert transformer_layers == vision_layers

        # Vision Model
        self.input_resolution = image_resolution
        self.output_dim = embed_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=vision_width,
            kernel_size=vision_patch_size,
            stride=vision_patch_size,
            bias=False
        )
        scale = vision_width ** -0.5
        self.vis_class_embedding = nn.Parameter(scale * torch.randn(vision_width))
        self.vis_positional_embedding = nn.Parameter(
            scale * torch.randn((image_resolution // vision_patch_size) ** 2 + 1, vision_width)
        )
        self.vis_ln_pre = LayerNorm(vision_width)
        self.vis_transformer = torch.nn.ModuleList()
        for _ in range(vision_layers):
            self.vis_transformer.append(
                CrossAttentionBlock(
                    vision_width, transformer_width, vision_heads
                )
            )
        self.vis_ln_post = LayerNorm(vision_width)
        self.vis_projection = nn.Parameter(scale * torch.randn(vision_width, embed_dim))
        self.vis_pred = torch.nn.Linear(embed_dim, 1)

        # Text Model
        self.attn_mask = self.build_attention_mask()
        self.transformer_width = transformer_width
        self.transformer_layers = vision_layers
        self.transformer = torch.nn.ModuleList()
        for _ in range(vision_layers):
            self.transformer.append(
                CrossAttentionBlock(
                    transformer_width, vision_width, transformer_heads
                )
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.text_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.text_ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_pred = torch.nn.Linear(embed_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.text_positional_embedding, std=0.01)

        proj_std = (self.transformer_width ** -0.5) * ((2 * self.transformer_layers) ** -0.5)
        attn_std = self.transformer_width ** -0.5
        fc_std = (2 * self.transformer_width) ** -0.5
        for block in self.transformer:
            if hasattr(block, 'attn'):
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if hasattr(block, 'cross_attn'):
                nn.init.normal_(block.cross_attn.q_proj_weight, std=attn_std)
                nn.init.normal_(block.cross_attn.k_proj_weight, std=attn_std)
                nn.init.normal_(block.cross_attn.v_proj_weight, std=attn_std)
                nn.init.normal_(block.cross_attn.out_proj.weight, std=proj_std)

            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image_text(self, image, text):
        # Prepare image tokenization
        image = self.conv1(image)  # shape = [*, width, grid, grid]
        image = image.reshape(image.shape[0], image.shape[1], -1)  # shape = [*, width, grid ** 2]
        image = image.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        image = torch.cat([self.vis_class_embedding.to(image.dtype) + torch.zeros(image.shape[0], 1, image.shape[-1], dtype=image.dtype, device=image.device), image], dim=1)  # shape = [*, grid ** 2 + 1, width]
        image = image + self.vis_positional_embedding.to(image.dtype)
        image = self.vis_ln_pre(image)

        # Prepare text tokenization
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.text_positional_embedding

        # Push through model
        out1, out2 = image, x
        for i in range(len(self.vis_transformer)):
            out1, out2 = (
                self.vis_transformer[i](out1, out2),
                self.transformer[i](out2, out1)
            )
    
        out1 = self.vis_ln_post(out1[:, 0, :])
        out2 = self.text_ln_final(out2)

        out1 = out1 @ self.vis_projection
        out2 = out2[torch.arange(out2.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return out1, out2


    def forward(self, images, text, labels=None):
        # images and text are pairings with labels --> 1 or 0 for pos/neg pair
        image_embeddings, text_embeddings = self.encode_image_text(images, text)

        image_preds = self.sigmoid(self.vis_pred(image_embeddings))
        text_preds = self.sigmoid(self.text_pred(text_embeddings))

        if labels is not None:
            total_loss = (
                torch.nn.functional.binary_cross_entropy(image_preds, labels) +
                torch.nn.functional.binary_cross_entropy(text_preds, labels)
            ) / 2
        else:
            total_loss = None

        return total_loss, image_preds
