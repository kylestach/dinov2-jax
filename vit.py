import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as onp

from typing import Type
from functools import partial

from mlp import Mlp
from attention import Attention
from block import Block
from patch_embed import PatchEmbed


class DinoViT(nn.Module):
    img_size: int = 224
    in_channels: int = 3

    patch_size: int = 14
    embed_dim: int = 384

    depth: int = 12

    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    BlockClass: Type[nn.Module] = Block
    AttentionClass: Type[nn.Module] = Attention
    FfnClass: Type[nn.Module] = Mlp
    EmbedLayer: Type[nn.Module] = PatchEmbed

    def _interpolate_pos_encoding(
        self, x: jnp.ndarray, w: int, h: int, pos_embed: jnp.ndarray
    ):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim),
            (1, w0, h0, dim),
            method="bicubic",
        )
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed), axis=1).astype(
            previous_dtype
        )

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, H, W, C = x.shape
        assert H == W == self.img_size, "x size must be (B, {}, {}, {})".format(
            self.img_size, self.img_size, C
        )

        x = self.EmbedLayer(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )(x)
        cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], *cls_token.shape[1:]))
        x = jnp.concatenate((cls_token, x), axis=1)

        num_patches = (self.img_size // self.patch_size) ** 2
        num_tokens = 1

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + num_tokens, self.embed_dim),
        )
        x = x + self._interpolate_pos_encoding(
            x, self.img_size, self.img_size, pos_embed
        )

        for i in range(self.depth):
            x = self.BlockClass(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path_rate=self.drop_path_rate,
                AttentionClass=self.AttentionClass,
                FfnClass=self.FfnClass,
                name=f"blocks.{i}",
            )(x, training=training)

        x_norm = nn.LayerNorm(name="norm")(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }
