import jax
import jax.numpy as jnp
from flax import linen as nn


class Attention(nn.Module):
    num_heads: int = 8
    attn_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_bias: bool = True
    proj_drop_rate: float = 0.0
    embed_dim: int = 384

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, N, C = x.shape
        assert (
            C == self.embed_dim
        ), f"Input embedding dimension ({C}) should match layer embedding dimension ({self.embed_dim})."
        qkv = nn.Dense(features=3 * C, use_bias=self.attn_bias, name="qkv")(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)

        # Attention matrix: (B, H, N, N)
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(C // self.num_heads)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop_rate, name="attn_drop")(
            attn, deterministic=not training
        )

        # Output: (B, N, H, C // H)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        x = nn.Dense(features=C, use_bias=self.proj_bias, name="proj")(x)
        x = nn.Dropout(rate=self.proj_drop_rate, name="proj_drop")(
            x, deterministic=not training
        )

        return x
