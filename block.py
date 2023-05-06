import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Type
from attention import Attention
from mlp import Mlp


class LayerScale(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            "gamma",
            lambda _, shape: self.initial_value * jnp.ones(shape),
            (x.shape[-1],),
        )
        return x * gamma


class DropPath(nn.Module):
    rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if self.rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.rate
            shape = (x.shape[0], 1, 1, 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng("dropout"), keep_prob, shape=shape
            )
            return x / keep_prob * random_tensor
        else:
            return x


class Block(nn.Module):
    num_heads: int = 6
    embed_dim: int = 384
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    AttentionClass: Type[nn.Module] = Attention
    FfnClass: Type[nn.Module] = Mlp

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        def attn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm1")(x)
            x = self.AttentionClass(
                num_heads=self.num_heads, embed_dim=self.embed_dim, name="attn"
            )(x, training=training)
            x = LayerScale(name="ls1")(x)
            return x

        def ffn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm2")(x)
            x = self.FfnClass(
                hidden_features=int(self.mlp_ratio * self.embed_dim),
                out_features=self.embed_dim,
                name="mlp",
            )(x, training=training)
            x = LayerScale(name="ls2")(x)
            return x

        if training:
            x = x + DropPath(
                rate=self.drop_path_rate, name="drop_path1", deterministic=not training
            )(attn_residual_func(x))
            x = x + DropPath(
                rate=self.drop_path_rate, name="drop_path2", deterministic=not training
            )(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)

        return x


if __name__ == "__main__":
    import functools

    attn_cls = functools.partial(
        Attention,
        num_heads=6,
        attn_bias=True,
        attn_drop_rate=0.0,
        proj_bias=True,
        proj_drop_rate=0.0,
    )
    block_cls = functools.partial(
        Block,
        AttentionClass=attn_cls,
        drop_path_rate=0.0,
    )
    block_def = block_cls()
    block_params = block_def.init(jax.random.PRNGKey(0), jnp.ones((32, 16, 384)))[
        "params"
    ]

    def print_param(path, param):
        print(".".join([p.key for p in path]), param.shape)

    params = jax.tree_util.tree_flatten_with_path(block_params)[0]
    for path, param in params:
        print_param(path, param)
