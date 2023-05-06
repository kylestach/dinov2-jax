import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Type


class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 14
    in_channels: int = 3
    embed_dim: int = 384
    norm_layer: Type[nn.Module] = None
    flatten_embedding: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, H, W, C = x.shape
        patch_H, patch_W = self.patch_size, self.patch_size
        assert (
            H % patch_H == 0 and W % patch_W == 0
        ), f"Image size ({H}*{W}) cannot be evenly divided by patch size ({patch_H}*{patch_W})."

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(patch_H, patch_W),
            strides=(patch_H, patch_W),
            name="proj",
            padding="VALID",
        )(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm_layer is not None:
            x = self.norm_layer(name="norm")(x)

        if not self.flatten_embedding:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x
