import jax
import jax.numpy as jnp
from flax import linen as nn


class Mlp(nn.Module):
    hidden_features: int = 1536
    out_features: int = 384
    act_layer: nn.Module = nn.gelu
    dropout_rate: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(features=self.hidden_features, use_bias=self.bias, name="fc1")(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop1")(
            x, deterministic=not training
        )
        x = nn.Dense(features=self.out_features, use_bias=self.bias, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop2")(
            x, deterministic=not training
        )
        return x
