import jax
import jax.numpy as jnp
import torch
import re
import functools

from vit import DinoViT


def load_vit_params(params_jax: dict, vit_pt: torch.nn.Module):
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(params_jax)
    dinov2_params = {path: param for path, param in vit_pt.named_parameters()}

    no_transpose = {
        "cls_token",
        "pos_embed",
        "mask_token",
    }
    dinov2_params_flat = []
    for path, param in jax_params_flat:
        shape = param.shape
        path = ".".join([p.key for p in path])
        path = re.sub(r"\.scale|.kernel", ".weight", path)
        if path in dinov2_params:
            dinov2_param = dinov2_params[path]
            if path not in no_transpose:
                if len(shape) == 4:
                    dinov2_param = torch.permute(dinov2_param, (2, 3, 1, 0))
                else:
                    dinov2_param = torch.permute(
                        dinov2_param, tuple(reversed(range(len(shape))))
                    )
            if shape != dinov2_param.shape:
                print(path, shape, dinov2_params[path])
            dinov2_params_flat.append(jnp.asarray(dinov2_param.detach().numpy()))
            dinov2_params.pop(path)
        else:
            print(path, shape, None)
            dinov2_params_flat.append(None)
    for path, param in dinov2_params.items():
        print(path, None, param.shape)

    return jax.tree_util.tree_unflatten(jax_param_pytree, dinov2_params_flat)


def load_dino_vits():
    num_heads = 6
    embed_dim = 384
    mlp_ratio = 4

    vit_cls = functools.partial(
        DinoViT,
        num_heads=num_heads,
        embed_dim=embed_dim,
        mlp_ratio=mlp_ratio,
        depth=12,
        img_size=518,
    )
    vit_def = vit_cls()
    vit_params = vit_def.init(jax.random.PRNGKey(0), jnp.ones((1, 518, 518, 3)))[
        "params"
    ]

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    params = load_vit_params(vit_params, dinov2_vits14)

    return (vit_def, params)


def test_dino_vits():
    import numpy as onp

    image = jax.random.uniform(jax.random.PRNGKey(0), (1, 518, 518, 3))
    jax_vit_def, jax_params = load_dino_vits()

    # JAX: forward pass
    image = jax.random.uniform(jax.random.PRNGKey(0), (1, 518, 518, 3))
    embed_jax = jax_vit_def.apply({"params": jax_params}, image, training=False)
    embed_jax = onp.asarray(embed_jax["x_norm_patchtokens"])

    # Torch: forward pass
    image_torch = torch.from_numpy(onp.asarray(image.transpose((0, 3, 1, 2)))).cuda()
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda()
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_vits14.eval()
    embed_torch = (
        dinov2_vits14.forward_features(image_torch)["x_norm_patchtokens"]
        .detach()
        .cpu()
        .numpy()
    )
    embed_torch2 = (
        dinov2_vits14.forward_features(torch.rand((1, 3, 518, 518), device="cuda"))[
            "x_norm_patchtokens"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    cosine_distance = (
        onp.sum(embed_torch * embed_jax)
        / onp.linalg.norm(embed_torch)
        / onp.linalg.norm(embed_jax)
    )
    cosine_distance2 = (
        onp.sum(embed_torch2 * embed_jax)
        / onp.linalg.norm(embed_torch2)
        / onp.linalg.norm(embed_jax)
    )

    # Cosine distance for the first pair (same image) should be close to 1
    assert cosine_distance > 0.999, cosine_distance
    # Cosine distance for the second pair (different images) should be further away.
    # It might still be close to 1, because random noise is semantically similar.
    assert cosine_distance2 < 0.95, cosine_distance2
