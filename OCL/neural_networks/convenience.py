from typing import Callable, Optional, Union

from torch import nn

from .wrappers import Residual


def get_activation_fn(name: Union[str, Callable], inplace: bool = True):
    if callable(name):
        return name
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation function {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: list[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
):
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    module = nn.Sequential(*layers)
    return Residual(module) if residual else module


def build_two_layer_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    activation_fn: str = "relu",
    initial_layer_norm: bool = False,
    residual: bool = False,
):
    return build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        features=[hidden_dim],
        activation_fn=activation_fn,
        initial_layer_norm=initial_layer_norm,
        residual=residual,
    )
