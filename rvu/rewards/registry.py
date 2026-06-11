"""Registry for reward functions."""

from typing import Dict, Type

from .base import Reward

_REGISTRY: Dict[str, Type[Reward]] = {}


def register_reward(name: str):
    """Decorator to register a reward class."""
    def wrapper(cls: Type[Reward]) -> Type[Reward]:
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_reward(name: str, **kwargs) -> Reward:
    """Instantiate a registered reward by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reward '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)
