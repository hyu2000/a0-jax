"""Useful functions."""

import importlib
import os.path
from functools import partial
from typing import Tuple, TypeVar, Optional

import chex
import jax
import jax.numpy as jnp
import pax

from games.env import Enviroment as E

T = TypeVar("T")


@pax.pure
def batched_policy(agent, states):
    """Apply a policy to a batch of states.

    Also return the updated agent.
    """
    return agent, agent(states, batched=True)


def replicate(value: T, repeat: int) -> T:
    """Replicate along the first axis."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


@jax.jit
def env_step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    """Execute one step in the enviroment."""
    env, reward = env.step(action)
    return env, reward


def import_class(path: str) -> E:
    """Import a class from a python file.

    For example:
    >> Game = import_class("connect_two_game.Connect2Game")

    Game is the Connect2Game class from `connection_two_game.py`.
    """
    names = path.split(".")
    mod_path, class_name = names[:-1], names[-1]
    mod = importlib.import_module(".".join(mod_path))
    return getattr(mod, class_name)


def select_tree(pred: jnp.ndarray, a, b):
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_util.tree_map(partial(jax.lax.select, pred), a, b)


def find_latest_ckpt(path_base: str) -> Optional[str]:
    """ look for {path_base}-{iter}.ckpt for the largest iter
    path_base can contain dir name
    """
    import glob
    dir = os.path.dirname(path_base)
    if not os.path.isdir(dir):
        print(f'creating {dir}')
        os.makedirs(dir, exist_ok=True)
        return None
    fnames = glob.glob(f'{path_base}-*.ckpt')
    if len(fnames) == 0:
        return None
    idx_start = len(path_base) + 1
    idx_gens = [int(fname[idx_start:-5]) for fname in fnames]
    idx_latest = sorted(idx_gens)[-1]
    return f'{path_base}-{idx_latest}.ckpt'
