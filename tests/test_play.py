import pickle
import random
import warnings

import jax
from jax import numpy as jnp

from play import agent_vs_agent_with_records
from utils import import_class
import coords


def test_avsa():
    warnings.filterwarnings("ignore")
    env = import_class('games.go_game.GoBoard9x9')()
    agent = import_class('policies.resnet_policy.ResnetPolicyValueNet128')(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    ckpt_filename = "../go_agent_9x9_128_sym.ckpt"
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    result = agent_vs_agent_with_records(
        agent, agent,
        env,
        rng_key,
        enable_mcts=True,
        num_simulations_per_move=2
    )
    moves, rewards = result
    # move_records = zip(moves.tolist(), rewards.tolist())
    assert jnp.sum(rewards != 0) == 1
    idx_last = jnp.argwhere(rewards != 0)[0][0]
    final_reward = rewards[idx_last]
    moves = moves[:idx_last + 1]
    assert all(moves[(idx_last + 1):] == -1)
    gtp_moves = [coords.flat_to_gtp(x) for x in moves]
    print()
    print(' '.join(gtp_moves), final_reward)
