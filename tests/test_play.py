import pickle
import random
import warnings
from functools import partial

import chex
import jax
from jax import numpy as jnp

from play import agent_vs_agent_with_records
from utils import import_class, replicate
from . import go, coords


def setupGo9():
    assert go.N == 9
    warnings.filterwarnings("ignore")
    env = import_class('games.go_game.GoBoard9x9')()
    agent = import_class('policies.resnet_policy.ResnetPolicyValueNet128')(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    ckpt_filename = "/Users/hyu/PycharmProjects/a0-jax/go_agent_9x9_128_sym.ckpt"
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    return env, agent, rng_key


def test_avsa():
    env, agent, rng_key = setupGo9()
    env, _ = env.step(jnp.array(30, dtype=int))  # D6

    result = agent_vs_agent_with_records(
        agent, agent,
        env,
        rng_key,
        enable_mcts=True,
        num_simulations_per_move=4
    )

    moves, rewards = result
    # move_records = zip(moves.tolist(), rewards.tolist())
    assert jnp.sum(rewards != 0) == 1
    idx_last = jnp.argwhere(rewards != 0)[0][0]
    final_reward = rewards[idx_last]
    moves = moves[:idx_last + 1]
    assert all(moves[(idx_last + 1):] == -1)
    gtp_moves = [coords.flat_to_gtp(x) for x in moves]
    print(f'Total {len(moves)} moves, last:', gtp_moves[-5:])
    print(final_reward)
    print(' '.join(gtp_moves))


def test_tmp():
    assert coords.flat_to_gtp(-1) == 'J10'


def convert_game_record_to_gtp(moves: chex.Array):
    gtp_moves = [coords.flat_to_gtp(x) for x in moves if x >= 0]
    game_len = len(gtp_moves)
    assert all(moves[game_len:] == -1)
    return ' '.join(gtp_moves)


def test_avsa_multi_games():
    """ vmap/jit, same as agent_vs_agent_multiple_games() """
    env, agent, rng_key = setupGo9()
    num_games = 4

    _rng_keys = jax.random.split(rng_key, num_games)
    rng_keys = jnp.stack(_rng_keys, axis=0)  # type: ignore
    avsa = partial(
        agent_vs_agent_with_records,
        enable_mcts=True,
        num_simulations_per_move=4
    )
    batched_avsa = jax.vmap(avsa, in_axes=(None, None, 0, 0))
    envs = replicate(env, num_games)
    results = batched_avsa(agent, agent, envs, rng_keys)

    moves, rewards = results
    assert rewards.shape[0] == num_games
    game_results = jnp.sum(rewards, axis=1)
    gtp_moves = [convert_game_record_to_gtp(x) for x in moves]
    print('\n'.join(gtp_moves))
    print(game_results)
