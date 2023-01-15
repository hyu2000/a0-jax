import pickle
import random
import warnings

import jax
from jax import numpy as jnp

from play import agent_vs_agent_multiple_games_with_records
from train_agent import format_game_record_gtp
from utils import import_class
import go


def setupGo5():
    assert go.N == 5
    warnings.filterwarnings("ignore")
    env = import_class('games.go_game.GoBoard5x5')()
    agent = import_class('policies.resnet_policy.ResnetPolicyValueNet128')(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    ckpt_filename = "/Users/hyu/PycharmProjects/a0-jax/go_agent_5-3.ckpt"
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    return env, agent, rng_key


def test_board():
    env, agent, rng_key = setupGo5()
    assert env.count == 0 and env.turn == 1
    env.render()
    env, _ = env.step(jnp.array(12, dtype=int))
    assert env.count == 1 and env.turn == -1
    env.render()
    env, _ = env.step(jnp.array(13, dtype=int))
    assert env.count == 2 and env.turn == 1
    env.render()


def test_avsa_multi_games():
    """ """
    env, agent, rng_key = setupGo5()
    num_games = 2

    results = agent_vs_agent_multiple_games_with_records(
        agent, agent, env, rng_key,
        num_simulations_per_move=2,
        num_games=num_games)

    game_results, game_records = results
    assert len(game_results) == num_games and len(game_records) == num_games
    gtp_moves = [format_game_record_gtp(result, x) for result, x in zip(game_results, game_records)]
    print('\n'.join(gtp_moves))
    print(game_results, [len(s.split()) for s in gtp_moves])
