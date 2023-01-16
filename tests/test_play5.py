import pickle
import random
import warnings

import jax
from jax import numpy as jnp

import coords
from play import agent_vs_agent_multiple_games_with_records
from train_agent import format_game_record_gtp
from utils import import_class, reset_env
import go


def setupGo5():
    assert go.N == 5
    warnings.filterwarnings("ignore")
    env = import_class('games.go_game.GoBoard5C2')()
    agent = import_class('policies.resnet_policy.ResnetPolicyValueNet128')(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    # ckpt_filename = "/Users/hyu/PycharmProjects/a0-jax/exp-5x5/go_agent_5-5.ckpt"
    ckpt_filename = "/Users/hyu/PycharmProjects/a0-jax/exp-go5C2/go_agent_5-5.ckpt"
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    # rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    rng_key = jax.random.PRNGKey(1234)
    return env, agent, rng_key


def test_board():
    env = import_class('games.go_game.GoBoard5x5')()
    assert env.count == 0 and env.turn == 1
    env.render()
    env, _ = env.step(jnp.array(12, dtype=int))
    assert env.count == 1 and env.turn == -1
    env.render()
    env, _ = env.step(jnp.array(13, dtype=int))
    assert env.count == 2 and env.turn == 1
    env.render()


def test_BoardC2():
    env = import_class('games.go_game.GoBoard5C2')()
    assert env.count == 1 and env.turn == -1
    env.render()
    env, _ = env.step(jnp.array(12, dtype=int))
    assert env.count == 2 and env.turn == 1
    env.render()

    env = reset_env(env)
    assert env.count == 1 and env.turn == -1


def test_illegal_move():
    env = import_class('games.go_game.GoBoard5C2')()
    assert env.count == 1 and env.turn == -1
    env.render()

    # B+R
    gtp_moves = 'C3 D3 B3 C4 B2 D1 D4 D5 E4 E3 B4 E5 B5 B1 D4 E4 C5 A2 D4 D2 A3 C4 A1 A5 C1 E1 B1 pass D4'  # A4
    for gtp_move in gtp_moves.split():
        idx_move = coords.to_flat(coords.from_gtp(gtp_move))
        env, reward = env.step(jnp.array(idx_move, dtype=int))
        assert reward == 0
        assert env.done == 0

    env.render()
    assert env.turn == 1
    env, reward = env.step(jnp.array(coords.to_flat(coords.from_gtp('A4')), dtype=int))
    assert reward < 0
    assert env.done


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
