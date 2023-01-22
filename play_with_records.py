
import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from fire import Fire

from games.env import Enviroment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


@partial(
    jax.jit,
    static_argnames=("num_simulations", "enable_mcts", "random_action"),
)
def play_one_move(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations: int = 1024,
    random_action: bool = True,
):
    """Play a move using agent's policy"""
    if enable_mcts:
        batched_env: Enviroment = replicate(env, 1)  # type: ignore
        rng_key, rng_key_1 = jax.random.split(rng_key)  # type: ignore
        policy_output = improve_policy_with_mcts(
            agent,
            batched_env,
            rng_key_1,  # type: ignore
            rec_fn=recurrent_fn,
            num_simulations=num_simulations,
        )
        action_weights = policy_output.action_weights[0]
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]
    else:
        action_logits, value = agent(env.canonical_observation())
        action_weights = jax.nn.softmax(action_logits, axis=-1)

    if random_action:
        action = jax.random.categorical(rng_key, jnp.log(action_weights), axis=-1)
    else:
        action = jnp.argmax(action_weights)
    return action, action_weights, value


def agent_vs_agent_with_records(
    agent1,
    agent2,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of agent1 vs agent2, with game history """

    def cond_fn(state):
        env = state[0]
        not_ended = env.is_terminated() == False
        not_too_long = env.count <= env.max_num_steps()
        return jnp.logical_and(not_ended, not_too_long)

    DUMMY_ACTION_REWARD = jnp.array(-1, dtype=int), jnp.array(0.)
    def step_fn(state, x):
        env, a1, a2, rng_key = state
        turn = env.turn
        # checkify.check(env.turn == turn, 'turn should match')
        game_not_over = cond_fn(state)

        rng_key_1, rng_key = jax.random.split(rng_key)
        action, _, _ = play_one_move(
            a1,
            env,
            rng_key_1,
            enable_mcts=enable_mcts,
            num_simulations=num_simulations_per_move,
        )
        env, reward = env_step(env, action)
        signed_reward = turn * reward
        new_state = (env, a2, a1, rng_key)

        result = jax.lax.cond(game_not_over,
                              lambda x: (new_state, (action, signed_reward)),
                              lambda x: (state, DUMMY_ACTION_REWARD),
                              state)
        return result

    state = (
        env,  # reset_env(env),
        agent1,
        agent2,
        rng_key,
        # the following two states are not necessary as env.turn, env.count carry the same info
        # env.turn,
        # jnp.array(1),
    )
    # state = jax.lax.while_loop(cond_fn, loop_fn, state)
    state, move_records = jax.lax.scan(step_fn, state, None, length=env.max_num_steps())
    return move_records


@partial(jax.jit, static_argnums=(4, 5, 6))
def agent_vs_agent_multiple_games_with_records(
    agent1,
    agent2,
    env,
    rng_key,
    enable_mcts: bool = True,
    num_simulations_per_move: int = 1024,
    num_games: int = 128,
):
    """Fast agent vs agent evaluation."""
    _rng_keys = jax.random.split(rng_key, num_games)
    rng_keys = jnp.stack(_rng_keys, axis=0)  # type: ignore
    avsa = partial(
        agent_vs_agent_with_records,
        enable_mcts=enable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )
    batched_avsa = jax.vmap(avsa, in_axes=(None, None, 0, 0))
    envs = replicate(env, num_games)
    results = batched_avsa(agent1, agent2, envs, rng_keys)

    moves, rewards = results
    assert rewards.shape[0] == num_games
    game_results = jnp.sum(rewards, axis=1)

    return game_results, moves


if __name__ == "__main__":
    Fire(main)
