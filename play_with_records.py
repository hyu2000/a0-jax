
import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from fire import Fire

from games.env import Enviroment
from go_utils import format_game_record_gtp
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env
import mylogging
import logging


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


@chex.dataclass(frozen=True)
class MoveOutput:
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array


def agent_vs_agent_with_records(
    agent1,
    agent2,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of agent1 vs agent2, with game history """
    def step_fn(state, x):
        env, a1, a2, rng_key = state
        turn = env.turn

        rng_key_1, rng_key = jax.random.split(rng_key)
        action, _, _ = play_one_move(
            a1,
            env,
            rng_key_1,
            enable_mcts=enable_mcts,
            num_simulations=num_simulations_per_move,
        )
        terminated = env.is_terminated()
        env, reward = env_step(env, action)
        signed_reward = turn * reward
        new_state = (env, a2, a1, rng_key)

        return new_state, MoveOutput(
            action=action,
            reward=signed_reward,
            terminated=jnp.logical_or(terminated, env.is_terminated())
        )

    state = (
        env,  # reset_env(env),
        agent1,
        agent2,
        rng_key,
    )
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

    return results


def main(
    game_class="games.go_game.GoBoard5C2",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
    ckpt_filename: str = "./exp-go5C2/colab/go_agent_5-25.ckpt",
    enable_mcts: bool = True,
    num_simulations_per_move: int = 128,
    num_games: int = 64,
):
    """Load agent's weight from disk and start the game."""
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    rng_key = jax.random.PRNGKey(1234)
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    logging.info(f'Starting {num_games} eval games')
    game_records = agent_vs_agent_multiple_games_with_records(
        agent,
        agent,
        env,
        rng_key,
        enable_mcts=enable_mcts,
        num_simulations_per_move=num_simulations_per_move,
        num_games=num_games
    )
    logging.info(f'Done {num_games} eval games')

    terminated = game_records.terminated.astype(int)
    # np.argmax picks the 1st in case of tie
    game_ilast = jnp.argmax(terminated == 1, axis=1)
    game_results = game_records.reward[jnp.arange(num_games), game_ilast]
    for i, (idx_last_move, result, actions) in enumerate(zip(game_ilast, game_results, game_records.action)):
        game_len = idx_last_move + 1
        gtp_moves = format_game_record_gtp(result, actions[:game_len])
        print(f'game {i}: {game_len} {gtp_moves}')
    win_count = jnp.sum(game_results == 1)
    loss_count = jnp.sum(game_results == -1)
    logging.info(f"  evaluation      {win_count} win - {loss_count} loss")


if __name__ == "__main__":
    print("Cores:", jax.local_devices(), 'num_devices=', jax.local_device_count())
    Fire(main)
