""" experiment w/ jax2tf """
import pickle
import warnings
from functools import partial

import jax
from jax.experimental import jax2tf
from jax import numpy as jnp

import numpy as np
import tensorflow as tf

from utils import import_class


def convert_and_save(agent, fname):
    def f_jax(x):
        return agent(x)

    my_model = tf.Module()
    # Save a function that can take scalar inputs.
    my_model.f = tf.function(jax2tf.convert(f_jax, enable_xla=False),
                             autograph=False,
                             # jit_compile = True,
                             input_signature=(tf.TensorSpec(shape=[5, 5, 9], dtype=tf.int8),))
    tf.saved_model.save(my_model, fname,
                        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True)
                        )


def main(
    game_class="games.go_game.GoBoard5C2",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
    ckpt_filename: str = "../exp-go5C2/colab/go_agent_5-25.ckpt",
):
    """Load agent's weight from disk """
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    convert_and_save(agent, '../exp-go5C2/tfmodel/myconv')


def convert_to_coreml(tfmodel):
    import coremltools as ct
    mlmodel = ct.convert(tfmodel,
                         source='tensorflow',
                         convert_to="mlprogram",
                         compute_precision=ct.precision.FLOAT16,
                         compute_units=ct.ComputeUnit.ALL)


def test_run_tf():
    # Restoring (note: the restored model does *not* require JAX to run, just XLA).
    my_model = tf.saved_model.load('../exp-go5C2/tfmodel/myconv')

    x = tf.ones([5, 5, 9], dtype=tf.int8)
    """ when jax2tf.convert(f_jax, enable_xla=True),
    int8 conv issue? https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md
    Node: 'jax2tf_f_jax_/XlaConvV2'
    UnimplementedError: Could not find compiler for platform METAL: NOT_FOUND: could not find registered compiler for platform METAL -- check target linkage [Op:__inference_restored_function_body_1664]
    """
    result = my_model.f(x)
    print(result)


def test_convert():
    # this works
    main()
