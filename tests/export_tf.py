""" experiment w/ jax2tf """
import pickle
import warnings
from functools import partial
from typing import Tuple

import jax
from jax.experimental import jax2tf
from jax import numpy as jnp

import numpy as np
import tensorflow as tf

from utils import import_class


def convert_and_save(agent, input_dims: Tuple, fname):
    def f_jax(x):
        return agent(x)

    def f_jax_batched(x):
        return agent(x, batched=True)

    my_model = tf.Module()
    # Save a function that can take scalar inputs.
    my_model.f = tf.function(
        jax2tf.convert(f_jax, enable_xla=False),
        autograph=False,
        # jit_compile=True,
        input_signature=(tf.TensorSpec(shape=input_dims, dtype=tf.int8),))
    batch_shape = (None, ) + input_dims
    polymorphic_shapes = str(batch_shape).replace('None', 'b')  # "(b, 5, 5, 9)"
    my_model.f_batched = tf.function(
        jax2tf.convert(f_jax_batched, enable_xla=False,
                       polymorphic_shapes=[polymorphic_shapes]),
        autograph=False,
        input_signature=(tf.TensorSpec(shape=batch_shape, dtype=tf.int8),))
    tf.saved_model.save(my_model, fname,
                        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))


def main(
    game_class="games.go_game.GoBoard5C2",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
    ckpt_filename: str = "../exp-go5C2/colab/go_agent_5-25.ckpt",
    tf_model_path: str = '../exp-go5C2/tfmodel/model5-25'
):
    """Load agent's weight from disk """
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    input_dims = env.observation().shape
    agent = import_class(agent_class)(
        input_dims=input_dims,
        num_actions=env.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    convert_and_save(agent, input_dims, tf_model_path)


def convert_to_coreml(tfmodel):
    import coremltools as ct
    mlmodel = ct.convert(tfmodel,
                         source='tensorflow',
                         convert_to="mlprogram",
                         compute_precision=ct.precision.FLOAT16,
                         compute_units=ct.ComputeUnit.ALL)
    return mlmodel


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


def test_convert_to_coreml():
    tfmodel = tf.saved_model.load('../exp-go5C2/tfmodel/myconv')
    """
    NotImplementedError: Expected model format: [SavedModel | [concrete_function] | tf.keras.Model | .h5 | GraphDef], got <tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject object at 0x16e49c160>
    """
    mlmodel = convert_to_coreml(tfmodel)
    # mlmodel.save('')


def test_convert5():
    # this works
    main()


def test_convert9():
    main(
        game_class="games.go_game.GoBoard9x9",
        ckpt_filename="../go_agent_9x9_128_sym.ckpt",
        tf_model_path="../exp-go9/tfmodel/model-218"
    )
