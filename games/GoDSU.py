import pax
import jax.numpy as jnp


class GoDSU(pax.Module):
    """
    on top of DSU, we track the total #liberties of chains
    """
    def __init__(self):
        super().__init__()

        self.board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int8)
        self.liberty = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int8)

    def take_action(self, action, turn):
        """
        if any dead enemy chains, remove them: this triggers a big change in liberty situation -> re-init
        otherwise, no capture, update liberties of neighboring chains:
          - for neighboring enemy chains (distinctive), reduce by 1;
          - our own chains are now joined as one: sum(liberties) + #(empty spaces around action) - #(own chains)
        """

    def check_suicide(self, action, turn):
        """
        - if loc has adjacent empty space, it's fine
        - now all 4 neighbors occupied:
            if any enemy chain has only 1 liberty, that chain is killed --> ok
            if any of our own chains has just 1 liberty --> suicide
            elif all enemy chains --> suicide
            otherwise, it joins our own chain(s), which has at least one liberty --> ok
        """