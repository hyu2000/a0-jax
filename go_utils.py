import chex
import coords


def format_game_record_gtp(result: int, moves: chex.Array) -> str:
    gtp_moves = [coords.flat_to_gtp(x) for x in moves if x >= 0]
    game_len = len(gtp_moves)
    assert all(moves[game_len:] == -1)
    result_str = 'B+R' if result > 0 else 'W+R' if result < 0 else 'B+T'
    return f'{result_str} %s' % ' '.join(gtp_moves)
