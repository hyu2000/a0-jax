from utils import find_latest_ckpt


def test_find_latest_ckpt():
    latest_ckpt = find_latest_ckpt('../go_agent_9x9_128_sym')
    assert latest_ckpt == '../go_agent_9x9_128_sym-218.ckpt'
