import datetime
import os
from utils import find_latest_ckpt


def test_find_latest_ckpt():
    latest_ckpt = find_latest_ckpt('../go_agent_9x9_128_sym')
    assert latest_ckpt == '../go_agent_9x9_128_sym-218.ckpt'


def test_create_ckpt_dir():
    # assert not os.path.exists('../exp-go5C2')
    latest_ckpt = find_latest_ckpt('../exp-go5C2/go_agent_5')
    assert latest_ckpt is None
    assert os.path.exists('../exp-go5C2')


def test_now():
    now = datetime.datetime.now()
    s = now.strftime('%H:%M:%S.%f')
    print(now, s)
