#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=2 \
BOARD_SIZE=5 \
python3 train_agent.py \
    --game-class="games.go_game.GoBoard5C2" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --random-seed=42 \
    --ckpt-filebase="/content/drive/MyDrive/dlgo/5x5/a0jax/exp-go5C2/go_agent_5" \
    --selfplay-batch-size=64 \
    --training-batch-size=128 \
    --learning-rate=1e-2 \
    --lr-decay-steps=1000000 \
    --num-simulations-per-move=128 \
    --num-simulations-per-move-eval=128 \
    --num-self-plays-per-iteration=1024 \
    --num-eval-games=32 \
    --num-iterations=6

#     --ckpt-filebase="./exp-go5C2/go_agent_5" \