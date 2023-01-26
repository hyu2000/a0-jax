#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=2 \
python3 train_agent.py \
    --game-class="games.go_game.GoBoard9x9" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --random-seed=42 \
    --ckpt-filebase="/content/drive/MyDrive/dlgo/9x9/go_agent_9" \
    --selfplay-batch-size=64 \
    --training-batch-size=128 \
    --learning-rate=1e-2 \
    --lr-decay-steps=1000000 \
    --num-simulations-per-move=128 \
    --num-simulations-per-move-eval=128 \
    --num-self-plays-per-iteration=512 \
    --num-eval-games=32 \
    --num-iterations=1
