#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=2 \
python3 train_agent.py \
    --game-class="games.go_game.GoBoard5x5" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --random-seed=42 \
    --ckpt-filebase="./go_agent_5" \
    --selfplay-batch-size=64 \
    --training-batch-size=128 \
    --num-simulations-per-move=32 \
    --learning-rate=1e-2 \
    --lr-decay-steps=1000000 \
    --num-eval-games=16 \
    --num-simulations-per-move-eval=32 \
    --num-self-plays-per-iteration=128 \
    --num-iterations=10
