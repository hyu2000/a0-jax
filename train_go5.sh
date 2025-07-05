#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=2 \
BOARD_SIZE=5 \
python3 train_agent.py \
    --game-class="games.go_game.GoBoard5C2" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --random-seed=42 \
    --ckpt-filebase="/content/drive/MyDrive/dlgo/5x5/a0jax/exp-go5C2/go_agent_5" \
    --selfplay-batch-size=64 \
    --training-batch-size=1024 \
    --learning-rate=1e-2 \
    --lr-decay-steps=4096 \
    --num-simulations-per-move=32 \
    --num-simulations-per-move-eval=32 \
    --num-self-plays-per-iteration=256 \
    --num-eval-games=32 \
    --num-iterations=180

#     --ckpt-filebase="./exp-go5C2/go_agent_5" \
#    --training-batch-size=1024 \  # this goes with LR. 1024 is about 32 games
#    --lr-decay-steps=4096 \  # 60 gen * 256/32 * 8 (symmetries) = 480 * 8
#    --num-self-plays-per-iteration=256 \  # num games per generation
