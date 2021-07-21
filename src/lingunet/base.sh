#!/bin/bash
set -e

python -W ignore run_lingunet.py \
    --evaluate \
    --eval_ckpt /srv/share/mhahn30/Projects/LED/model_runs/checkpoints/softmax_res_connect/softmax_res_connect_unseenAcc0.0548_epoch13.pt
    --name base \
