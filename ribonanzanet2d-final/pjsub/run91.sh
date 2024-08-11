#!/bin/sh

#------ pjsub option --------# 
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00 
#PJM -g gs58
#PJM -j


#------- Program execution -------#
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
nvidia-smi
cd /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final
echo "Start training"
python train_ribonanzanet.py --save_dir saved_models_mamaba --epochs 100 --batch_size 32 --lr 0.0001 --max_seq_length 91