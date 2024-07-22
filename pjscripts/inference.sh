#!/bin/sh
  
#------ pjsub option --------# 
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=4:00:00 
#PJM -g gs58
#PJM -j


#------- Program execution -------#
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
nvidia-smi
cd /work/gs58/d58004/ideas/RibonanzaNet
accelerate launch --num_processes=8 --mixed_precision=fp16 --dynamo_backend=no inference.py --config_path configs/pairwise.yaml

