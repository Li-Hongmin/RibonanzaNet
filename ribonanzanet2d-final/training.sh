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
# echo "Step1...................."
# python step1.py
echo “Step2....................”
python step2.py
echo “Step3....................”
python step3.py
echo "make_submission...................."
python make_submission.py
