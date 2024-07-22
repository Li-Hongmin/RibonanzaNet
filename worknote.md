# 工作笔记

目前应该把那个数据整理一下。

我要引入新的技术
mamba还有pair weight averaging
之类的，
还有就是多任务学习
来提升模型的性能。

最后就是要把这个模型finetune一个更好的水解预测。
想想应该是可以的。

## 数据整理

数据存放
/work/gs58/d58004/datasets/stanford-ribonanza-rna-folding
eterna_openknot_metadata  rmdb_data.csv                       supplementary_silico_predictions
OLD                       sample_submission.csv               test_sequences.csv
rhofold_pdbs              sequence_libraries                  train_data.csv
Ribonanza_bpp_files       stanford-ribonanza-rna-folding.zip  train_data_QUICK_START.csv

###
数据量
(base) [d58004@wisteria08 stanford-ribonanza-rna-folding]$ wc -l train_data_QUICK_START.csv
335617 train_data_QUICK_START.csv
(base) [d58004@wisteria08 stanford-ribonanza-rna-folding]$ wc -l train_data.csv 
1643681 train_data.csv

#!/bin/sh
  
#------ pjsub option --------# 
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=4:00:00 
#PJM -g gs58
#PJM -j


#------- Program execution -------#
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
nvidia-smi
cd /work/gs58/d58004/ideas/RibonanzaNet
accelerate launch --mixed_precision=fp16 --dynamo_backend=no inference.py --config_path configs/pairwise.yaml

~                                                                   