# 工作笔记

                                                      

export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
python make_submission.py --para RibonanzaNet-Deg_30.pt
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_30.pt.csv -m "RibonanzaNet-Deg_30.pt"

python make_submission.py --para RibonanzaNet-Deg_20.pt
python make_submission.py --para RibonanzaNet-Deg_21.pt

python make_submission.py --para RibonanzaNet-Deg_31_retrain.pt

kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_31_retrain.pt.csv -m "RibonanzaNet-Deg_31_retrain.pt"


kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_20.pt.csv -m "RibonanzaNet-Deg_20.pt"
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_21.pt.csv -m "RibonanzaNet-Deg_21.pt"


当前，尝试了全部的伪标签，能打尽量打的，效果不佳，那是step3.
后来，限制了到68位，那是step3_68.  Deg_30_68
再后来，伪标签使用了新的模型，彻底变成了semi-supervised step3_68_pseudo_recompute.  Deg_31_68_re

python make_submission.py --para RibonanzaNet-Deg_30_68.pt
python make_submission.py --para RibonanzaNet-Deg_31_68.pt


kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_30_68.pt.csv -m "RibonanzaNet-Deg_30_68.pt"
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_31_68.pt.csv -m "RibonanzaNet-Deg_31_68.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_30_68_re.pt.csv -m "RibonanzaNet-Deg_30_68_re.pt"
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_31_68_re.pt.csv -m "RibonanzaNet-Deg_31_68_re.pt"

## 2024-08-10
这个是需要在交互环境下运行的。
今天要把上次finetune的模型，评测一下。

pjsub --interact -g gs58 -L rscgrp=interactive-a,node=1

export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export TRITON_CACHE_DIR="/work/gs58/d58004/tmp/triton"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.001-epochs20-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-0-freezed-FinetuneDeg-epoch20.pt

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/highSN_lr0.001-epochs20-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-2-annealed-FinetuneDeg-epoch11.pt

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.001-epochs20-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-0-freezed-FinetuneDeg-epoch20.pt.csv -m "pseudo_lr0.001-epochs20-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-0-freezed-FinetuneDeg-epoch20.pt"
## 2024-08-11 增加了epoch 100的模型

pseudo_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-1-unfreezed-FinetuneDeg-epoch024.pt

highSN_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-2-annealed-FinetuneDeg-epoch013.pt

```bash
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export TRITON_CACHE_DIR="/work/gs58/d58004/tmp/triton"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
export TRANSFORMERS_CACHE="/work/gs58/d58004/tmp/transformers"

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-1-unfreezed-FinetuneDeg-epoch024.pt

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/highSN_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-2-annealed-FinetuneDeg-epoch013.pt

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-1-unfreezed-FinetuneDeg-epoch024.pt.csv -m "pseudo_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-1-unfreezed-FinetuneDeg-epoch024.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-2-annealed-FinetuneDeg-epoch013.pt.csv -m "highSN_lr0.0001-epochs100-wd0.001-max_seq_length130-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-2-annealed-FinetuneDeg-epoch013.pt"

```
## 2024-08-12
今天尝试不同长度的 

-rw-r----- 1 d58004 gs58 61168846 Aug 12 14:12 highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 61168846 Aug 12 14:10 highSN_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 45578578 Aug 12 01:06 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 45579738 Aug 12 05:01 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 61168194 Aug 12 03:54 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 61169498 Aug 12 06:48 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 61168194 Aug 12 04:45 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 61169498 Aug 12 06:07 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt

使用上面的模型预测，提交。
```bash
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export TRITON_CACHE_DIR="/work/gs58/d58004/tmp/triton"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"

echo "model 1"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt 

echo "model 2"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/highSN_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt


echo "model 5"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt

echo "model 6"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt

echo "model 7"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt

echo "model 8"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt

```

-rw-r----- 1 d58004 gs58 32225584 Aug 12 15:54 submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt.csv
-rw-r----- 1 d58004 gs58 32356171 Aug 12 15:57 submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt.csv
-rw-r----- 1 d58004 gs58 32230178 Aug 12 16:11 submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt.csv
-rw-r----- 1 d58004 gs58 32241501 Aug 12 16:14 submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt.csv
-rw-r----- 1 d58004 gs58 32371729 Aug 12 16:20 submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt.csv
-rw-r----- 1 d58004 gs58 32333284 Aug 12 16:24 submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt.csv

把上面的提交到kaggle

```bash

kaggle competitions submit -c stanford-covid-vaccine -f submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt.csv -m "submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt.csv -m "submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-2-annealed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-0-freezed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length91-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaTrue-use_mamba_endFalse-1-unfreezed-FinetuneDeg-epoch.pt"

```

当前存在的问题是

```bash
export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export TRITON_CACHE_DIR="/work/gs58/d58004/tmp/triton"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
echo "model 3"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt --config_path configs/pairwise_no_mamba.yaml

echo "model 4"
python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt --config_path configs/pairwise_no_mamba.yaml
```

mamba is used at the end
Traceback (most recent call last):
  File "/work/02/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/make_submission.py", line 116, in <module>
    model.load_state_dict(torch.load(args.para,map_location=device))
  File "/work/02/gs58/d58004/mambaforge/envs/torch/lib/python3.9/site-packages/torch/nn/modules/module.py", line 2152, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for finetuned_RibonanzaNet:
        Missing key(s) in state_dict: "mamba_end.dt_bias", "mamba_end.A_log", "mamba_end.D", "mamba_end.in_proj.weight", "mamba_end.conv1d.weight", "mamba_end.conv1d.bias", "mamba_end.norm.weight", "mamba_end.out_proj.weight". 

结果发现这个没有发挥作用，因为我的参数没有传递过去。
use_mamba_end 还是 False的

## 2024-08-13

今天完成了几个模型的训练，提交。

-rw-r----- 1 d58004 gs58 47444750 Aug 13 19:14 highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-2-annealed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 47444162 Aug 13 08:40 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 47445338 Aug 13 16:37 pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt

-rw-r----- 1 d58004 gs58 53422702 Aug 13 11:58 use_hybridTrue-lr0.001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-pseudo_FinetuneDeg-epoch.pt
-rw-r----- 1 d58004 gs58 53424024 Aug 13 19:53 use_hybridTrue-lr0.001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-pseudo_FinetuneDeg-epoch.pt

```bash

export MPLCONFIGDIR="/work/gs58/d58004/tmp/matplotlib"
export WANDB_CONFIG_DIR="/work/gs58/d58004/tmp/wandb"
export TRITON_CACHE_DIR="/work/gs58/d58004/tmp/triton"
export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"

echo "model 9"

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-2-annealed-FinetuneDeg-epoch.pt --config_path configs/pairwise_no_mamba.yaml

echo "model 10"

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt --config_path configs/pairwise_no_mamba.yaml

echo "model 11"

python make_submission.py --para /work/gs58/d58004/ideas/RibonanzaNet/ribonanzanet2d-final/saved_models_mamaba/pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt --config_path configs/pairwise_no_mamba.yaml
```

提交到kaggle

```bash
kaggle competitions submit -c stanford-covid-vaccine -f submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-2-annealed-FinetuneDeg-epoch.pt.csv -m "submission_highSN_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-2-annealed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-0-freezed-FinetuneDeg-epoch.pt"

kaggle competitions submit -c stanford-covid-vaccine -f submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt.csv -m "submission_pseudo_lr0.0001-epochs100-wd0.001-max_seq_length68-sn_threshold5.0-noisy_threshold1.0-batch_size32-use_mambaFalse-use_mamba_endTrue-1-unfreezed-FinetuneDeg-epoch.pt"

```

## 2024-08-16
我发现还是得按原来的节奏，也就是先得用2阶段的训练，然后再用3阶段的训练。
也就是说，先用2阶段的比较好的数据+噪声数据的伪标签，然后再用3阶段的加上测试数据的伪标签。
