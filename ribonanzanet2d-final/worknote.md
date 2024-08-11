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
