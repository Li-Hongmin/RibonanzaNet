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
