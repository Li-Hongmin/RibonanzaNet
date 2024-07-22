# 工作笔记

                                                      

export PATH="/work/02/gs58/d58004/mambaforge/envs/torch/bin/:$PATH"
python make_submission.py --para RibonanzaNet-Deg_30.pt
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_30.pt.csv -m "RibonanzaNet-Deg_30.pt"

python make_submission.py --para RibonanzaNet-Deg_20.pt
python make_submission.py --para RibonanzaNet-Deg_21.pt

kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_20.pt.csv -m "RibonanzaNet-Deg_20.pt"


kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_20.pt.csv -m "RibonanzaNet-Deg_20.pt"
kaggle competitions submit -c stanford-covid-vaccine -f submission_RibonanzaNet-Deg_21.pt.csv -m "RibonanzaNet-Deg_21.pt"


```python

