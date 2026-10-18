# OVDL

## 安装

```bash
pip install -r requirements.txt
mkdir -p logs
```

## CinC2017 主实验

```bash
nohup bash run_cinc2017_lowmem.sh ./data/CinC2017 cuda:0 \
  --epochs 100 --seeds 42 43 44 45 46 \
  --out-root runs/cinc2017_postnorm_main_5seeds \
  > logs/cinc2017_main_5seeds.log 2>&1 &
```

## CPSC2018 主实验

```bash
nohup bash run_cpsc2018_lowmem.sh ./data/Training_WFDB cuda:0 \
  --labels ./data/Training_WFDB/labels.csv \
  --reference-csv ./data/REFERENCE.csv \
  --epochs 100 --seeds 42 43 44 45 46 \
  --durations fixed variable --variants main \
  --out-root runs/cpsc2018_postnorm_main_5seeds \
  > logs/cpsc2018_main_5seeds.log 2>&1 &
```

## CinC2017 消融

```bash
nohup bash run_cinc2017_lowmem.sh ./data/CinC2017 cuda:0 \
  --epochs 100 --seeds 42 43 44 45 46 \
  --suite ablations --include-reset-all \
  --out-root runs/cinc2017_postnorm_ablations_5seeds \
  > logs/cinc2017_ablations_5seeds.log 2>&1 &
```

## CPSC2018 消融

```bash
nohup bash run_cpsc2018_lowmem.sh ./data/Training_WFDB cuda:0 \
  --labels ./data/Training_WFDB/labels.csv \
  --reference-csv ./data/REFERENCE.csv \
  --epochs 100 --seeds 42 43 44 45 46 \
  --durations variable \
  --variants all_updates full_trace no_trace reset_neurons reset_all \
  --out-root runs/cpsc2018_postnorm_ablations_5seeds \
  > logs/cpsc2018_ablations_5seeds.log 2>&1 &
```

## 查看日志

```bash
tail -f logs/cinc2017_main_5seeds.log
tail -f logs/cpsc2018_main_5seeds.log
tail -f logs/cinc2017_ablations_5seeds.log
tail -f logs/cpsc2018_ablations_5seeds.log
```
