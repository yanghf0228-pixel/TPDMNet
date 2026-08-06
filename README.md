# TPDMNet

This repository provides the review-version implementation of **Temporal Prototype-guided Dynamic Memory Network (TPDMNet)** for multi-step traffic flow forecasting.

This review package keeps the TPDMNet backbone and uses a linear prediction head for the released training entry.

## Environment

```bash
conda create -n tpdmnet python=3.10 -y
conda activate tpdmnet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default package index does not provide a suitable GPU build.

## Data

Place the PeMS datasets under `data/` as follows:

```text
data/
  PeMS03/PEMS03.npz
  PeMS03/PEMS03.csv
  PeMS04/PEMS04.npz
  PeMS04/PEMS04.csv
  PeMS07/PEMS07.npz
  PeMS07/PEMS07.csv
  PeMS08/PEMS08.npz
  PeMS08/PEMS08.csv
```

The `.npz` file should contain a `data` array with traffic flow values. The `.csv` file is used to construct the physical adjacency matrix.

## Training

Run TPDMNet on PeMS08 with seed 10:

```bash
python run.py --dataset PEMSD8 --type P --model TPDMNet --mode train --seed 10 --device cuda:0
```

For a short sanity check:

```bash
python run.py --dataset PEMSD8 --type P --model TPDMNet --mode train --seed 10 --epochs 3 --device cuda:0
```

## Evaluation

After training, the best checkpoint is saved in:

```text
experiments/<dataset>/P/<timestamp>/best_model.pth
```

Evaluate a saved checkpoint with:

```bash
python run.py --dataset PEMSD8 --type P --model TPDMNet --mode test --device cuda:0 --log_dir experiments/PEMSD8/P/<timestamp>
```

## Reproduction Scripts

Shell scripts for the PeMS datasets are provided in `scripts/`:

```bash
bash scripts/run_pems03.sh
bash scripts/run_pems04.sh
bash scripts/run_pems07.sh
bash scripts/run_pems08.sh
```

## Code Organization

```text
model/
  TPDMNet.py        TPDMNet, TLSE, TPMR, DGLP, and prediction heads
  TCDMU.py          Time-Conditioned Dynamic Message Unit
  TCDMN.py          Time-Conditioned Dynamic Message Network
  BasicTrainer.py   Training and evaluation loop
lib/
  dataloader.py     Data loading and graph construction
  metrics.py        MAE, RMSE, and MAPE
config_file/P/      Dataset-specific training configurations
```

The release version removes intermediate experiment logs, local machine paths, and exploratory figure-generation utilities.
