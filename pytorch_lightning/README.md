# hydra-pytorch-lightning-seed
A Project Template Seed using Hydra, PyTorch and PyTorch-Lightning to solve an equation.

## Main Devstack
- python 3.6, [conda/anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/)
- [PyTorch-lightning](https://www.pytorchlightning.ai/)
- [hydra](https://hydra.cc/)

## This project seed includes a dummy DnCNN model for MNIST filtering
Structure
```bash
.
├── config      # hydra configures
├── data        # data scripts
├── __init__.py
├── model       # model scripts
├── test.py     # Test entrypoint
├── train.py    # Train entrypoint
```
### Setup python environment
```
# install requirements
conda env create -n mlseed -f ./envs/conda_env.yml
conda activate mlseed
cd project
```
 
### Training

### Experience 01 - Unwrapped with Fixed Baseline

```bash
python train_equation.py \
    model=eqnmodel12 \
    data=train_data_eqn \
    data.experiment=1 > processing/train/log/exp_1.txt
```
### Experience 02 - Wrapped with Fixed Baseline

```bash
python train_equation.py \
    model=eqnmodel12 \
    data=train_data_eqn \
    data.experiment=2 > processing/train/log/exp_2.txt
```

### Experience 03 - Wrapped with Random Baseline
```bash
python train_equation.py \
    model=eqmodel \
    data=train_data_eqn \
    data.experiment=3 > processing/train/log/exp_3.txt
```


### Experience 01 - Unwrapped with Fixed Baseline with Rosenbrock Data

```bash
python train_equation.py \
    model=eqnmodel12 \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \ 
    data.high_limit=1 \
    data.low_limit=-1 \
    data=train_data_eqn \
    data.experiment="rosenbrock" > processing/train/log/rosenbrock_exp_1.txt
```

### Experience 02 - Wrapped with Random Baseline with Rosenbrock Data

```bash
python train_equation.py \
    model=eqnmodel12 \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \ 
    data.high_limit=1 \
    data.low_limit=-1 \
    data=train_data_eqn \
    data.experiment="rosenbrock2" > processing/train/log/rosenbrock_exp_2.txt
```
### Experience 03 - Wrapped with Fixed Baseline with Rosenbrock Data

```bash
python train_equation.py \
    model=eqmodel \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \
    data.high_limit=1 \
    data.low_limit=-1 \
    data=train_data_eqn \
    data.experiment="rosenbrock3" > processing/train/log/rosenbrock_exp_3.txt
```


### Testing
Test with pretrained checkpoint file after training 

__N.B.: Please update the checkpoint_path according to your experience checkpoints. Here I added the dummy checkpoint_path__

### Experience 01 - Unwrapped with Fixed Baseline

```bash
python test_equation.py \
    model=eqnmodel12 \
    data=test_data_eqn \
    data.experiment=1 \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-1 (unwrapped_fixed_baseline)/checkpoints/epoch=499-step=49999.ckpt' \
```
### Experience 02 - Wrapped with Fixed Baseline

```bash
python test_equation.py \
    model=eqnmodel12 \
    data=test_data_eqn \
    data.experiment=2 \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-2 (unwrapped_fixed_baseline)/checkpoints/epoch=499-step=49999.ckpt' \
```

### Experience 03 - Wrapped with Random Baseline

```bash
python test_equation.py \
    model=eqmodel \
    data=test_data_eqn \
    data.experiment=3 \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-3 (wrapped_random_baseline)/checkpoints/epoch=499-step=49999.ckpt' \
```

### Experience 01 - Unwrapped with Fixed Baseline with Rosenbrock Data

```bash
python test_equation.py \
    model=eqnmodel12 \
    data=test_data_eqn \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \
    data.experiment="rosenbrock" \
    data.high_limit=1 \
    data.low_limit=-1 \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-3 (wrapped_random_baseline)/checkpoints/epoch=499-step=49999.ckpt' > processing/train/log/rosenbrock_exp-01_log_test.txt
```

### Experience 02 - Wrapped with Random Baseline with Rosenbrock Data

```bash
python test_equation.py \
    model=eqnmodel12 \
    data=test_data_eqn \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \
    data.experiment="rosenbrock2" \
    data.high_limit=1 \
    data.low_limit=-1 \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-3 (wrapped_random_baseline)/checkpoints/epoch=499-step=49999.ckpt' > processing/train/log/rosenbrock_exp-01_log_test.txt
```
### Experience 03 - Wrapped with Fixed Baseline with Rosenbrock Data

```bash
python test_equation.py \
    model=eqmodel \
    data=test_data_eqn \
    data.high_limit=1 \
    data.low_limit=-1 \
    data._target_=data.rosenbrok_dataloader.EqnDataLoader \
    data.experiment="rosenbrock3" \
    checkpoint_path=$PWD'/processing/train/lightning_logs/Exp-3 (wrapped_random_baseline)/checkpoints/epoch=499-step=49999.ckpt' > processing/train/log/rosenbrock_exp-01_log_test.txt
```