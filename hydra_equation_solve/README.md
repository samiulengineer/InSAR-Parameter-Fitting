# hydra-pytorch-lightning-seed
A Project Template Seed using Hydra, PyTorch and PyTorch-Lightning to solve a equation (y = ax1+bx2) here, a,b = baseline (for constant a,b = 1,2)

## Main Devstack
- python 3.8.5, [conda/anaconda](https://www.anaconda.com/)
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
├── test_equation.py     # Test entrypoint
├── train_equation.py    # Train entrypoint
```

### Training

Train with predefined configs - model: **eqmodel** and data: **train_data_eqn**


### Training unwrapped with fixed baseline (Experiment 1)
```bash
python train_equation.py \
    model=eqnmodel12 \
    data=train_data_eqn \
    data.experiment=1 \
```

### Training wrapped with fixed baseline (Experiment 2)
```bash
python train_equation.py \
    model=eqnmodel12 \
    data=train_data_eqn \
    data.experiment=2 \
```
### Training wrapped with random baseline (Default) (Experiment 3)
```bash
python train_equation.py \
    model=eqmodel \
    data=train_data_eqn \
    data.experiment=3 \
```
### Testing
Test with pretrained checkpoint file after training 


### Testing unwrapped with fixed baseline (Experiment 1)
```bash
#Recommend to use absolute path for checkpoint_path then you do not need extract $PWD
python test_equation.py 
       model=eqnmodel12 \
       data=test_data_eqn 
       data.experiment=1 \
       checkpoint_path=$PWD'/processing/train/lightning_logs/version_0/checkpoints/epoch\=25-step\=2599.ckpt'
```

### Testing wrapped with fixed baseline (Experiment 2)
```bash
#Recommend to use absolute path for checkpoint_path then you do not need extract $PWD
python test_equation.py \
    model=eqnmodel12 \
    data=test_data_eqn \
    data.experiment=2 \
    checkpoint_path = $PWD'/processing/train/lightning_logs/version_0/checkpoints/epoch\=25-step\=2599.ckpt'
```
### Testing wrapped with random baseline (Experiment 3)
```bash
#Recommend to use absolute path for checkpoint_path then you do not need extract $PWD
python test_equation.py \
       model=eqmodel \
       data=test_data_eqn \
       data.experiment=3 \
       checkpoint_path = $PWD'/processing/train/lightning_logs/version_0/checkpoints/epoch\=25-step\=2599.ckpt'
```