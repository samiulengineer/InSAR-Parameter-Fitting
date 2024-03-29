dl-optimizer-poc

Generate POC dataset

python ./data/generate_train_db.py


Train POC FC model

python train.py \
    model=poc_fc \
    data=poc_pf \
    processing_dir='./processing/train/pocfc/' \


Test POC FC model
python test.py \
    model=poc_fc \
    data=poc_pf \
    data.train=False \
    +pl_trainer.deterministic=True \
    checkpoint_path=$PWD'/processing/train/pocfc/lightning_logs/version_0/checkpoints/epoch\=99-step\=6199.ckpt' \
    processing_dir='./processing/test/pocfc/' \



    
hydra-pytorch-lightning-seed
A Project Template Seed using Hydra, PyTorch and PyTorch-Lightning

Main Devstack
python 3.6, conda/anaconda
PyTorch
PyTorch-lightning
hydra
This project seed includes a dummy DnCNN model for MNIST filtering
Structure

.
├── config      # hydra configures
├── data        # data scripts
├── __init__.py
├── model       # model scripts
├── test.py     # Test entrypoint
├── train.py    # Train entrypoint
Setup python environment
# install requirements
conda env create -n mlseed -f ./envs/conda_env.yml
conda activate mlseed
cd project
Training
Train with predefined configs - model: dncnn_small and data: train_dummy_mnist

python train.py \
    model=dncnn_small \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_small_mnist/' \
Train with predefined configs - model: dncnn_small and data: train_dummy_mnist

python train.py \
    model=dncnn_large \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_large_mnist/' \
Train with CLI custom configs by overriding existing configs. hydra override grammar

# override existing config in yaml file
python train.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_mid_mnist/' \

# or append new config for pl.trainer with +
python train.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    +pl_trainer.benchmark=True \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_mid_mnist/' \
Resume a training
Specify a flag of lightning trainer

python train.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    +pl_trainer.benchmark=True \
    pl_trainer.max_epochs=1000 \
    +pl_trainer.resume_from_checkpoint=$PWD'/processing/train/dncnn_mid_mnist/lightning_logs/version_0/checkpoints/epoch\=19-step\=8450.ckpt' \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_mid_mnist_res/' \
Testing
Test with pretrained checkpoint file after training

# Recommend to use absolute path for checkpoint_path then you do not need extract $PWD
python test.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    data=test_dummy_mnist \
    +pl_trainer.deterministic=True \
    checkpoint_path=$PWD'/processing/train/dncnn_mid_mnist/lightning_logs/version_0/checkpoints/epoch\=19-step\=8450.ckpt' \
    processing_dir='./processing/test/dncnn_mid_mnist/' \