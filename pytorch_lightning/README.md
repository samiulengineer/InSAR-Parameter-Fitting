# InSAR Filtering Project Seed Template

## Prepare conda env
# create from env.yml
```bash
conda env create -n insar -f conda_env/env.yml 
conda activate insar
```
# manual install 
```bash
conda create --name insar python=3.6
conda activate insar
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pytorch-lightning -c conda-forge
pip install -U hydra-core 
pip install -U MRC-InSAR-Common
```

## Run train real with dummy_cnn and dummy_dncnn
run validation each 500 steps and only do inference on one validation image
you can set processing_dir to arbitrary path
better to put to /mnt/hdd1/{you_name} if you are using mrcg1

powered by [hydra](https://hydra.cc), you can easily switch between different model/data_module as well as their configurations by simply modifying the config file or override in CLI.

use _dummy_cnn_ model
```bash
cd project
python train_real.py \
            model=dummy_cnn \
            data=mrcg1_real_single \
            pl_trainer.val_check_interval=500 \
            pl_trainer.limit_val_batches=1 \
            processing_dir=./test_processing
```
use _dummy_dncnn_ model
```bash
cd project
python train_real.py \
            model=dummy_dncnn \
            data=mrcg1_real_single \
            pl_trainer.val_check_interval=500 \
            pl_trainer.limit_val_batches=1 \
            processing_dir=./quick_seed_processing
```

use _dummy_dncnn_ model with **ri-mse** loss 
```bash
cd project
python train_real.py \
            model=dummy_dncnn \
            model.loss_type=ri-mse \
            data=mrcg1_real_single \
            pl_trainer.val_check_interval=500 \
            pl_trainer.limit_val_batches=1 \
            processing_dir=./quick_seed_processing
```
## Status:
- [x] seed InSAR parameter fitting template with hydra and pytorch_lightning
- [x] real datamodule on mrcg1 - config/data/mrcg1_real

## Extra Information 
conv factors for dataset at :/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/
```
conv1:
bagdad.tsx.sm_dsc = -0.0110745533168
cerroverde.tsx.sm_dsc = -0.0110745533168
chino.csk.sm_asc = -0.0110171730107
chino.tsx.sm_dsc = -0.0110745533168
cortez.tsx.sm_dsc = -0.0110745533168
elabra.tsx.sm_dsc = -0.0110745525134
escondida.tsx.sm_dsc = -0.0110745522839
grasbergmine.tsx.sm_dsc = -0.0110745533168
miami.tsx.sm_dsc = -0.0110745522839
```
```
conv2:
bagdad.tsx.sm_dsc = -0.00134047881374
cerroverde.tsx.sm_dsc = -0.00146586945695
chino.csk.sm_asc = -0.000867724227899
chino.tsx.sm_dsc = -0.00105610756222
cortez.tsx.sm_dsc = -0.00122202886324
elabra.tsx.sm_dsc = -0.00133224968845
escondida.tsx.sm_dsc = -0.00144886933121
grasbergmine.tsx.sm_dsc = -0.00106524620632
miami.tsx.sm_dsc = -0.00160177996359
```
