# InSar-Parameter fitting 
```Tensorflow.keras``` Implementation

## Models

In this repository we implement custom NewCnn and RBDN architecture using `Keras-TensorFLow` framework. The following models are available in this repository.

* NewCnn
* RBDN

## Setup

First clone the github repo in your local or server machine by following:
```
git clone https://github.com/samiulengineer/InSAR-Coding.git
```

Create a new environment and install dependency from `requirement.txt` file. Before start training check the variable inside config.yaml i.e. `height`, `in_channels`.

## Experiments

After setup the required folders and package run one of the following code.
```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name rbdn \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment regular \
    --patchify False \
    --patch_size 512 \
    --weights False \
```

## Testing

Run following model for evaluating train model on test dataset.
```
python train.py --gpu "0" \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name rbdn \
    --load_model_name my_model.hdf5 \
    --plot_single False \
    --index -1 \
    --patchify False \
    --patch_size 512 \
    --experiment regular \
```
