import os
import sys
import math
import glob
import argparse
import time
from loss import *
import segmentation_models as sm
from model import RBDN
from metrics import get_metrics
from tensorflow import keras
from utils import set_gpu, SelectCallbacks, get_config_yaml, create_paths
from dataset import get_dataloader
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit("True")
#mixed_precision.set_global_policy('mixed_float16')


# Parsing variable
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs")
parser.add_argument("--batch_size")
parser.add_argument("--index")
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
parser.add_argument("--weights")

# args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------

config = get_config_yaml('config.yaml', {})
create_paths(config)


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------

print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weigth = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))



# Dataset
# ----------------------------------------------------------------------------------------------

train_dataset, val_dataset = get_dataloader(config)

config['height'] = train_dataset.patch_size
config['width'] = train_dataset.patch_size
config['in_channels'] = train_dataset.stack_size * 4
config['num_classes'] = 1

# enable training strategy
metrics = list(get_metrics(config).values())
adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])

model = RBDN(config["height"], config["width"], config["in_channels"])
model.compile(optimizer = adam, loss = tf.keras.losses.MeanSquaredError())


# Set up Callbacks
# ----------------------------------------------------------------------------------------------

loggers = SelectCallbacks(val_dataset, model, config)
model.summary()

# fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    validation_data = val_dataset,
                    verbose = 1, 
                    epochs = config['epochs'],
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    validation_freq = 30
                    )
print("training time minute: {}".format((time.time()-t0)/60))
#model.save('/content/drive/MyDrive/CSML_dataset/model/my_model.h5')

# for batch in train_dataset:
#     break