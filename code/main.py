print('Importing pachages...')
import json
import sys
from pathlib import Path

from utils import *

from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

import keras
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model

HW_path = Path(os.getcwd())

data_folder = HW_path / 'data'
results_folder = HW_path / 'results'

cache_dir = HW_path / 'cache'

config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)

## Data
print('Importing Data...')

x_train = np.load(data_folder / 'persica.csv_train_x.npy', allow_pickle=True).tolist()
y_train = np.load(data_folder / 'persica.csv_train_y.npy').tolist()
#x_test = np.load(data_folder / 'persica.csv_test_x.npy', allow_pickle=True).tolist()
#y_test = np.load(data_folder / 'persica.csv_test_y.npy').tolist()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
max_length = 512 #9331 #max([len(tokenizer.tokenize(x)) for x in x_train]) + 2

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length, return_tensors="tf")
val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=max_length, return_tensors="tf")
#test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=max_length)

## Data
print('Creating Datasets...')

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))
#test_dataset = customDataset(test_encodings, y_test)

## Model
print('Creating Model...')

hyperparameters = dict(
    epochs=15,
    batch_size=32,
    lr=5e-5,
    model_name="HooshvareLab/bert-base-parsbert-uncased",
)

try:
    last_model_num = int((results_folder / 'last_model_number.txt').read_text())
except(FileNotFoundError):
    last_model_num = 0
    

model_number = str(last_model_num + 1)

with open(results_folder / 'last_model_number.txt', 'w') as f:
    f.write(model_number)
    
model_path = results_folder / model_number 
    
create_if_not_exist(model_path)
    
hyperparameters_path = results_folder / model_number / 'hyperparameters.json'.format(model_number)

with open(hyperparameters_path, 'w') as file:
     file.write(json.dumps(hyperparameters))

# sys.exit()

model_name = hyperparameters['model_name']
batch_size = hyperparameters['batch_size']
lr = hyperparameters['lr']
epochs = hyperparameters['epochs']

model = TFAutoModel.from_pretrained(model_name, cache_dir=cache_dir)


# inp = Input()
# out = hfmodel(inp)
# out = 

# model = Model(inp, out)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss = keras.losses.CategoricalCrossentropy(name='loss')

model.compile(optimizer=optimizer, loss=loss) # can also use any keras loss fn

## Model Train
print('Training Model...')

# check points 
create_if_not_exist(model_path / 'models')

ckp_path = str(model_path / 'models' / 'weights-improvement-{epoch:02d}.hdf5')
    
filename = model_path / 'train.csv'

history_logger = CSVLogger(filename, separator=",", append=True)
checkpoint = ModelCheckpoint(ckp_path)
callbacks_list = [checkpoint, history_logger]

model.fit(
    x=train_dataset(batch_size),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=val_dataset(batch_size),
    validation_batch_size=batch_size,
)

