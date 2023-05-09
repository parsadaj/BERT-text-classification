print('Importing packages...')
import json
import sys
from pathlib import Path

from utils import *

from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModel

from tqdm import tqdm

# import keras
# import tensorflow as tf
# from keras.callbacks import CSVLogger, ModelCheckpoint
# from keras.layers import Input
# from keras.models import Model

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import functional as F

HW_path = Path(os.getcwd())

data_folder = HW_path / 'data'
results_folder = HW_path / 'results'

cache_dir = HW_path / 'cache'

# config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)

## Data
print('Importing Data...')

# x_train = np.load(data_folder / 'persica.csv_train_x.npy', allow_pickle=True).tolist()
# y_train = np.load(data_folder / 'persica.csv_train_y.npy').tolist()
x_test = np.load(data_folder / 'persica.csv_test_x.npy', allow_pickle=True).tolist()
y_test = np.load(data_folder / 'persica.csv_test_y.npy').tolist()

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
max_length = 512 #max([len(tokenizer.tokenize(x)) for x in x_train]) + 2

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=max_length)
#test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=max_length)

print('Creating Datasets...')

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, y_train)
val_dataset = IMDbDataset(val_encodings, y_val)

## Model
print('Creating Model...')

hyperparameters = dict(
    epochs=15,
    batch_size=1,
    lr=5e-5,
    model_name="HooshvareLab/bert-base-parsbert-uncased",
    warmup_steps=500,
    weight_decay=0.01
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
warmup_steps = hyperparameters['warmup_steps']
weight_decay = hyperparameters['weight_decay']

num_labels = 11
device = 'cuda'

base_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, num_labels=num_labels)
model = CustomTorchModel(base_model, num_labels)
model.to(device)

model.train()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optim = AdamW(model.parameters(), lr)
loss_f = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print('epoch {}'.format(epoch))
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = F.one_hot(batch['labels'], num_classes=num_labels).float().to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
        loss = loss_f(outputs, labels)
        loss.backward()
        optim.step()
    print()

model.eval()


# training_args = TrainingArguments(
#     output_dir=model_path,          # output directory
#     num_train_epochs=epochs,              # total number of training epochs
#     per_device_train_batch_size=batch_size,  # batch size per device during training
#     per_device_eval_batch_size=batch_size,   # batch size for evaluation
#     warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
#     weight_decay=weight_decay,               # strength of weight decay
#     logging_dir=model_path / 'logs.csv',            # directory for storing logs
#     logging_steps=1,
# )

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss
    
# trainer = CustomTrainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset,             # evaluation dataset
# )

# trainer.train()