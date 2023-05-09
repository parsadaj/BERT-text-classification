import numpy as np
import os
import re
import subprocess
import json


URL = r" <URL> "
NUM = r" <NUM> "
USER = r" <USERNAME> "

SEP = ' '
NEWLINE = '\n'

STOP_WORDS = [
    'در',
    'از',
    'که',
    'با',
    'و',
    'یا',
    'به',
    'تا',
    'حتی',
    'ها',
    'های',
    'را',
    'برای',
    '.',
    '؟',
    '!',
    '،',
    'اما',
    'زیرا',
    'چون'
]


# def preprocess_text(text: str):
#     text = re.sub("\.[\.]+|[()\[\]{}<>\"\']", '', text)

#     text = re.sub("ك", 'ک', text)
#     text = re.sub('ي', 'ی', text)
#     text = re.sub('ؤ', 'و', text)
#     text = re.sub('[أآ]', 'ا', text)
#     text = re.sub('ة', 'ه', text)

#     url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
#     text = re.sub(url_pattern, URL, text)

#     ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:(\d)+)?'
#     text = re.sub(ip_pattern, URL, text)

#     email_pattern = '\S+@\S+'
#     text = re.sub(email_pattern, URL, text)

#     legal_farsi = 'ا-ی'
#     legal_arabic = 'ئ'
#     legal_english = 'a-zA-z'
#     legal_number = '0-9۰-۹'
#     legal_other = '\s<>\n'

#     illegal_pattern = "[^{}]".format(legal_arabic + legal_english + legal_farsi + legal_number + legal_other)

#     text = re.sub(illegal_pattern, ' ', text)

#     text = re.sub("(?<=[a-zئA-Zا-ی])(?=\d)", ' ', text)
#     text = re.sub("(?<=\d)(?=[a-zئA-Zا-ی])", ' ', text)

#     num_pattern = r"\b\d+\b"
#     text = re.sub(num_pattern, NUM, text)
    

#     words = re.split("\s+", text.strip())

    
#     return words


def create_datasets(data_path, results_path):
    f = open(data_path, encoding='utf-8')
    
    n_lines = get_num_lines(data_path) // 7 + 1
    counter = 0
    
    texts = {'train':[], 'test':[]}   
    labels = {'train':[], 'test':[]}   
    
    label_to_index = {}
    index_to_label = {}

    while True:
        if (counter+1) % 1000 == 0:
            print('{} / {}'.format(counter+1, n_lines), end='\r')
        counter += 1
        
        for i in range(7):
            temp = f.readline()
            if i == 2:
                doc = temp
            if i == 6:
                label_kolli = temp

        if len(doc) == 0:
            break
        
        doc = SEP + doc + SEP
                
        phase = np.random.choice(['train', 'test'], p=[0.8, 0.2])
        
        doc = re.sub(r'(?!<\d)\.(?!\d)|[^\s\w.]', ' ', doc)
        
        for stop_word in STOP_WORDS:
            stop_pattern = SEP + stop_word + SEP
            doc = re.sub(stop_pattern, SEP*2, doc)
        for stop_word in STOP_WORDS:
            stop_pattern = SEP + stop_word + SEP
            doc = re.sub(stop_pattern, SEP*2, doc)
        
        if len(doc.strip()) == 0:
            continue
        
        texts[phase].append(doc.strip())
        
        if label_kolli in label_to_index:
            label_index = label_to_index[label_kolli]
        else:
            label_index = len(label_to_index)
            label_to_index[label_kolli] = label_index
            index_to_label[label_index] = label_kolli
            
        labels[phase].append(label_index)

        
    f.close()
    
    save_path_train = data_path + '_train_x.npy'
    save_path_test = data_path + '_test_x.npy'
    
    save_path_train_y = data_path + '_train_y.npy'
    save_path_test_y = data_path + '_test_y.npy'

    np.save(save_path_test, texts['test'])
    np.save(save_path_train, texts['train'])
    
    np.save(save_path_test_y, labels['test'])
    np.save(save_path_train_y, labels['train'])
    
    with open(os.path.join(results_path, "index_to_label.json"), "w", encoding='utf-8') as fp:
        json.dump(index_to_label, fp) 
        
    with open(os.path.join(results_path, "label_to_index.json"), "w", encoding='utf-8') as fp:
        json.dump(label_to_index, fp) 


def pad_list(l: list, content, width):
    l.extend([content for _ in range((width - len(l)))])
    return l
    
def get_num_lines(path):
    """counts number of lines in the given file

    Args:
        path (string): path to file

    Returns:
        int: number of lines
    """
    path = os.path.normpath(path)
    if os.name != 'posix' or os.name == 'nt':
        resp = subprocess.check_output(' '.join(['find', '/v', '/c', '"&*fake&*"', path]))
        return int(resp.split()[2])
    
    return int(subprocess.check_output(['wc', '-l', path]).split()[0])

# if __name__ == '__main__':
#     preprocess_text('سلام بمنلت. تانخلتپلذ لذر؟ بیهت نن تد پن پت م!')
    
# import keras
# import tensorflow as tf
# class customDataset(keras.utils.Sequence):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#         self.last_index = 0
#         self.n = len(labels)

#     def __getitem(self, i, j):
#         item = {key:  tf.convert_to_tensor(np.array(val[i:j]), dtype=tf.float32) for key, val in self.encodings.items()}
        
#         item['label'] =  tf.convert_to_tensor(np.array(self.labels[i:j]), dtype=tf.float32)
#         return item

#     def __len__(self):
#         return len(self.labels)
    
#     def __call__(self, batch_size):
#         i = 0
#         while True:
#             if i >= len(self):
#                 i = i % len(self)
                
#             batch_start = i
#             batch_end = i + batch_size
        
#             batch = self.__getitem(batch_start, batch_end)
    
#             yield batch
#             i = batch_end
        


def create_if_not_exist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        

import torch.nn as nn
class CustomTorchModel(nn.Module):
    def __init__(self, base_model, num_labels, head):
        super(CustomTorchModel, self).__init__()
        
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.linear = nn.Linear(768, num_labels)
        
        self.feature_extractor1 = nn.Linear(768, 256)
        self.feature_extractor2 = nn.Linear(256, 64)
        self.feature_extractor3 = nn.Linear(64, 11)
        
        self.layer__s = [base_model, dropout,]
            
        if head == 2:
            self.activation = nn.ReLU() # relu for model 21
        
        if head == 3:
            self.activation = nn.Sigmoid()
                
        self.softmax = nn.Softmax()
        self.head = head
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        if self.head == 1:
            outputs = outputs[0][:, 0, ...]
            outputs = self.dropout(outputs)
            outputs = self.linear(outputs)
            outputs = self.softmax(outputs) 
        elif self.head in (2, 3):
            outputs = outputs[0][:, 0, ...]
            outputs = self.dropout2(outputs)
            outputs = self.feature_extractor1(outputs)
            outputs = self.activation(outputs)
            outputs = self.feature_extractor2(outputs)
            outputs = self.activation(outputs)
            outputs = self.feature_extractor3(outputs)
            outputs = self.softmax(outputs)
        elif self.head == 4:
            outputs = outputs[0][:, 0, ...]
            outputs = self.dropout2(outputs)
            outputs = self.feature_extractor1(outputs)
            outputs = self.activation(outputs)
            outputs = self.feature_extractor2(outputs)
            outputs = self.activation(outputs)
            outputs = self.feature_extractor3(outputs)
            outputs = self.softmax(outputs)
               
        return outputs


