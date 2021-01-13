import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
import numpy as np
import time

string = ["Learn about the recruiting process", "Learn recruiting process"]
df = pd.DataFrame(string, columns=["text"])

device = torch.device("cpu")
bert = AutoModel.from_pretrained('bert-base-uncased')

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokens_test = tokenizer.batch_encode_plus(
    [strr.lower() for strr in string],
    max_length = 10, # was '25'
    padding='max_length',
    truncation=True
)
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

path = 'saved_weights.pt'
model = BERT_Arch(bert)
model.load_state_dict(torch.load(path))

start = time.time()
# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().numpy()

preds = np.argmax(preds, axis=1)
print("Time to predict:", time.time()-start)
print("\nPredictions:\n", preds)
print(string)
