import time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
start_time = time.time()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast
import sqlite3 as sql
import sys

# I typically use "cpu" because my laptop does not include an NVIDIA GPU
# There really is not too much data, so it does not matter if I have a gpu to train
# training typically takes about 0.5 hours
# warning: changing "cpu" to "gpu" will result in errors.
device = torch.device("cpu")

# create the default architecture for a Bert model
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.2) # changing this does not provide any change to accuracy
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

# training using data from giant.db
src = '../GiantDB/giant.db'

# create the connection
conn = None
try:
    conn = sql.connect(src)
except sql.Error as e:
    print(e)
    sys.exit(1)

# load the data
df = pd.read_sql_query("select distinct(title), yes_no from giant_jobs", conn)
# won't be needing this anymore
conn.close()
#print(df.head) # -> everything looks good

# create the test, train, and validation sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['title'], df['yes_no'],
                                                                    test_size=0.3,
                                                                    stratify=df['yes_no'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                test_size=0.3,
                                                                stratify=temp_labels)

# print("test_text", type(test_text)) # Series

# import the bert model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# the set of length of messages in the training set
set_len = [len(i.split()) for i in train_text]

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    padding='max_length',
    truncation=True
)

## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

## to test the dataset after you have a saved_weights.pt file, uncomment below

# print("Test Text:\n", test_text)
# print("\nTest Seq:\n", test_seq)
# print("\nTest Y:\n", test_y)
#
# load weights of best model
# path = 'saved_weights.pt'
# model = BERT_Arch(bert)
# model.load_state_dict(torch.load(path))
#
# start = time.time();
# # get predictions for test data
# with torch.no_grad():
#     preds = model(test_seq.to(device), test_mask.to(device))
#     preds = preds.detach().numpy()
#
# preds = np.argmax(preds, axis=1)
# print("Time to predict:", time.time()-start)
# print("\nPredictions:\n", preds)
# # print(classification_report(test_y, preds))
#
# sys.exit()

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# freeze all the parameters so you don't change the default bert unless necessary
for param in bert.parameters():
    param.requires_grad = False

# create the model for bert data and set model to run on cpu
model = BERT_Arch(bert)
model = model.to(device)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

from sklearn.utils.class_weight import compute_class_weight

# compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print("Class Weights:", class_weights)

# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)

# send the weights to compute on the CPU
weights = weights.to(device)

# use the cross entropy loss function
cross_entropy = nn.NLLLoss(weight=weights)

epochs = 10

# now to train the BERT model

def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    # iterate over all of the batches
    for step, batch in enumerate(train_dataloader):
        # update the progress after every 50 batches not the 0th batch
        if step % 50 == 0 and step != 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))
        # send all batches to the cpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # clean previously calculated gradients
        model.zero_grad()
        # get the model predictions
        preds = model(sent_id, mask)
        # get the model loss
        loss = cross_entropy(preds, labels)
        # add on loss to total_loss
        total_loss += loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the gradients to 1.0 to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update the parameters
        optimizer.step()
        # put model predictions to CPU if not already
        # preds = preds.detach().cpu().numpy()
        preds = preds.detach().numpy()
        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # the predictions are in the form of (batch #, match size, class #)
    # reshape the predictions in the form of (# samples, # classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds

# evaluate the model

def evaluate():
    print("\nEvaluating...")
    # deactivate the dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(val_dataloader):
        # give an update every 50 batches
        if step % 50 == 0 and step != 0:
            elapsed = format_time(time.time()-t0)
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        # put the batch on the CPU
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        # don't want autograd for this part
        with torch.no_grad():
            # get model predictions & cross entropy loss
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            # update total loss
            total_loss += loss.item()
            # if not on cpu, push predictions to cpu
            # preds = preds.detach().cpu().numpy()
            preds = preds.detach().numpy()
            total_preds.append(preds)
        avg_loss = total_loss / len(val_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        return avg_loss, total_preds

# now to start fine-tuning the model

# set initial loss to infinite
best_valid_loss = float('inf')
# initialize train loss and valid loss for each epoch
train_losses, valid_losses = [], []

# train and evaluate for each epoch
# save the weights when you get a better validation loss score, less is better
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print("Saving a better model...")
        torch.save(model.state_dict(), 'saved_weights.pt')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"\nTraining Loss: {train_loss:.3f}")
    print(f"\nValidation Loss: {valid_loss:.3f}")

print("Completed Training in:", (time.time() - start_time), "sec.")
