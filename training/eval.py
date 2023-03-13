import pandas as pd
import numpy as np

import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

from dataset import Dataset

df = pd.read_csv("F://NTC_Tickets//csv_small//csv//cleaned_completed_val.csv")
df = df.dropna()
df = df.drop_duplicates()
#df = df.drop(index=df.index[df["network_impact"]=="No_Impact"]).reset_index(drop=True)
#df = df.dropna()

def evaluate(model, test_data):
    target_names = ['Degraded', 'No_Impact', 'Outage', 'Threatened']
#    target_names = ['Degraded', 'Outage', 'Threatened']
    test = Dataset(test_data, train=False)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0

    predicted = []
    actual = []
    with torch.no_grad():

        #for test_input, tf_features,test_label in tqdm(test_dataloader):
        for test_input,test_label in tqdm(test_dataloader):
#        for test_input, test_label in tqdm(test_dataloader):
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              #tf_features = tf_features.to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              #output = model(input_id, tf_features, mask)
              output = model(input_id, mask)
#
#             output = model(input_id, mask)
              #import pdb;pdb.set_trace()
              predicted.extend(output.argmax(dim=1).tolist())
              actual.extend(test_label.tolist())

    print('Accuracy: ', round(accuracy_score(actual, predicted), 3))
    print('Precision: ', round(precision_score(actual, predicted, average='micro', zero_division=1), 3))
    print('Recall: ', round(recall_score(actual, predicted, average='micro'), 3))
    print('F1 score: ', round(f1_score(actual, predicted, average='micro', zero_division=1), 3))

    print(classification_report(actual, predicted, target_names=target_names, digits=3))

    c_m = confusion_matrix(actual, predicted, labels=[0, 1, 2, 3])
    print("Confusion matrix: ")
    print(c_m)


    cm = confusion_matrix(actual, predicted, normalize="true").diagonal()
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    diag = cm.diagonal()
    print("Class-wise Accuracy: ")
    for i, cls in enumerate(target_names):
        print(cls+" : ", round(cm[i], 3))


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased') # it should be uncased
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

#    def forward(self, input_id, tf_features, mask):
    def forward(self, input_id, mask):
        #vectors, _ = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        #pooled_output = torch.sum(vectors*tf_features,axis=1)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        #import pdb;pdb.set_trace()
        return final_layer

model = BertClassifier()
model.load_state_dict(torch.load("./models/best_model.pth"))
model.eval()

evaluate(model, df)
