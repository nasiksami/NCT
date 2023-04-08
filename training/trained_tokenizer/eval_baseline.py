import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix
from dataset import Dataset

#df = pd.read_csv("F://NTC_Tickets//NCT//data//csv_small//csv//cleaned_completed_val.csv")
df = pd.read_csv("./csv_processed_raw/csv/cleaned_completed_test.csv")
df = df.dropna()
df = df.drop_duplicates()


def evaluate(model, test_data):
    target_names = ['Degraded', 'No_Impact', 'Outage', 'Threatened']
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


        for test_input,test_label in tqdm(test_dataloader):

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
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

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

model = BertClassifier()
model.load_state_dict(torch.load("./models/baseline_custom_data_model.pth"))
#model.load_state_dict(torch.load("F://NTC_Tickets//NCT//training//models//best_model_tf_idf_augmented.pth"))
model.eval()
evaluate(model, df)