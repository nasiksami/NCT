import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from dataset import Dataset
import time
from transformers import logging
logging.set_verbosity_error()
from sklearn.metrics import classification_report, confusion_matrix
import gc


#df_train = pd.read_csv("./csv_processed_raw/csv/custom_augmented_train.csv")
df_train = pd.read_csv(".//csv_small//csv//cleaned_completed_test.csv")
df_train = df_train.dropna()
df_train = df_train.drop_duplicates()

#df_val = pd.read_csv("./csv_processed_raw/csv/cleaned_completed_test.csv")
df_val = pd.read_csv(".//csv_small//csv//cleaned_completed_test.csv")
df_val = df_val.dropna()
df_val = df_val.drop_duplicates()

print("Training Size: {}\nValidation Size: {}\n".format(len(df_train), len(df_val)))

degraded_weight = len(df_val[df_val["network_impact"]!="Degraded"])/len(df_val)
no_impact_weight = len(df_val[df_val["network_impact"]!="No_Impact"])/len(df_val)
outage_weight = len(df_val[df_val["network_impact"]!="Outage"])/len(df_val)
threatened_weight = len(df_val[df_val["network_impact"]!="Threatened"])/len(df_val)
WEIGHTS = torch.tensor([degraded_weight, no_impact_weight, outage_weight, threatened_weight])


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('./language_model/bert-base-uncased-finetuned-NCT')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, tf_features, mask):
    #def forward(self, input_id, mask): #use for baseline

       # _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False) #use for baseline
        vectors, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        pooled_output = torch.sum(vectors * tf_features, axis=1)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs, n_splits=5):

    global WEIGHTS
    #train = Dataset(train_data, train=True)
    #val = Dataset(val_data, train=False)

    data = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)
    # USING KFOLD CROSS VALIDATION
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    #folds = kfold.split(data)

    # USING STRATIFIED CROSS VALIDATION
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True , random_state=42)
    folds = kfold.split(data, data['network_impact'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)

    criterion = nn.CrossEntropyLoss(weight=WEIGHTS)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    #best_accuracy = 0.0 # for saving one model per epoch
    #best_models = []    # for saving one model per fold
    start_time_model = time.time()


    if use_cuda:
        model = model.cuda()
        WEIGHTS = WEIGHTS.cuda()
        criterion = criterion.cuda()

    for fold, (train_idx, val_idx) in enumerate(folds):

        train_dataset = data.iloc[train_idx]
        val_dataset = data.iloc[val_idx]
        print(f"Fold {fold + 1} - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        train_dataloader = torch.utils.data.DataLoader(Dataset(train_dataset, train=True), batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(Dataset(val_dataset, train=False), batch_size=batch_size)
        best_accuracy_per_fold = 0.0



        #
        # model = BertClassifier()
        # model.to(device)
        # optimizer = Adam(model.parameters(), lr=learning_rate)


        for epoch_num in range(epochs):
            total_acc_train = 0
            total_loss_train = 0
            start_time = time.time()
            model.train()

            for train_input, tf_features, train_label in tqdm(train_dataloader):
                # for train_input, train_label in tqdm(train_dataloader): #use for baseline
                train_label = train_label.to(device)
                tf_features = tf_features.to(device)  # comment for baseline
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, tf_features, mask)
                # output = model(input_id, mask) #use for baseline
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                acc = (output.detach().cpu().argmax(dim=1) == train_label.detach().cpu()).sum().item()

                #acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                model.eval()
                val_preds = []
                val_labels = []

                for val_input, val_tf, val_label in val_dataloader:
                    # for val_input, val_label in val_dataloader: #use for baseline
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    val_tf = val_tf.to(device)  # comment for baseline
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    output = model(input_id, val_tf, mask)
                    # output = model(input_id, mask) #use for baseline
                    val_preds.extend(output.argmax(dim=1).detach().cpu().numpy())
                    val_labels.extend(val_label.detach().cpu().numpy())
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    acc = (output.argmax(dim=1) == val_label).sum().item()  #
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                    | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_dataset): .3f}')

            end_time = time.time()
            print("Time taken to train this epoch ", epoch_num + 1, " : ", (end_time - start_time) / 60, "minutes")


            # save the best model so far per fold

            #if total_acc_val / len(val_data) >= best_accuracy: # for saving model per epoch

            if total_acc_val / len(val_dataset) >= best_accuracy_per_fold:
                best_accuracy_per_fold  = total_acc_val / len(val_dataset)
                best_model = model.state_dict()
                torch.save(best_model,f'./models/Cross_Validation_Experiments/best_model_custom_augmented_fold_{fold + 1}.pth')
                #tasks: how to take multiple models and select the mean of them

            # calculate evaluation metrics at the end of each fold
            if epoch_num == epochs - 1:
                # calculate classification report, confusion matrix, and class-wise accuracy
                target_names = ['Degraded', 'No_Impact', 'Outage', 'Threatened']
                report = classification_report(val_labels, val_preds, target_names=target_names, digits=4, zero_division=1)
                cm = confusion_matrix(val_labels, val_preds)
                class_accuracy = cm.diagonal() / cm.sum(axis=1)
                class_accuracy_output=[]
                for i, cls in enumerate(target_names):
                    class_accuracy_output.append(cls + " : " + str(round(class_accuracy[i], 4)))

                # save metrics to file
                with open(f'./models/Cross_Validation_Experiments/fold_{fold + 1}_epoch_{epoch_num + 1}_metrics.txt', 'w') as f:
                    f.write(f'Classification Report:\n{report}\n\n')
                    f.write(f'Confusion Matrix:\n{cm}\n\n')
                    f.write('Class-wise Accuracy:\n')
                    for cls_acc in class_accuracy_output:
                        f.write(cls_acc + '\n')
                    f.write('\n')



        del model
        del optimizer
        del train_label
        del tf_features
        del mask
        del input_id

        torch.cuda.empty_cache()
        gc.collect()

        model = BertClassifier()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)

        end_time_model = time.time()
        print("Total time taken to train the whole model: ", (end_time_model - start_time_model) / 60, "minutes")


EPOCHS = 2
batch_size = 4
model = BertClassifier()
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)



