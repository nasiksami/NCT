import pandas as pd
import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import KFold
from dataset import Dataset
import sklearn
import time

#df_train = pd.read_csv("./csv_processed_raw/csv/custom_augmented_train.csv")
df_train = pd.read_csv(".//csv_small//csv//cleaned_completed_test.csv")
df_train = df_train.dropna()
df_train = df_train.drop_duplicates()

#df_val = pd.read_csv("./csv_processed_raw/csv/cleaned_completed_test.csv")
df_val = pd.read_csv(".//csv_small//csv//cleaned_completed_test.csv")
df_val = df_val.dropna()
df_val = df_val.drop_duplicates()

print("Training Size: {}\nValidation Size: {}\n".format(len(df_train), len(df_val)))


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #self.bert = BertModel.from_pretrained('./language_model/bert-base-uncased-finetuned-NCT')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

  #  def forward(self, input_id, tf_features, mask):

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        #pooled_output = torch.sum(vectors * tf_features, axis=1)

        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    global WEIGHTS
    train = Dataset(train_data, train=True)
    val = Dataset(val_data, train=False)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    criterion = nn.CrossEntropyLoss()  # weight=WEIGHTS)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    #best_model = None
    start_time_model = time.time()

    if use_cuda:
        model = model.cuda()
        # WEIGHTS = WEIGHTS.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        start_time = time.time()

        #for train_input, tf_features, train_label in tqdm(train_dataloader):

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            #tf_features = tf_features.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            #output = model(input_id, tf_features, mask)
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

          #  for val_input, val_tf, val_label in val_dataloader:
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
               # val_tf = val_tf.to(device) #comment
                input_id = val_input['input_ids'].squeeze(1).to(device)

               # output = model(input_id, val_tf, mask)
                output = model(input_id, mask) #comment

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        end_time = time.time()
        print("Time taken to train this epoch " ,epoch_num, " : ", end_time - start_time, "seconds")

        # if (epoch_num+1)%2==0:
        #    torch.save(model.state_dict(), "./models/completed_weighted_fusion_model_"+str(epoch_num+1)+".pth")


        # save the best model so far
        if total_acc_val / len(val_data) >= best_accuracy:
            best_accuracy = total_acc_val / len(val_data)
            best_model = model.state_dict()
            torch.save(best_model, './models/baseline_custom_data_model.pth')

    end_time_model = time.time()
    print("Total time taken to train the whole model: ", end_time_model - start_time_model, "seconds")


    # saving the best model from all epochs
    #torch.save(best_model, './models/best_model_from_epoch_' + str(epoch_num + 1) + '.pt')


EPOCHS =10
model = BertClassifier()
# model.load_state_dict(torch.load("./completed_weighted_fusion_model_10.pth"), strict=True)
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)
#torch.save(model.state_dict(), "./completed_weighted_fusion_model_new_tokenizer_n1_back_translated_10.pth")
