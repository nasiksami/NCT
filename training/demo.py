import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import logging
logging.set_verbosity_error()


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') # it should be uncased
        #self.bert = BertModel.from_pretrained('./language_model/bert-base-uncased-finetuned-NCT')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False) #use for baseline
        #vectors, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        #pooled_output = torch.sum(vectors * tf_features, axis=1)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

model = BertClassifier()
model.load_state_dict(torch.load("./models/best_model_tf_idf_augmented_N1.pth"))


def predict_class(model, sentence):
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True, truncation=True, max_length=512)
    # tokenizer = BertTokenizer.from_pretrained('./language_model/bert-base-uncased-finetuned-NCT')
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_sentence['input_ids'].to(device)
    attention_mask = encoded_sentence['attention_mask'].to(device)
    # tf_features = torch.tensor(get_tf_features([sentence]), dtype=torch.float).to(device)  # comment for baseline
    # tf_features = None #uncomment for baseline

    input_ids = input_ids.to(torch.int64)
    # tf_features = tf_features.to(torch.float32)
    attention_mask = attention_mask.to(torch.float32)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)  # comment for baseline
        # outputs = model(input_ids, attention_mask) #uncomment for baseline
        predicted_class = torch.argmax(outputs).item()
        if predicted_class == 0:
            return "Degraded"
        elif predicted_class == 1:
            return "No_Impact"
        elif predicted_class == 2:
            return "Outage"
        elif predicted_class == 3:
            return "Threatened"

while True:
    headline = input("Please enter the Headline of the ticket: ")
    description = input("Please enter the description for the ticket: ")
    sentence = headline + " " + description
    print()
    print("Input sentence:", sentence)
    prediction = predict_class(model, sentence)
    print("Prediction for your ticket is:", prediction)
    another_prediction = input("Do you want to make another prediction? (yes/no): ")
    if another_prediction.lower() == "no":
        break

print("Network Ticket Classification Program Ended Succesfully.")






