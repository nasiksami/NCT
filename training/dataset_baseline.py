import torch
import numpy as np
from transformers import BertTokenizer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#tokenizer = AutoTokenizer.from_pretrained('./trained_tokenizer')
#tokenizer = BertTokenizer.from_pretrained('./trained_tokenizer')
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


labels = {'Degraded':0,
          'No_Impact':1,
          'Outage':2,
          'Threatened':3
          }
TOP_N = 1

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, train=True):

        self.labels = [labels[label] for label in df['network_impact']]
        self.texts = []

        for i, row in df.iterrows():
            if row["_headline"] != row["_description"]:
                tmp = tokenizer(str(row["_headline"])+" "+\
                                            str(row["_description"]),
                                            padding='max_length',
                                            max_length=512, truncation=True,
                                            return_tensors="pt")
                self.texts.append(tmp)

            else:
                tmp = tokenizer(str(row["_description"]),
                                        padding='max_length',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt")
                self.texts.append(tmp)

    def get_top_tf_idf_words(self, response, top_n=2):
        sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
        return self.feature_names[response.indices[sorted_nzs]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_tf_features(self, idx):
        feature_names = self.get_top_tf_idf_words(self.tf_features[idx], TOP_N)
        ret_array = np.zeros_like(self.texts[idx]["input_ids"])
        for item in feature_names:
            ret_array[np.where(self.texts[idx]["input_ids"] == int(item))]=1
        return ret_array.reshape((-1,1))

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y #comment