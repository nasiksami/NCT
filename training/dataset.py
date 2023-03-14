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

TOP_N = 3

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, train=True):

        self.labels = [labels[label] for label in df['network_impact']]
        self.texts = []
        self.tf_idf = [] #comment for baseline

        for i, row in df.iterrows():
            if row["_headline"] != row["_description"]:
                tmp = tokenizer(str(row["_headline"])+" "+\
                                            str(row["_description"]),
                                            padding='max_length',
                                            max_length=512, truncation=True,
                                            return_tensors="pt")
                self.texts.append(tmp)
                self.tf_idf.append(" ".join([str(x) for x in tmp.input_ids[0].numpy()]))  #comment for baseline
            else:
                tmp = tokenizer(str(row["_description"]),
                                        padding='max_length',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt")
                self.texts.append(tmp)
                self.tf_idf.append(" ".join([str(x) for x in tmp.input_ids[0].numpy()])) #comment for baseline


        # comment full block for baseline
        if train:
            vectorizer = TfidfVectorizer(use_idf=True,
                            smooth_idf=True,
                            ngram_range=(1,1))
            # import pdb;pdb.set_trace()
            tfidf = vectorizer.fit(self.tf_idf)
            pickle.dump(tfidf, open("./tfidf_new_tokenizer.pickle", "wb"))
            self.tf_features = vectorizer.transform(self.tf_idf)
        else:
            vectorizer = pickle.load(open("./tfidf_new_tokenizer.pickle", "rb"))
            self.tf_features = vectorizer.transform(self.tf_idf)

        self.feature_names = np.array(vectorizer.get_feature_names_out())


        #print(len(self.texts))


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
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_tf_features(self, idx):
        feature_names = self.get_top_tf_idf_words(self.tf_features[idx], TOP_N)
        ret_array = np.zeros_like(self.texts[idx]["input_ids"])
        for item in feature_names:
            ret_array[np.where(self.texts[idx]["input_ids"] == int(item))]=1
        return ret_array.reshape((-1,1))

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_tf = self.get_batch_tf_features(idx) #comment for baseline
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_tf, batch_y

        #return batch_texts, batch_y #comment #use for baseline