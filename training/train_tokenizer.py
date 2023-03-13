import pandas as pd
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer

path = "./CSV/completed_train.csv"
df = pd.read_csv(path)
print(df.shape)
df = df.dropna()
print(df.shape)

def get_training_corpus():
    #for start_idx in range(0, len(df), 1000):
    #    samples = df[start_idx : start_idx + 1000]
        for i, row in df.iterrows():
#            import pdb;pdb.set_trace()
            if row["_headline"] != row["_description"]:
                yield row["_headline"] + " " + row["_description"]
            else:
                yield row["_headline"]
        #yield samples["_description"]

training_corpus = get_training_corpus()


old_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 30522 is the vocabulary size
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 30522)
example = df.loc[0, "_description"]

tokens = tokenizer.tokenize(example)
print("New tokens", tokens)
print("Old tokens", old_tokenizer.tokenize(example))

tokenizer.save_pretrained("./trained_tokenizer")

#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
