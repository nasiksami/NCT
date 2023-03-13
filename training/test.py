import pandas as pd
import torch
from dataset import Dataset

#df = pd.read_csv("./CSV/cleaned_dataset.csv", nrows=50)

#_list = [str(row["_headline"]) +" "+ str(row["_description"]) for i,row in df.iterrows() if row["_headline"] != row["_description"]]

#print(_list)

#train = Dataset(df, train=False)
#print(len(train))
#dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
#print(dataloader)
#for inp, tf, label in dataloader:
#    import pdb;pdb.set_trace()
#    print(tf)

x = torch.randn(2, 1, 5)
mask = x.ge(0.5)
print(x)
print(mask)

print(torch.masked_select(x, mask))
