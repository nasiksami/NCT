import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./trained_tokenizer')

heading = 'I will watch horse but not dog'
des = 'this is apple not orange' 

bert_input = tokenizer(heading,padding='max_length', max_length = 50,
                       truncation=True, return_tensors="pt")

#import pdb;pdb.set_trace()
try:
    index = (bert_input.attention_mask[0] == 0).nonzero()[0] #first element
except Exception as ex:
    index = len(bert_input.attention_mask[0])


bert_input2 = tokenizer(des, padding='max_length', max_length=(50-index)+1,truncation=True, return_tensors="pt")

#print(bert_input.input_ids.shape)
elements = (len(bert_input.input_ids[0])-index)+1 # to skip the second [cls] token
input_ids = torch.cat((bert_input.input_ids[0][:index], bert_input.input_ids[0][index:] + bert_input2.input_ids[0][1:elements])).reshape((1,-1))



elements = (len(bert_input.token_type_ids[0])-index)+1
token_type_ids = torch.cat((bert_input.token_type_ids[0][:index], bert_input.token_type_ids[0][index:] + bert_input2.token_type_ids[0][1:elements])).reshape((1,-1))


elements = (len(bert_input.attention_mask[0])-index)+1
attention_mask = torch.cat((bert_input.attention_mask[0][:index], bert_input.attention_mask[0][index:] + bert_input2.attention_mask[0][1:elements])).reshape((1,-1))
#print(input_ids.shape)
#import pdb;pdb.set_trace()

bert_input['input_ids'] = input_ids
bert_input['token_type_ids'] = token_type_ids
bert_input['attention_mask'] = attention_mask

example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)

print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])
