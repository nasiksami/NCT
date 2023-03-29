import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from tqdm import tqdm

df = pd.read_csv("F://NTC_Tickets//NCT//data//csv_small//csv//cleaned_completed_val.csv")

df = df.dropna()

print("Degraded: ", len(df[df["network_impact"] == "Degraded"]))
#print("No Impact: ", len(df[df["network_impact"] == "No_Impact"]))
print("Outage: ", len(df[df["network_impact"] == "Outage"]))
print("Threatened: ", len(df[df["network_impact"] == "Threatened"]))

syn_aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3,
                         lang='eng',
                         stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                         force_reload=False,
                         verbose=0)

anto_aug = naw.AntonymAug(name='Antonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng', stopwords=None,
                          tokenizer=None,
                          reverse_tokenizer=None, stopwords_regex=None, verbose=0)

random_aug = naw.RandomWordAug(action='delete', name='RandomWord_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                               target_words=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                               verbose=0)

spell_aug = naw.SpellingAug(dict_path=None, name='Spelling_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                            tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
                            verbose=0)

split_aug = naw.SplitAug(name='Split_Aug', aug_min=1, aug_max=10, aug_p=0.3, min_char=4, stopwords=None, tokenizer=None,
                         reverse_tokenizer=None, stopwords_regex=None, verbose=0)

TOPK = 20  # default=100
ACT = 'insert'  # "substitute"

aug_bert = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased',
    device='cuda',
    action=ACT, top_k=TOPK)

aug_w2v = naw.WordEmbsAug(
    model_type='glove', model_path='./content/glove.6B.300d.txt',
    action="substitute")

aug = naf.Sometimes([
    split_aug, spell_aug, random_aug, anto_aug, syn_aug, aug_bert, aug_w2v
], aug_p=0.5)
# ], aug_p=0.5, pipeline_p=0.5)

# print(df.info())


# degraded = 5x
# no_impact = 0.5x
# Outage = 1x
# Threatened = 2x

degraded_df = df[df["network_impact"] == "Degraded"]
#no_impact_df = df[df["network_impact"] == "No_Impact"]
outage_df = df[df["network_impact"] == "Outage"]
threatened_df = df[df["network_impact"] == "Threatened"]

#no_impact_df = no_impact_df.sample(frac=0.5, random_state=1002)
# list_df = [degraded_df, threatened_df]
aug_times = 5
augmented_df = []
for index, row in tqdm(degraded_df.iterrows()):
    heading = aug.augment(row['_headline'], n=aug_times)
    description = aug.augment(row['_description'], n=aug_times)
    label = ["Degraded" for _ in range(aug_times)]

    for i in range(aug_times):
        tmp_dict = {}
        tmp_dict["_headline"] = heading[i]
        tmp_dict["_description"] = description[i]
        tmp_dict["network_impact"] = label[i]
        augmented_df.append(tmp_dict)

augmented_df = pd.DataFrame(augmented_df)
degraded_df = pd.concat([augmented_df, degraded_df], ignore_index=True)

aug_times = 2
augmented_df = []
for index, row in tqdm(threatened_df.iterrows()):
    heading = aug.augment(row['_headline'], n=aug_times)
    description = aug.augment(row['_description'], n=aug_times)
    label = ["Threatened" for _ in range(aug_times)]

    for i in range(aug_times):
        tmp_dict = {}
        tmp_dict["_headline"] = heading[i]
        tmp_dict["_description"] = description[i]
        tmp_dict["network_impact"] = label[i]
        augmented_df.append(tmp_dict)

augmented_df = pd.DataFrame(augmented_df)
threatened_df = pd.concat([augmented_df, threatened_df], ignore_index=True)

#aug_df = pd.concat([no_impact_df, outage_df, degraded_df, threatened_df], ignore_index=True)
aug_df = pd.concat([ outage_df, degraded_df, threatened_df], ignore_index=True)

print("Degraded: ", len(aug_df[aug_df["network_impact"] == "Degraded"]))
#print("No Impact: ", len(aug_df[aug_df["network_impact"] == "No_Impact"]))
print("Outage: ", len(aug_df[aug_df["network_impact"] == "Outage"]))
print("Threatened: ", len(aug_df[aug_df["network_impact"] == "Threatened"]))

aug_df.to_csv(".//csv_small//csv//Augmented_cleaned_completed_val.csv", index=False)
