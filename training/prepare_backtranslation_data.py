import pandas as pd
import nltk
from tqdm import tqdm
import random
import googletrans
from googletrans import Translator
#from textblob import TextBlob
#from textblob.translate import NotTranslated


path = './CSV/completed_train.csv'
df = pd.read_csv(path)
df = df.dropna()

degraded_df = df[df["network_impact"]=="Degraded"].reset_index(drop=True)
no_impact_df = df[df["network_impact"]=="No_Impact"].reset_index(drop=True)
no_impact_df = no_impact_df.drop_duplicates()
outage_df = df[df["network_impact"]=="Outage"].reset_index(drop=True)
threatened_df = df[df["network_impact"]=="Threatened"].reset_index(drop=True)

print("Degraded: ", len(degraded_df))
print("No Impact: ", len(no_impact_df))
print("Outage: ", len(outage_df))
print("Threatened: ", len(threatened_df))

TARGET = len(no_impact_df) # 120986

sr = random.SystemRandom()
language = list(googletrans.LANGUAGES.keys()) # ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]
translator = Translator()


# augmenting degraded class
for i in tqdm(range(TARGET-len(degraded_df))):
    random_idx = random.choice(degraded_df.index)
    headline = str(degraded_df.loc[random_idx, '_headline'])
    description = str(degraded_df.loc[random_idx, '_description'])
    #import pdb;pdb.set_trace()
    try:
        to_lang = sr.choice(language)
        text1 = translator.translate(headline, src='en', dest=to_lang)# TextBlob(headline)
        #text1 = text1.translate(to=to_lang)
        text1 = translator.translate(text1.text, dest="en")


        text2 = translator.translate(description, src='en', dest=to_lang)#TextBlob(description)
        #text2 = text2.translate(to=to_lang)
        text2 = translator.translate(text2.text, dest='en') # text2.translate(to="en")
    except Exception as ex:
        print("Not Translated ", ex)

    degraded_df.loc[len(degraded_df)] = [text1.text, text2.text, "Degraded"]
    degraded_df = degraded_df.sample(frac=1).reset_index(drop=True)

# augmenting outage class
for i in tqdm(range(TARGET-len(outage_df))):
    random_idx = random.choice(outage_df.index)
    headline = str(outage_df.loc[random_idx, '_headline'])
    description = str(outage_df.loc[random_idx, '_description'])
    #import pdb;pdb.set_trace()
    try:
        to_lang = sr.choice(language)
        text1 = translator.translate(headline, src='en', dest=to_lang)# TextBlob(headline)
        #text1 = text1.translate(to=to_lang)
        text1 = translator.translate(text1.text, dest="en")


        text2 = translator.translate(description, src='en', dest=to_lang)#TextBlob(description)
        #text2 = text2.translate(to=to_lang)
        text2 = translator.translate(text2.text, dest='en') # text2.translate(to="en")
    except Exception as ex:
        print("Not Translated ", ex)

    outage_df.loc[len(outage_df)] = [text1.text, text2.text, "Outage"]
    outage_df = outage_df.sample(frac=1).reset_index(drop=True)


# augmenting threatened class
for i in tqdm(range(TARGET-len(threatened_df))):
    random_idx = random.choice(threatened_df.index)
    headline = str(threatened_df.loc[random_idx, '_headline'])
    description = str(threatened_df.loc[random_idx, '_description'])
    #import pdb;pdb.set_trace()
    try:
        to_lang = sr.choice(language)
        text1 = translator.translate(headline, src='en', dest=to_lang)# TextBlob(headline)
        #text1 = text1.translate(to=to_lang)
        text1 = translator.translate(text1.text, dest="en")


        text2 = translator.translate(description, src='en', dest=to_lang)#TextBlob(description)
        #text2 = text2.translate(to=to_lang)
        text2 = translator.translate(text2.text, dest='en') # text2.translate(to="en")
    except Exception as ex:
        print("Not Translated ", ex)

    threatened_df.loc[len(threatened_df)] = [text1.text, text2.text, "Threatened"]
    threatened_df = threatened_df.sample(frac=1).reset_index(drop=True)


print("Degraded: ", len(degraded_df))
print("No Impact: ", len(no_impact_df))
print("Outage: ", len(outage_df))
print("Threatened: ", len(threatened_df))

new_df = pd.concat([degraded_df, no_impact_df, outage_df, threatened_df], ignore_index=True)
new_df = new_df.sample(frac=1).reset_index(drop=True)
new_df.to_csv("./CSV/completed_train_back_translated.csv", index=False)
