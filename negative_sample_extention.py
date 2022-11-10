import pandas as pd
filename = "Synthetic_xfm_t5wtense_logical_equivalence_train_v4"
dataframe = pd.read_csv("./output_result/"+filename+".csv")
df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
import random
from sklearn.utils import shuffle

negative_dataframe = dataframe.loc[dataframe['label'] == 0]
negative_dataframe.reset_index(drop=True, inplace=True)

cut_line = dataframe.shape[0]

negative_sample_k = 2  ## add 50% negative samples
for index, row in dataframe.iterrows():
    df = df.append(
        {'sentence1': row["sentence1"], 'sentence2': row["sentence2"],
         'label': row["label"]},
        ignore_index=True)
    
    if row["label"] == 1:
        for i in range(negative_sample_k):
            negative_sentence = negative_dataframe["sentence2"][random.randint(0,negative_dataframe.shape[0]-1)]
            df = df.append(
                {'sentence1': row["sentence1"], 'sentence2': negative_sentence,
                'label': 0},
                ignore_index=True)

df = shuffle(df)
df_final = df.head(cut_line)

df_final.to_csv("./output_result/"+filename+"_negative_samples_1_"+str(negative_sample_k+1)+".csv",index = None,encoding = 'utf8')