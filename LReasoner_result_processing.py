import pandas as pd

# dataframe = pd.read_csv("./output_result/lreasoner_synthetic_sentence_list_negative.csv")
# df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
lreasoner_resultlist = "./output_result/lreasoner_synthetic_extended_sentences_list_updated.csv"
#
# for i in range(0, dataframe.shape[0], 2):
#     df = df.append(
#         {'sentence1': dataframe["Original_Sentence"][i], 'sentence2': dataframe['LReasoner_generated_positive_samples'][i],
#          'label': dataframe["Positive_samples_labels"][i]},
#         ignore_index=True)
#
#     df = df.append(
#         {'sentence1': dataframe["Original_Sentence"][i],
#          'sentence2': dataframe['LReasoner_generated_negative_samples'][i],
#          'label': dataframe["Negative_samples_labels"][i]},
#         ignore_index=True)

# df.to_csv(lreasoner_resultlist)

df = pd.read_csv(lreasoner_resultlist)

our_method_trainset = "./output_result/Synthetic_xfm_t5wtense_logical_equivalence_train.csv"
our_method_devset = "./output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation.csv"

df_ourtrain = pd.read_csv(our_method_trainset)
df_ourdev = pd.read_csv(our_method_devset)

df_trainset = pd.merge(df_ourtrain,df,on=['sentence1','label'],how='left')
df_devset = pd.merge(df_ourdev,df,on=['sentence1','label'],how='left')

df_trainset = df_trainset.drop(columns=['sentence2_x','Unnamed: 0'])
df_trainset = df_trainset.rename(columns={'sentence2_y':'sentence2'})

df_devset = df_devset.drop(columns=['sentence2_x','Unnamed: 0'])
df_devset = df_devset.rename(columns={'sentence2_y':'sentence2'})

df_trainset.to_csv("./output_result/Synthetic_LReasoner_logical_equivalence_train.csv",index = None,encoding = 'utf8')
df_devset.to_csv("./output_result/Synthetic_LReasoner_logical_equivalence_validation.csv",index = None,encoding = 'utf8')
