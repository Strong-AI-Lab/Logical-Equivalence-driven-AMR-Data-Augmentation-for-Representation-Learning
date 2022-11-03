import pandas as pd

dataframe = pd.read_csv("./Synthetic_xfm_t5wtense_logical_equivalence_list_filtered_bleu_score_4_76.csv")
df_shuffled = dataframe.sample(frac=1).reset_index(drop=True)

total_lines = df_shuffled.shape[0]

trainset_index = int(total_lines * 0.8)
training_set = df_shuffled.iloc[:trainset_index]
validation_set = df_shuffled.iloc[trainset_index:total_lines]

training_set.columns=['Origin','sentence1','sentence2','BLEU_Score','label','Tag','logic_words']
validation_set.columns=['Origin','sentence1','sentence2','BLEU_Score','label','Tag','logic_words']

training_set = training_set.drop(['Origin','BLEU_Score','Tag','logic_words'],axis=1)
validation_set = validation_set.drop(['Origin','BLEU_Score','Tag','logic_words'],axis=1)
training_set.to_csv("./output_result/Synthetic_xfm_t5wtense_logical_equivalence_train_v5.csv",index = None,encoding = 'utf8')
validation_set.to_csv("./output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation_v5.csv",index = None,encoding = 'utf8')

##############################################################
# import pandas as pd
#
# # dataframe = pd.read_csv("./output_result/lreasoner_synthetic_sentence_list_negative.csv")
# # df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
# our_model_resultlist = "./output_result/Synthetic_xfm_t5wtense_logical_equivalence_list_renamed.csv"
# #
# # for i in range(0, dataframe.shape[0], 2):
# #     df = df.append(
# #         {'sentence1': dataframe["Original_Sentence"][i], 'sentence2': dataframe['LReasoner_generated_positive_samples'][i],
# #          'label': dataframe["Positive_samples_labels"][i]},
# #         ignore_index=True)
# #
# #     df = df.append(
# #         {'sentence1': dataframe["Original_Sentence"][i],
# #          'sentence2': dataframe['LReasoner_generated_negative_samples'][i],
# #          'label': dataframe["Negative_samples_labels"][i]},
# #         ignore_index=True)
#
# # df.to_csv(our_model_resultlist)
#
# df = pd.read_csv(our_model_resultlist)
#
# our_method_trainset = "./output_result/Synthetic_xfm_t5wtense_logical_equivalence_train.csv"
# our_method_devset = "./output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation.csv"
#
# df_ourtrain = pd.read_csv(our_method_trainset)
# df_ourdev = pd.read_csv(our_method_devset)
#
# df_trainset = pd.merge(df_ourtrain,df,on=['sentence1','label'],how='left')
# df_devset = pd.merge(df_ourdev,df,on=['sentence1','label'],how='left')
#
# df_trainset = df_trainset.drop(columns=['sentence2_x'])
# df_trainset = df_trainset.rename(columns={'sentence2_y':'sentence2'})
#
# df_devset = df_devset.drop(columns=['sentence2_x'])
# df_devset = df_devset.rename(columns={'sentence2_y':'sentence2'})
#
# df_trainset.to_csv("./output_result/Synthetic_our_model_logical_equivalence_train_version2.csv",index = None,encoding = 'utf8')
# df_devset.to_csv("./output_result/Synthetic_our_model_logical_equivalence_validation_version2.csv",index = None,encoding = 'utf8')
