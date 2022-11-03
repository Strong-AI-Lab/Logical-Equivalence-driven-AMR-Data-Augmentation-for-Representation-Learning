from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
##### model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer('microsoft/mpnet-base')
# model = SentenceTransformer('nghuyong/ernie-2.0-en')
# model = SentenceTransformer('microsoft/deberta-base')
# model = SentenceTransformer('roberta-base')
# model = SentenceTransformer('t5-base')
# model = SentenceTransformer('chitanda/merit-roberta-large-v1')
# model = SentenceTransformer('chitanda/merit-albert-v2-xxlarge-v1')
# model = SentenceTransformer('chitanda/merit-deberta-v2-xlarge-v1')
# model = SentenceTransformer('chitanda/merit-deberta-v2-xxlarge-v1')


# model = SentenceTransformer('./LReasoner_official_model/roberta_model/')
# model = SentenceTransformer('./LReasoner_official_model/albert_model/albert_cp_cts19_extn5_len_352_rr_s42_ac1_rev/')
# model = SentenceTransformer('nli-roberta-base')
model = SentenceTransformer('./BERT/Transformers/roberta-large')

# df=pd.read_csv("./synthetic_logical_equivalence_inequivalence_data_pair.csv")
# df=pd.read_csv("./synthetic_logical_equivalence_sentence_pair_testset_change_name.csv")
df=pd.read_csv("./synthetic_logical_equivalence_sentence_pair_testset.csv")
data = []
df_output = pd.DataFrame(data,columns=['ID','Sentence1','Sentence2', 'Label', 'Tag', 'Prediction'])

# query_embedding = model.encode('All young people are cold.')
# passage_embedding = model.encode(['No young people are not cold.'])

# query_embedding = model.encode('Damien agrees to go golfing.')
# passage_embedding = model.encode(['Damien does agree to go golfing.'])

# query_embedding = model.encode('If Anne is green then Anne is blue.')
# passage_embedding = model.encode(['if Anne is not blue then Anne is not green.'])

# query_embedding = model.encode('If someone is rough and nice then they are green.')
# passage_embedding = model.encode(['If someone is rough and nice then they are not green.'])
total_count = 0
for index, row in df.iterrows():
    query_embedding = model.encode(row['Sentence1'])
    passage_embedding = model.encode([row['Sentence2']])

    # print("Similarity:", util.cos_sim(query_embedding, passage_embedding))
    similarity = util.cos_sim(query_embedding, passage_embedding)
    if row['Label'] == 1:
        if similarity.item() > 0.5:
            total_count = total_count + 1
    else:
        if similarity.item() < 0.5:
            total_count = total_count + 1
    df_output = df_output.append(
        {'ID': row['ID'],
         'Sentence1': row['Sentence1'],
         'Sentence2': row['Sentence2'],
         'Label': row['Label'],
         'Tag': row['Tag'],
         'Prediction': similarity.item()}, ignore_index=True)

# df_output.to_csv("./lreasoner_roberta_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./lreasoner_albert_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./mpnet_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./ernie-2.0_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./deberta_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./roberta_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./t5_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./merit_roberta_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./merit_albert_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./merit_deberta_xlarge_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./merit_deberta_xxlarge_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
# df_output.to_csv("./roberta_nli_base_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
df_output.to_csv("./roberta_logical_equivalence_pretrained_synthetic_logical_equivalence_inequivalence_prediction.csv",index = None,encoding = 'utf8')
print("The total accuracy is: ", total_count/df.shape[0])

