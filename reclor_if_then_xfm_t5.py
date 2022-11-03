import amrlib
# from amrlib.models.parse_xfm.inference import Inference
import penman
import json
import pandas as pd
# import t5.evaluation.metrics as t5
# from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
now = int(round(time.time()*1000))
now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
print("The start time is: ",now02)

def bleu(targets, predictions, smooth=1.0):
  """Computes BLEU score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    bleu_score across all targets and predictions
  """

  bleu_score = sacrebleu.sentence_bleu(predictions[0], targets,
                                       smooth_method="exp",
                                       smooth_value=smooth,
                                       lowercase=False,
                                       tokenize="intl")
  return {"bleu": bleu_score.score}

## To convert sentences to graphs
stog = amrlib.load_stog_model("../amrlib/models/model_parse_xfm_bart_large-v0_1_0")
# stog = Inference("./models/model_parse_xfm_bart_large-v0_1_0")
data = []
df = pd.DataFrame(data,columns=['Original_Sentence','Generated_Sentence','BLEU_Score'])
whole_dict = []

file = open("../amrlib/extracted_data_if_then/reclor.csv", 'r', encoding='utf-8')
sentence_list = []
dataframe = pd.read_csv("../amrlib/extracted_data_if_then/reclor.csv")
for index, row in dataframe.iterrows():
    sentence_list.append(row['Sentences'])

graphs = stog.parse_sents(sentence_list)

## To convert graphs to sentences
gtos = amrlib.load_gtos_model("../amrlib/models/model_generate_t5-v0_1_0")
sents, _ = gtos.generate(graphs)
for sent_id in range(len(sents)):
    bleu_score = bleu([sentence_list[sent_id]], [sents[sent_id]])
    df = df.append({'Original_Sentence': sentence_list[sent_id], 'Generated_Sentence': sents[sent_id], 'BLEU_Score': bleu_score['bleu']},ignore_index=True)

df.to_csv("../amrlib/extracted_data_if_then/reclor_if_then_xfm_t5_bleu_list.csv",index = None,encoding = 'utf8')

now = int(round(time.time()*1000))
now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
print("The end time is: ",now02)