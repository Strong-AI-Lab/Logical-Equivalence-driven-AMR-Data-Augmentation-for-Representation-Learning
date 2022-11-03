import amrlib
from amrlib.models.parse_xfm.inference import Inference
import penman
import json
import pandas as pd
# import t5.evaluation.metrics as t5
# from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import penman
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

def swappable_conditions(g):
    for inst in g.instances():
        if (len(g.edges(source=inst.source, role=':ARG0')) == 1  ## this if is working for contraposition law
         and len(g.edges(source=inst.source, role=':ARG1')) == 1
         and len(g.edges(source=inst.source, role=':condition')) == 1):
            yield inst.source

# graphs = ["(p / possible-01            :ARG1 (u / use-01                  :ARG0 y                  :ARG1 (c / computer))) :condition(h / have-03      :ARG0 (y / you)      :ARG1 (s / skill            :topic (k / keyboard)))"]
# graphs = ["'# ::snt I am happy today.(h / happy-01      :ARG1 (ii / i)      :time (t / today)	  :polarity -	  :polarity -)'"]
# graphs = ["'(h / unhappy-01      :ARG1 (ii / i)      :time (t / today)	  :polarity -)'"]
# graphs = ["(p / possible-01      :ARG1 (f / fly-01            :ARG0 (b / bird                  :quant (s / all))))"]
# graphs = ["(h / have-03      :ARG0 (y / you)      :ARG1 (s / skill            :topic (k / keyboard))      :condition (p / possible-01            :ARG1 (u / use-01                  :ARG0 y                  :ARG1 (c / computer))))"]
# graphs = ["(p / possible-01:ARG1 (u / use-01                          :ARG0 y                         :ARG1 (c / computer)):condition (h / have-03   :ARG0 (y / you)   :ARG1 (s / skill            :topic (k / keyboard))))"]
graphs = ["(h / happy-01:ARG1 (ii / i             :polarity -))"]
return_list = []
# for graph in graphs:
#     g = penman.decode(graph)
#     if len(list(swappable_conditions(g))) != 0:
#         swap_condition = list(swappable_conditions(g))
#         z0 = swap_condition[0]
#         z_arg0 = g.edges(source=z0, role=':ARG0')[0].target
#         z_arg1 = g.edges(source=z0, role=':ARG1')[0].target
#         z_condition = g.edges(source=z0, role=':condition')[0].target
#         g.triples.remove((z0, ':ARG0', z_arg0))  # remove the triples
#         g.triples.remove((z0, ':ARG1', z_arg1))
#         g.triples.remove((z0, ':condition', z_condition))
#         g.epidata.clear()
#         g.triples.insert(0,(z_condition, ':condition', z0))
#         g.triples.insert(1,(z0, ':ARG0', z_arg0))
#         g.triples.insert(2,(z0, ':ARG1', z_arg1))
#         new_graph = penman.encode(g)
#         return_list.append(new_graph)


## To convert graphs to sentences
gtos = amrlib.load_gtos_model("./pretrained_models/model_generate_t5wtense-v0_1_0")
sents, _ = gtos.generate(return_list)

now = int(round(time.time()*1000))
now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
print("The end time is: ",now02)