from readability.readability import Readability
import language_tool_python
grammar_tool = language_tool_python.LanguageToolPublicAPI('es')
from rouge_score import rouge_scorer
from rouge_score import scoring

import nltk
nltk.download('punkt')

import bert_score
# from fast_bleu import BLEU, SelfBLEU
import numpy

def bertscore(generation, ref):
    [precision_s, recall_s, f1_s] = bert_score.score(cands=generation, refs=ref, lang='en')
    return [round(float(num), 2) for num in f1_s]


def corpus_bertscore(generation, ref):
    [precision_s, recall_s, f1_s] = bert_score.score(cands=generation, refs=ref, lang='en')
    return f1_s


# def self_bleu_score(generation):
#     self_bleu = SelfBLEU(generation)
#     result = self_bleu.get_score()
#     return round(float(numpy.mean(result[4])), 2)

def gramm_err_rate(generation):
    matches = grammar_tool.check(generation)
    if len(generation.split()) < 1:
        return -1
    else:
        error_rate = round((len(matches) / len(generation.split())) * 100, 2)
        return error_rate


def get_readability(generation):
    if len(generation.split(' ')) < 150:
        generation = generation * (int(150 / len(generation.split(' '))) + 1)
    try:
        r = Readability(generation)

        f = r.flesch()
        fk = r.flesch_kincaid()
        gf = r.gunning_fog()
        cl = r.coleman_liau()
        lw = r.linsear_write()
        ari = r.ari()
        s = r.spache()
        dc = r.dale_chall()

        readability_result = {
            "flesch": round(f.score, 2),
            "flesch_kincaid": round(fk.score, 2),
            "gunning_fog": round(gf.score, 2),
            "coleman_liau": round(cl.score, 2),
            "linsear_write": round(lw.score, 2),
            "ari": round(ari.score, 2),
            "spache": round(s.score, 2),
            "dale_chall": round(dc.score, 2)
        }

        return readability_result
    except:
        readability_result = {
            "flesch": 0.0,
            "flesch_kincaid": 0.0,
            "gunning_fog": 0.0,
            "coleman_liau": 0.0,
            "linsear_write": 0.0,
            "ari": 0.0,
            "spache": 0.0,
            "dale_chall": 0.0
        }

        return readability_result

"""
This method is a copy of rouge() from https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
except the logging process is explicitly removed
"""
def rouge(targets, predictions, score_keys=None):
    """Computes rouge score.
    Args:
      targets: list of strings
      predictions: list of strings
      score_keys: list of strings with the keys to compute.
    Returns:
      dict with score_key: rouge score across all targets and predictions
    """
    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)
    aggregator = scoring.BootstrapAggregator()

    def _prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(" . ", " .\n")
        return summary

    for prediction, target in zip(predictions, targets):
        target = _prepare_summary(target)
        prediction = _prepare_summary(prediction)
        aggregator.add_scores(scorer.score(target=target, prediction=prediction))
    result = aggregator.aggregate()

    return {key: result[key].mid.fmeasure * 100 for key in score_keys}