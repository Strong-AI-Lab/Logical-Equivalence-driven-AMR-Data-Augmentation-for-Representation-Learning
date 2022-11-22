import math
import string

import sacrebleu
import t5.evaluation.metrics as t5
import metrics as metrics
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


def get_bleu_result(candidates, label):
    # BLEU score
    max_bleu_score = 0.0
    best_index = 0
    for i in range(len(candidates)):
        # BLEU
        bleu_analysis = bleu([label], [candidates[i]])
        if max_bleu_score < bleu_analysis['bleu']:
            max_bleu_score = bleu_analysis['bleu']
            best_index = i

    return {"bleu_pred": candidates[best_index], "bleu_score": round(max_bleu_score,3)}


# ROUGE score
def get_rouge_result(candidates, label):

    max_rouge_score = 0.0
    best_rouge_index = 0
    for i in range(len(candidates)):
        # ROUGE
        rouge_analysis = metrics.rouge([label], [candidates[i]])
        rouge_score = rouge_analysis['rougeLsum']
        if max_rouge_score < rouge_score:
            max_rouge_score = rouge_score
            best_rouge_index = i
    return {"rouge_pred": candidates[best_rouge_index], "rouge_score": round(max_rouge_score,3)}


def get_readability_result(predictions, target_text):
    target_readability = metrics.get_readability(target_text[0])['flesch_kincaid']
    best_readability_index = 0
    best_readability = 0.0
    min_delta = math.inf
    for i in range(len(predictions)):
        current_readability = metrics.get_readability(predictions[i])['flesch_kincaid']
        current_delta = abs(current_readability - target_readability)
        if (current_delta < min_delta):
            best_readability = current_readability
            best_readability_index = i
    return {"readability_pred": predictions[best_readability_index], "readability_score": round(best_readability, 3)}


def get_bert_score_result(candidates, target_text):
    max_bert_score = 0.0
    best_index = 0
    print(candidates)
    print(target_text)
    for i in range(len(candidates)):
        score = metrics.bertscore([candidates[i]], [target_text])
        print(score)
        score = round(sum(score) / len(score), 3)
        if max_bert_score < score:
            max_bert_score = score
            best_index = i

    return {"bert-score_pred": candidates[best_index], "bert-score": round(max_bert_score, 3)}

def get_grammatical_error(predictions):

    lowest_grammatical_error_index = 0

    lowest_grammatical_error = 0.0

    for i in range(len(predictions)):

        score = metrics.gramm_err_rate(predictions[i])

        if lowest_grammatical_error > score:

            lowest_grammatical_error = score

            lowest_grammatical_error_index = i

    return {"lowest_gramm_err_pred": predictions[lowest_grammatical_error_index],
            "grammatical_error_rate": round(lowest_grammatical_error, 3)}


def corpus_score(input_text, target_text, predictions, text2data=False):
    results = {"input": input_text, "label": target_text}
    bleu_pred = get_bleu_result(predictions, target_text)
    results.update(bleu_pred)
    results.update(get_rouge_result(predictions, target_text))
    # results.update(get_bert_score_result(predictions, target_text))
    results.update(get_readability_result(predictions, target_text))
    # results.update(get_grammatical_error(predictions))

    return results


def corpus_score_total(input_text, target_text, predictions):
    total_list = []

    for i in range(len(predictions)):
        # BLEU
        bleu_analysis = t5.bleu(target_text, [predictions[i]])
        # Rouge
        rouge_analysis = metrics.rouge(target_text, [predictions[i]])
        rouge_score = rouge_analysis['rougeLsum']

        total_list.append({"input": input_text, "label": target_text,
         "bleu_pred": predictions[i], "bleu_score": round(bleu_analysis['bleu'],3),
         "rouge_pred": predictions[i], "rouge_score": round(rouge_score,3)})

    return total_list


def get_corpus_bertscore(corpus_bert_score):
    return {"bert_score": corpus_bert_score}


def batch_score(inputs, targets, predictions, text2data=False):
    results = []
    for i in range(len(predictions)):
        results += [corpus_score(inputs[i], targets[i], predictions[i], text2data=text2data)]
    generation = []
    for res in results:
        generation.append(res['bleu_pred'])
    corpus_bert_score = metrics.corpus_bertscore(generation, targets)
    for inp in range(len(results)):
        results[inp].update(get_corpus_bertscore(round(float(corpus_bert_score[inp]), 2)))
    return results


def batch_score_total(inputs, targets, predictions):
    results = []
    for i in range(len(predictions)):
        json_list = corpus_score_total(inputs[i], targets[i], predictions[i])
        for j in json_list:
            results += [j]

    return results


def corpus_bert_score_result(inputs, targets, predictions, text2data=False):
    results = []
    for i in range(len(predictions)):
        results += [corpus_score(inputs[i], targets[i], predictions[i], text2data=text2data)]

    return results