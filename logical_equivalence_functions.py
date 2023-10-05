import amrlib
from amrlib.models.parse_xfm.inference import Inference
import penman
import json
import pandas as pd
# import t5.evaluation.metrics as t5
# from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import os
from BERT.bert_config import opt
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
import penman
import time
import re
import copy
from nltk.corpus import wordnet
import string

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

def swappable_conditions_contraposition(g):
    for inst in g.instances():
        if (inst.target == 'have-condition-91'  ## this if is working for contraposition law
         and len(g.edges(source=inst.source, role=':ARG1')) == 1
         and len(g.edges(source=inst.source, role=':ARG2')) == 1):
            yield inst.source

def swappable_conditions_contraposition_negative_sample_root(g):
    for inst in g.instances():
        if (len(g.edges(source=inst.source, role=':condition')) == 1):
            yield inst.source

def swappable_conditions_commutative(g):
    for inst in g.instances():
        if (inst.target == 'and'   ## this elif is working for commutative law
         and len(g.edges(source=inst.source, role=':op1')) == 1
         and len(g.edges(source=inst.source, role=':op2')) == 1):
            yield inst.source

def swappable_conditions_implication(g):
    for inst in g.instances():
        if (inst.target == 'or'   ## this elif is working for implication law
         and len(g.edges(source=inst.source, role=':op1')) == 1
         and len(g.edges(source=inst.source, role=':op2')) == 1):
            yield inst.source

def quantifier_target_extractor(g):
    for inst in g.instances():
        if (len(g.edges(source=inst.source, role=':quant')) == 1):
            yield inst.source

    for inst in g.instances():
        if (len(g.edges(source=inst.source, role=':mod')) == 1):
            yield  inst.source

def swappable_conditions_contraposition_2(g):
    graph = penman.encode(g)
    if ":ARG" in graph and ":condition" in graph:
        start = graph.index("(")
        end = graph.index(":condition")
        # split_new_condition = ":condition "+ graph[start:end] +":polarity -)"
        split_new_condition = ":condition " + graph[start:end]
        # split_old_condition = graph[end+len(":condition "):len(graph)-2] + "\n" +":polarity -" + "\n"
        split_old_condition = graph[end + len(":condition "):len(graph) - 2]
        pattern = r':degree \([^)]+\)'      

        if ":polarity -" not in split_new_condition and ":polarity -" not in split_old_condition:
            split_new_condition = split_new_condition + ":polarity -)"
            split_old_condition = split_old_condition + "\n" +":polarity -" + "\n"
        elif ":polarity -" in split_new_condition and ":polarity -" in split_old_condition:
            split_new_condition = split_new_condition.replace(":polarity -","") + ")"
            split_old_condition = split_old_condition.replace(":polarity -","") + "\n"
            split_new_condition = re.sub(pattern, '', split_new_condition)
            split_old_condition = re.sub(pattern, '', split_old_condition)          
        elif ":polarity -" not in split_new_condition and ":polarity -" in split_old_condition:
            split_new_condition = split_new_condition + ":polarity -)"
            split_old_condition = split_old_condition.replace(":polarity -","") + "\n"
            split_old_condition = re.sub(pattern, '', split_old_condition)          
        elif ":polarity -" in split_new_condition and ":polarity -" not in split_old_condition:
            split_new_condition = split_new_condition.replace(":polarity -","") + ")"
            split_new_condition = re.sub(pattern, '', split_new_condition)
            split_old_condition = split_old_condition + "\n" +":polarity -" + "\n"          

        new_contructed_graph = split_old_condition + split_new_condition + ")"
        # decoded_g = penman.decode(new_contructed_graph)
        # return decoded_g
        return new_contructed_graph

def swappable_conditions_contraposition_negative_sample(g):
    swap_condition = list(swappable_conditions_contraposition_negative_sample_root(g))
    if len(swap_condition) > 0:
        z0 = swap_condition[0]
        z1 = g.edges(source=z0, role=":condition")[0].target
        if (z0, ':polarity', '-') not in g.triples and (z1, ':polarity', '-') not in g.triples:
            g.triples.append((z1, ":polarity", '-'))
        elif (z0, ':polarity', '-') in g.triples and (z1, ':polarity', '-') in g.triples:
            g.triples.remove((z1, ":polarity", '-'))
        elif (z0, ':polarity', '-') not in g.triples and (z1, ':polarity', '-') in g.triples:
            g.triples.remove((z1, ":polarity", '-'))
        elif (z0, ':polarity', '-') in g.triples and (z1, ':polarity', '-') not in g.triples:
            g.triples.append((z1, ":polarity", '-'))

        new_graph = penman.encode(g)
        return new_graph

def swappable_conditions_implication_2(g):
    g_temp = copy.deepcopy(g)
    graph = penman.encode(g_temp)
    if ":ARG" in graph and ":condition" in graph:
        swap_condition = list(swappable_conditions_contraposition_negative_sample_root(g_temp))
        if len(swap_condition) > 0:
            z0 = swap_condition[0]
            z1 = g_temp.edges(source=z0, role=":condition")[0].target
            if (z1, ':polarity', '-') not in g_temp.triples:
                g_temp.triples.append((z1, ":polarity", '-'))
            elif (z1, ':polarity', '-') in g_temp.triples:
                g_temp.triples.remove((z1, ":polarity", '-'))
        graph = penman.encode(g_temp)

        start = graph.index("(")
        end = graph.index(":condition")
        # split_new_condition = ":condition "+ graph[start:end] +":polarity -)"
        split_new_condition = ":op2 " + graph[start:end] + ")"
        # split_old_condition = graph[end+len(":condition "):len(graph)-2] + "\n" +":polarity -" + "\n"
        split_old_condition = ":op1 " + graph[end + len(":condition "):len(graph) - 2] + ")"

        new_contructed_graph = "(root / or \n" + split_old_condition + "\n"+split_new_condition + ")"
        # updated_grapg.triples.append(('root', ':op1', z1))
        # decoded_g = penman.decode(new_contructed_graph)
        # return decoded_g
        return new_contructed_graph

def swappable_conditions_implication_2_negative_samples_generation(g):
    graph = penman.encode(g)
    if ":ARG" in graph and ":condition" in graph:
        start = graph.index("(")
        end = graph.index(":condition")
        # split_new_condition = ":condition "+ graph[start:end] +":polarity -)"
        split_new_condition = ":op2 " + graph[start:end] + ")"
        # split_old_condition = graph[end+len(":condition "):len(graph)-2] + "\n" +":polarity -" + "\n"
        split_old_condition = ":op1 " + graph[end + len(":condition "):len(graph) - 2] + ")"

        new_contructed_graph = "(root / or \n" + split_old_condition + "\n"+split_new_condition + ")"
        # updated_grapg.triples.append(('root', ':op1', z1))
        # decoded_g = penman.decode(new_contructed_graph)
        # return decoded_g
        return new_contructed_graph

def contraposition(graphs, sentence_list, logic_word_list):
    return_list = []
    label_list = []
    sentence_and_tag_list = []
    if graphs is None:
        return
    for index, graph in enumerate(graphs):
        if graph is None:
            continue
        g = penman.decode(graph)
        negative_sample_g = copy.deepcopy(g)
        if len(list(swappable_conditions_contraposition(g))) != 0:
            swap_condition = list(swappable_conditions_contraposition(g))
            z0 = swap_condition[0]
            z1 = g.edges(source=z0, role=':ARG1')[0].target
            z5 = g.edges(source=z0, role=':ARG2')[0].target
            g.triples.remove((z0, ':ARG1', z1))  # remove the triples
            g.triples.remove((z0, ':ARG2', z5))
            g.triples.append((z0, ':ARG1', z5))  # add the replacements
            g.triples.append((z0, ':ARG2', z1))
            if (z1, ':polarity', '-') not in g.triples and (z5, ':polarity', '-') not in g.triples:
                g.triples.append((z1, ':polarity', '-'))  # add polarity -
                g.triples.append((z5, ':polarity', '-'))
                negative_sample_g.triples.append((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') in g.triples and (z5, ':polarity', '-') in g.triples:
                g.triples.remove((z1, ':polarity', '-'))  # add polarity -
                g.triples.remove((z5, ':polarity', '-'))
                negative_sample_g.triples.remove((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') not in g.triples and (z5, ':polarity', '-') in g.triples:
                g.triples.append((z1, ':polarity', '-'))  # add polarity -
                g.triples.remove((z5, ':polarity', '-'))
                negative_sample_g.triples.remove((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') in g.triples and (z5, ':polarity', '-') not in g.triples:
                g.triples.remove((z1, ':polarity', '-'))  # add polarity -
                g.triples.append((z5, ':polarity', '-'))
                negative_sample_g.triples.append((z5, ':polarity', '-'))
            new_graph = penman.encode(g)
            return_list.append(new_graph)
            label_list.append(1)
            sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])

            ## append negative samples
            negative_sample_graph = penman.encode(negative_sample_g)
            return_list.append(negative_sample_graph)
            label_list.append(0)
            sentence_and_tag_list.append([sentence_list[index],logic_word_list[index]])
            # return_list.append(g)
        else:
            return_result = swappable_conditions_contraposition_2(g)
            negative_return_result = swappable_conditions_contraposition_negative_sample(g)
            if return_result is not None:
                return_list.append(return_result)
                label_list.append(1)
                sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])
            if negative_return_result is not None:
                return_list.append(negative_return_result)
                label_list.append(0)
                sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])
    return return_list, label_list, sentence_and_tag_list


def commutative(graphs, sentence_list, logic_word_list):
    return_list = []
    label_list = []
    sentence_and_tag_list = []
    if graphs is None:
        return
    for index, graph in enumerate(graphs):
        if graph is None:
            continue
        g = penman.decode(graph)
        negative_sample_g = copy.deepcopy(g)
        if len(list(swappable_conditions_commutative(g))) != 0:
            swap_condition = list(swappable_conditions_commutative(g))
            z0 = swap_condition[0]
            z1 = g.edges(source=z0, role=':op1')[0].target
            z5 = g.edges(source=z0, role=':op2')[0].target
            g.triples.remove((z0, ':op1', z1))  # remove the triples
            g.triples.remove((z0, ':op2', z5))
            g.triples.append((z0, ':op1', z5))  # add the replacements
            g.triples.append((z0, ':op2', z1))
            new_graph = penman.encode(g)
            return_list.append(new_graph)
            label_list.append(1)
            sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])

            if (z1, ':polarity', '-') not in g.triples and (z5, ':polarity', '-') not in g.triples:
                negative_sample_g.triples.append((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') in g.triples and (z5, ':polarity', '-') in g.triples:
                negative_sample_g.triples.remove((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') not in g.triples and (z5, ':polarity', '-') in g.triples:
                negative_sample_g.triples.remove((z5, ':polarity', '-'))
            elif (z1, ':polarity', '-') in g.triples and (z5, ':polarity', '-') not in g.triples:
                negative_sample_g.triples.append((z5, ':polarity', '-'))

            negative_sample_graph = penman.encode(negative_sample_g)
            return_list.append(negative_sample_graph)
            label_list.append(0)
            sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])

    return return_list, label_list, sentence_and_tag_list

def implication(graphs, sentence_list, logic_word_list):
    return_list = []
    label_list = []
    sentence_and_tag_list = []
    if graphs is None:
        return
    for graph_index, graph in enumerate(graphs):
        if graph is None:
            continue
        g = penman.decode(graph)
        negative_sample_g = copy.deepcopy(g)
        if len(list(swappable_conditions_implication(g))) != 0:
            swap_condition = list(swappable_conditions_implication(g))
            z0 = swap_condition[0]
            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":instance":
                    g.triples[index] = item[:2] + ('have-condition-91',)
                    break
            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":op1":
                    g.triples[index] = item[:1] + (':ARG1',) + item[2:3]
                    break
            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":op2":
                    g.triples[index] = item[:1] + (':ARG2',) + item[2:3]
                    break

            z1 = g.edges(source=z0, role=':ARG1')[0].target
            z5 = g.edges(source=z0, role=':ARG2')[0].target
            g.triples.remove((z0, ':ARG1', z1))
            g.triples.remove((z0, ':ARG2', z5))
            g.triples.append((z0, ':ARG1', z5))
            g.triples.append((z0, ':ARG2', z1))
            if (z1, ':polarity', '-') not in g.triples:
                g.triples.append((z1, ':polarity', '-'))
            elif (z1, ':polarity', '-') in g.triples:
                g.triples.remove((z1, ':polarity', '-'))

            new_graph = penman.encode(g)
            return_list.append(new_graph)
            label_list.append(1)
            sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])

            swap_condition_negative_sample = list(swappable_conditions_implication(negative_sample_g))
            z0_neg = swap_condition_negative_sample[0]
            z1_neg = negative_sample_g.edges(source=z0_neg, role=':op1')[0].target
            z5_neg = negative_sample_g.edges(source=z0_neg, role=':op2')[0].target
            if (z1_neg, ':polarity', '-') in negative_sample_g.triples:
                negative_sample_g.triples.remove((z1_neg, ':polarity', '-'))
            elif (z5_neg, ':polarity', '-') in negative_sample_g.triples:
                negative_sample_g.triples.remove((z5_neg, ':polarity', '-'))
            else:
                negative_sample_g.triples.append((z5_neg, ':polarity', '-'))
            new_negative_graph = penman.encode(negative_sample_g)
            return_list.append(new_negative_graph)
            label_list.append(0)
            sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])

        elif len(list(swappable_conditions_contraposition(g))) != 0:
            swap_condition = list(swappable_conditions_contraposition(g))
            z0 = swap_condition[0]
            z1 = g.edges(source=z0, role=':ARG1')[0].target
            z5 = g.edges(source=z0, role=':ARG2')[0].target
            g.triples.remove((z0, ':ARG1', z1))  # remove the triples
            g.triples.remove((z0, ':ARG2', z5))
            g.triples.append((z0, ':ARG1', z5))  # add the replacements
            g.triples.append((z0, ':ARG2', z1))
            if (z5, ':polarity', '-') not in g.triples:
                g.triples.append((z5, ':polarity', '-'))
            elif (z5, ':polarity', '-') in g.triples:
                g.triples.remove((z5, ':polarity', '-'))

            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":instance":
                    g.triples[index] = item[:2] + ('or',)
                    break
            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":ARG1":
                    g.triples[index] = item[:1] + (':op1',) + item[2:3]
                    break
            for index, item in enumerate(g.triples):
                if item[0] == z0 and item[1] == ":ARG2":
                    g.triples[index] = item[:1] + (':op2',) + item[2:3]
                    break
            new_graph = penman.encode(g)
            return_list.append(new_graph)
            label_list.append(1)
            sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])

            swap_condition_negative_sample = list(swappable_conditions_contraposition(negative_sample_g))
            z0_neg = swap_condition_negative_sample[0]
            z1_neg = negative_sample_g.edges(source=z0_neg, role=':ARG1')[0].target
            z5_neg = negative_sample_g.edges(source=z0_neg, role=':ARG2')[0].target
            if (z1_neg, ':polarity', '-') in negative_sample_g.triples:
                negative_sample_g.triples.remove((z1_neg, ':polarity', '-'))
            elif (z5_neg, ':polarity', '-') in negative_sample_g.triples:
                negative_sample_g.triples.remove((z5_neg, ':polarity', '-'))
            else:
                negative_sample_g.triples.append((z5_neg, ':polarity', '-'))
            new_negative_graph = penman.encode(negative_sample_g)
            return_list.append(new_negative_graph)
            label_list.append(0)
            sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])

        else:
            return_result = swappable_conditions_implication_2(g)
            negative_result = swappable_conditions_implication_2_negative_samples_generation(g)
            if return_result is not None:
                return_list.append(return_result)
                label_list.append(1)
                sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])
            if negative_result is not None:
                return_list.append(negative_result)
                label_list.append(0)
                sentence_and_tag_list.append([sentence_list[graph_index], logic_word_list[graph_index]])
    return return_list, label_list, sentence_and_tag_list

def demorgan(graphs):
    return_list = []
    if graphs is None:
        return
    for graph in graphs:
        if graph is None:
            continue
        g = penman.decode(graph)
        if ":quant" in graph or ":mod" in graph:
            z0 = list(quantifier_target_extractor(g))[0]
            if (z0, ':polarity', '-') in g.triples:
                g.triples.remove((z0, ':polarity', '-'))

            else:
                old_g = copy.deepcopy(g)
                for item in g.triples:
                    if item[1] == ':polarity' and item[2] == '-':
                        g.triples.remove((item))
                        break
                if len(old_g.triples) == len(g.triples):
                    g.triples.append((z0, ':polarity', '-'))
            # temp_graph = penman.encode(g)
            # pattern = re.compile(r':quant|:mod \(([a-z]+)\s*\/\s*([a-z]+)\)')
            # matched_string = re.search(pattern, temp_graph).group()
            # new_graph = ""
            quant_mod_object = []
            for item in g.triples:
                if ":quant" == item[1] or ":mod" == item[1]:
                    quant_mod_object.append(item[2])
                    break

            updated_g = copy.deepcopy(g)
            for item in g.triples:
                if quant_mod_object[0] == item[0] and ":instance" == item[1]:
                    if item[2] == "all":
                        updated_g.triples.remove(item)
                        updated_g.triples.append((quant_mod_object[0],":instance","some"))
                        break
                    elif item[2] == "some":
                        updated_g.triples.remove(item)
                        updated_g.triples.append((quant_mod_object[0], ":instance", "all"))
                        break

            # if "all" in matched_string:
            #     replaced_string = matched_string.replace("all", "some")
            #     new_graph = temp_graph.replace(matched_string,replaced_string)
            # elif "some" in matched_string:
            #     replaced_string = matched_string.replace("some", "all")
            #     new_graph = temp_graph.replace(matched_string, replaced_string)
            # return_list.append(new_graph)
            return_list.append(penman.encode(updated_g))
    return return_list

def double_negation(graphs, sentence_list, logic_word_list):
    return_list = []
    negative_list = []
    label_list = []
    sentence_and_tag_list = []
    return_sents = []
    returned_sentence_and_tag_list = []
    if graphs is None:
        return
    for index, graph in enumerate(graphs):
        if graph is None:
            continue
        g = penman.decode(graph)
        updated_g = copy.deepcopy(g)
        negative_g = copy.deepcopy(g)
        if ":polarity -" not in graph:  ## We only consider the case that the sentence does not have negation.
            # for item in g.triples:
            #     position = 0
            #     if '-' in item[2]:
            #         position = item[2].index('-')
            #     if position > 0:
            #         stem = item[2][0:position]
            #         syn = wordnet.synsets(stem)[0]
            #         good = wordnet.synset(syn.name())
            #         antonym = good.lemmas()[0].antonyms()
            #         if len(antonym) > 0:
            #             if wordnet.synsets(antonym[0].name())[0].pos() == 'a':
            #                 updated_g.triples.remove(item)
            #                 updated_g.triples.append((item[0],item[1],antonym[0].name()+item[2][position:len(item[2])]))
            #                 break
            z0 = updated_g.instances()[0].source
            updated_g.triples.append((z0, ':polarity', '-'))
            temp_graph = penman.encode(updated_g)
            start = temp_graph.index("\n")
            return_list.append(temp_graph[start+1:len(temp_graph)])
            sentence_and_tag_list.append([sentence_list[index], logic_word_list[index]])
            negative_list.append(penman.encode(negative_g))

    if len(return_list) > 0:
        gtos = amrlib.load_gtos_model("./pretrained_models/model_generate_t5wtense-v0_1_0")
        sents, _ = gtos.generate(return_list)
        punctuation_string = string.punctuation

        for idx, sent in enumerate(sents):
            temp_sent = copy.deepcopy(sent)
            for i in punctuation_string:
                temp_sent = temp_sent.replace(i, '')
            splited_sent = temp_sent.split()
            for stem in splited_sent:
                if len(wordnet.synsets(stem)) > 0:
                    syn = wordnet.synsets(stem)[0]
                    good = wordnet.synset(syn.name())
                    antonym = good.lemmas()[0].antonyms()
                    if len(antonym) > 0:
                        if wordnet.synsets(antonym[0].name())[0].pos() == 'a':
                            sent = sent.replace(stem,antonym[0].name())
                            returned_sentence_and_tag_list.append(sentence_and_tag_list[idx])
                            return_sents.append(sent)
                            label_list.append(1)
                            break

        neg_sents, _ = gtos.generate(negative_list)

        for idx, sent in enumerate(neg_sents):
            temp_sent = copy.deepcopy(sent)
            for i in punctuation_string:
                temp_sent = temp_sent.replace(i, '')
            splited_sent = temp_sent.split()
            for stem in splited_sent:
                if len(wordnet.synsets(stem)) > 0:
                    syn = wordnet.synsets(stem)[0]
                    good = wordnet.synset(syn.name())
                    antonym = good.lemmas()[0].antonyms()
                    if len(antonym) > 0:
                        if wordnet.synsets(antonym[0].name())[0].pos() == 'a':
                            sent = sent.replace(stem, antonym[0].name())
                            returned_sentence_and_tag_list.append(sentence_and_tag_list[idx])
                            return_sents.append(sent)
                            label_list.append(0)
                            break

    return return_sents, label_list, returned_sentence_and_tag_list


## To convert sentences to graphs
stog = amrlib.load_stog_model("./pretrained_models/model_parse_xfm_bart_large-v0_1_0")
# stog = Inference("./models/model_parse_xfm_bart_large-v0_1_0")


### Test sample case
# sentence_list = ["If you can use a computer, then you have no keyboarding skills."]
# sentence_list = ["Some birds can fly."]
# sentence_list = ["All birds can't fly."]
# sentence_list = ["I am happy."]
# sentence_list = ["Xiao Ming is either on his way to school or on his way home."]
# sentence_list = ["If you can use a computer, then you have keyboarding skills.", "If you are happy, then you are funny."]

## Test batch cases
sentence_list = []
logic_word_list = []
dataframe_list = []
dataframe_list_single_sentences = []
flag = "Synthetic" ## BBC, Cosmos, LogiQA, RACE, ReClor, Wikihop, Synthetic
if flag == "BBC":
    dataframe_BBC = pd.read_csv("./extracted_data/BBCs.csv")
    dataframe_list.append(dataframe_BBC)
elif flag == "Cosmos":
    dataframe_Cosmos = pd.read_csv("./extracted_data/Cosmos.csv")
    dataframe_list.append(dataframe_Cosmos)
elif flag == "LogiQA":
    dataframe_Logi = pd.read_csv("./extracted_data/Logi.csv")
    dataframe_list.append(dataframe_Logi)
elif flag == "RACE":
    dataframe_RACE = pd.read_csv("./extracted_data/RACE.csv")
    dataframe_list.append(dataframe_RACE)
elif flag == "ReClor":
    dataframe_ReClor = pd.read_csv("./extracted_data/ReClor.csv")
    dataframe_list.append(dataframe_ReClor)
elif flag == "Wikihop":
    dataframe_Wikihop = pd.read_csv("./extracted_data/Wikihop.csv")
    dataframe_list.append(dataframe_Wikihop)
elif flag == "Synthetic":
    dataframe_synthetic = pd.read_csv("./output_result/synthetic_sentences.csv")
    dataframe_list.append(dataframe_synthetic)
    dataframe_synthetic_2 = pd.read_csv("./output_result/synthetic_single_no_logic_words_sentences.csv")
    dataframe_list_single_sentences.append(dataframe_synthetic_2)

keywords_list = [flag]
data = []
df = pd.DataFrame(data, columns=['Origin', 'Original_Sentence', 'Generated_Sentence', 'BLEU_Score', 'Label', 'Tag',
                                     'logic_words'])
for idx, item in enumerate(dataframe_list_single_sentences):
    for index, row in item.iterrows():
        sentence_list.append(row['Sentences'])
        logic_word_list.append(row['Logic-words'])

    graphs = stog.parse_sents(sentence_list)
    double_negation_list, double_negation_label_list, double_negation_sentence_and_tag_list = [], [], []
    double_negation_list, double_negation_label_list, double_negation_sentence_and_tag_list = double_negation(graphs, sentence_list, logic_word_list)

    ## To convert graphs to sentences
    gtos = amrlib.load_gtos_model("./pretrained_models/model_generate_t5wtense-v0_1_0")
    if len(double_negation_list) > 0:
        # sents, _ = gtos.generate(double_negation_list)
        for sent_id in range(len(double_negation_list)):
            bleu_score = bleu([double_negation_sentence_and_tag_list[sent_id][0]], [double_negation_list[sent_id]])
            df = df.append(
                {'Origin': keywords_list[idx], 'Original_Sentence': double_negation_sentence_and_tag_list[sent_id][0],
                 'Generated_Sentence': double_negation_list[sent_id], 'BLEU_Score': bleu_score['bleu'],
                 'Label': double_negation_label_list[sent_id], 'Tag': "Double negation law",
                 "logic_words": double_negation_sentence_and_tag_list[sent_id][1]}, ignore_index=True)

for idx, item in enumerate(dataframe_list):
    for index, row in item.iterrows():
        sentence_list.append(row['Sentences'])
        logic_word_list.append(row['Logic-words'])

    # sentence_list.append("If Bob is happy, then Alice is funny.")
    # sentence_list.append("Bob is happy or Alice is funny.")
    # sentence_list.append("Fiona is not bad or Harry is bad.")
    # sentence_list.append("If the dinosaur is not high, then Erin is not awful.")
    # sentence_list.append("Dave is beautiful and the snake is tiny.")
    # sentence_list.append("the bald eagle is clever and the wolf is fierce.")
    # sentence_list.append("Most business ethics courses and textbooks confine themselves to considering specific cases and principles.")
    # sentence_list.append("All birds can fly.")
    # sentence_list.append("If Harry is not huge, then the crocodile is dull.")
    # sentence_list.append("If the wolf is not slow, then the lion is not boring.")
    # sentence_list.append("If the cat is not short, then Gary is slow.")
    # sentence_list.append("If Harry is not sleepy, then Harry is cute.")
    # sentence_list.append("If the mouse is dull, then the cat is not dull.")
    # logic_word_list.append(["if,then"])
    # logic_word_list.append(["or"])
    # logic_word_list.append(["and"])
    # logic_word_list.append(["all"])


    graphs = stog.parse_sents(sentence_list)

    ## logical equivalence
    ## contraposition law
    # graphs = ["# ::snt If you can use a computer, you have keyboarding skills.\n(z0 / have-condition-91\n      :ARG1 (z1 / have-03\n            :ARG0 (z2 / you)\n            :ARG1 (z3 / skill\n                  :topic (z4 / keyboard-01)))\n      :ARG2 (z5 / possible-01\n            :ARG1 (z6 / use-01\n                  :ARG0 z2\n                  :ARG1 (z7 / computer))))"]

    contraposition_list, contraposition_label_list, contraposition_sentence_and_tag_list = [], [], []
    commutative_list, commutative_label_list, commutative_sentence_and_tag_list = [], [], []
    implication_list, implication_label_list, implication_sentence_and_tag_list = [], [], []
    double_negation_list, double_negation_label_list, double_negation_sentence_and_tag_list = [], [], []
    # demorgan_list = []

    contraposition_list, contraposition_label_list, contraposition_sentence_and_tag_list = contraposition(graphs, sentence_list, logic_word_list)
    commutative_list, commutative_label_list, commutative_sentence_and_tag_list = commutative(graphs, sentence_list, logic_word_list)
    implication_list, implication_label_list, implication_sentence_and_tag_list = implication(graphs, sentence_list, logic_word_list)

    # demorgan_list = demorgan(graphs)
    # graph_list = graph_list + commutative_list + demorgan_list

    ## To convert graphs to sentences
    gtos = amrlib.load_gtos_model("./pretrained_models/model_generate_t5wtense-v0_1_0")
    if len(contraposition_list) > 0:
        sents, _ = gtos.generate(contraposition_list)
        for sent_id in range(len(sents)):
            bleu_score = bleu([contraposition_sentence_and_tag_list[sent_id][0]], [sents[sent_id]])
            df = df.append({'Origin':keywords_list[idx],'Original_Sentence': contraposition_sentence_and_tag_list[sent_id][0], 'Generated_Sentence': sents[sent_id], 'BLEU_Score': bleu_score['bleu'], 'Label': contraposition_label_list[sent_id], 'Tag':"Contraposition law", 'logic_words':contraposition_sentence_and_tag_list[sent_id][1]},ignore_index=True)
    if len(commutative_list) > 0:
        sents, _ = gtos.generate(commutative_list)
        for sent_id in range(len(sents)):
            bleu_score = bleu([commutative_sentence_and_tag_list[sent_id][0]], [sents[sent_id]])
            df = df.append({'Origin':keywords_list[idx],'Original_Sentence': commutative_sentence_and_tag_list[sent_id][0], 'Generated_Sentence': sents[sent_id], 'BLEU_Score': bleu_score['bleu'], 'Label': commutative_label_list[sent_id], 'Tag':"Commutative law", 'logic_words':commutative_sentence_and_tag_list[sent_id][1]},ignore_index=True)
    if len(implication_list) > 0:
        sents, _ = gtos.generate(implication_list)
        for sent_id in range(len(sents)):
            bleu_score = bleu([implication_sentence_and_tag_list[sent_id][0]], [sents[sent_id]])
            df = df.append({'Origin':keywords_list[idx],'Original_Sentence': implication_sentence_and_tag_list[sent_id][0], 'Generated_Sentence': sents[sent_id], 'BLEU_Score': bleu_score['bleu'], 'Label': implication_label_list[sent_id], 'Tag':"Implication law", 'logic_words':implication_sentence_and_tag_list[sent_id][1]},ignore_index=True)

    df.to_csv("./"+keywords_list[idx]+"_xfm_t5wtense_logical_equivalence_list.csv",index = None,encoding = 'utf8')

now = int(round(time.time()*1000))
now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
print("The end time is: ",now02)
