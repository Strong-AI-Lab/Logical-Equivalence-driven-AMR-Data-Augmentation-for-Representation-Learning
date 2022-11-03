# -*- coding: utf-8 -*-
"""
Created on Wed March 3 2021
​
@author: Qiming Bao
​
Depth=4 new data generation for negation rule
​
"""

import json
import itertools
import random

people_name = ['Anne', 'Alan', 'Bob', 'Charlie', 'Dave', 'Erin', 'Harry', 'Gary', 'Fiona']
people_relations_1 = ['is', 'is not']

people_attributes_1 = ['big', 'strong', 'high', 'huge']
people_attributes_2 = ['short', 'thin', 'small', 'little', 'tiny']

people_attributes_3 = ['wealthy', 'smart', 'nice', 'quiet', 'kind', 'clever']
people_attributes_4 = ['poor', 'dull', 'rough', 'bad', 'sad']

people_attributes_5 = ['old', 'experienced']
people_attributes_6 = ['young', 'energetic']

id = 0
whole_dict = []
item = list(itertools.permutations(people_name, 4))
for index in range(0, len(item)):
    id += 1
    people = item[index][0]
    people_1 = item[index][1]
    people_2 = item[index][2]
    people_3 = item[index][3]
    random.shuffle(people_attributes_1)
    random.shuffle(people_attributes_2)
    random.shuffle(people_attributes_3)
    random.shuffle(people_attributes_4)
    random.shuffle(people_attributes_5)
    random.shuffle(people_attributes_6)
    context = [people + " " + people_relations_1[0] + " " + people_attributes_1[0] + ". " +
               people + " " + people_relations_1[0] + " " + people_attributes_1[1] + ". " +
               people_1 + " " + people_relations_1[0] + " " + people_attributes_2[0] + ". " +
               people_1 + " " + people_relations_1[0] + " " + people_attributes_2[1] + ". " +
               people_2 + " " + people_relations_1[0] + " " + people_attributes_3[0] + ". " +
               people_3 + " " + people_relations_1[0] + " " + people_attributes_4[0] + ". " +
               people_3 + " " + people_relations_1[0] + " " + people_attributes_4[1] + ". " +
               "If someone is not " + people_attributes_1[2] + " then they are " + people_attributes_4[0] + ". " +
               "If someone is not " + people_attributes_4[2] + " then they are " + people_attributes_3[3] + ". " +
               "If someone is " + people_attributes_3[0] + " then they are " + people_attributes_3[1] + ". " +
               "If someone is " + people_attributes_3[1] + " and not " + people_attributes_4[3] + " then they are " +
               people_attributes_3[2] + ". " +
               "If someone is " + people_attributes_4[0] + " and not " + people_attributes_1[2] + " then they are " +
               people_attributes_4[4] + ". " +
               "If someone is " + people_attributes_4[4] + " then they are " + people_attributes_2[4] + ". " +
               "All " + people_attributes_2[4] + " people are " + people_attributes_2[3] + ". " +
               "If someone is " + people_attributes_2[0] + " and " + people_attributes_2[1] + " then they are " +
               people_attributes_4[3] + ". " +
               "If someone is " + people_attributes_4[3] + " and not " + people_attributes_3[1] + " then they are " +
               people_attributes_4[1] + ". " +
               "If someone is " + people_attributes_4[1] + " then they are " + people_attributes_6[0] + ". " +
               "If someone is " + people_attributes_6[0] + " then they are " + people_attributes_6[1] + ". " +
               "If someone is " + people_attributes_3[3] + " then they are " + people_attributes_3[4] + ". " +
               "If someone is " + people_attributes_3[4] + " then they are " + people_attributes_3[5] + ". " +
               "All " + people_attributes_3[5] + " people are " + people_attributes_3[1] + ". " +
               "If someone is " + people_attributes_3[2] + " then they are " + people_attributes_5[0] + ". " +
               "All " + people_attributes_5[0] + " people are " + people_attributes_5[1] + "."
               ]

    question0 = people + " " + people_relations_1[0] + " " + people_attributes_2[3] + "."
    label0 = "true"
    QDep0 = "4"
    question0_0 = people + " " + people_relations_1[1] + " " + people_attributes_2[3] + "."
    label0_0 = "false"
    QDep0_0 = "4"
    question1 = people_3 + " " + people_relations_1[0] + " " + people_attributes_3[1] + "."
    label1 = "true"
    QDep1 = "4"
    question1_1 = people_3 + " " + people_relations_1[1] + " " + people_attributes_3[1] + "."
    label1_1 = "false"
    QDep1_1 = "4"
    question2 = people_2 + " " + people_relations_1[0] + " " + people_attributes_5[1] + "."
    label2 = "true"
    QDep2 = "4"
    question2_2 = people_2 + " " + people_relations_1[1] + " " + people_attributes_5[1] + "."
    label2_2 = "false"
    QDep2_2 = "4"
    question3 = people_1 + " " + people_relations_1[0] + " " + people_attributes_6[1] + "."
    label3 = "true"
    QDep3 = "4"
    question3_3 = people_1 + " " + people_relations_1[1] + " " + people_attributes_6[1] + "."
    label3_3 = "false"
    QDep3_3 = "4"

    test_dict = {
        'id': "NegationRule-D4-" + str(id),
        'context': context[0],
        'questions': [{
            'id': "NegationRule-D4-" + str(id) + "1",
            'text': question0,
            'label': label0,
            'meta': {
                "QDep": QDep0,
                "QCat": "0_not_notTrue_true_true"
            }},
            {'id': "NegationRule-D4-" + str(id) + "2",
             'text': question0_0,
             'label': label0_0,
             'meta': {
                 "QDep": QDep0_0,
                 "QCat": "0_0_not_notTrue_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "3",
             'text': question1,
             'label': label1,
             'meta': {
                 "QDep": QDep1,
                 "QCat": "0_not_notTrue_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "4",
             'text': question1_1,
             'label': label1_1,
             'meta': {
                 "QDep": QDep1_1,
                 "QCat": "0_0_not_notTrue_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "5",
             'text': question2,
             'label': label2,
             'meta': {
                 "QDep": QDep2,
                 "QCat": "0_true_trueNot_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "6",
             'text': question2_2,
             'label': label2_2,
             'meta': {
                 "QDep": QDep2_2,
                 "QCat": "0_0_true_trueNot_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "7",
             'text': question3,
             'label': label3,
             'meta': {
                 "QDep": QDep3,
                 "QCat": "0_true_trueNot_true_true"
             }},
            {'id': "NegationRule-D4-" + str(id) + "8",
             'text': question3_3,
             'label': label3_3,
             'meta': {
                 "QDep": QDep3_3,
                 "QCat": "0_0_true_trueNot_true_true"
             }}]
    }

    whole_dict.append(test_dict)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


with open('NegationRule-D4.jsonl', 'w') as f:
    for index in whole_dict:
        json.dump(index, f, default=set_default)
        f.write('\n')