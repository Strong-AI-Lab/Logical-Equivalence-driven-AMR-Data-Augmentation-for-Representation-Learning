# -*- coding: utf-8 -*-
"""
Created on Wed March 3 2021

@author: Qiming Bao

Depth=3 new data generation for non negation rule animal

"""

import json
import itertools
import random

animal_name = ['the bald eagle', 'the tiger', 'the bear', 'the lion', 'the wolf', 'the crocodile', 'the dinosaur',
               'the snake', 'the leopard']
animal_name_1 = ['the cat', 'the dog', 'the mouse', 'the rabbit', 'the squirrel']
animal_relations = ['is', 'is not']
animal_relations_1 = ['likes', 'chases', 'needs', 'visits', 'attacks', 'sees']
# animal_relations_1_1 = {'does not like', 'does not chase', 'does not need', 'does not visit', 'does not eat'}
animal_attributes_1 = ['kind', 'quiet', 'round', 'nice', 'smart']
animal_attributes_2 = ['dull', 'rough', 'lazy', 'slow', 'sleepy', 'tired']

animal_attributes_3 = ['big', 'strong', 'awful', 'fierce', 'heavy', 'obese']
animal_attributes_4 = ['furry', 'small', 'cute', 'lovely', 'beautiful', 'adorable']

id = 0
whole_dict = []
item = list(itertools.permutations(animal_name, 4))
for index in range(0, len(item)):
    id += 1
    random.shuffle(animal_name)
    animal = item[index][0]
    animal_1 = item[index][1]

    random.shuffle(animal_name_1)
    animal_2 = animal_name_1[0]
    animal_3 = animal_name_1[1]

    random.shuffle(animal_relations_1)
    random.shuffle(animal_attributes_1)
    random.shuffle(animal_attributes_2)
    random.shuffle(animal_attributes_3)
    random.shuffle(animal_attributes_4)

    context = [animal + " " + animal_relations[0] + " " + animal_attributes_2[0] + ". " +
               animal + " " + animal_relations[0] + " " + animal_attributes_2[1] + ". " +
               animal + " " + animal_relations[0] + " " + animal_attributes_2[2] + ". " +
               animal + " " + animal_relations_1[0] + " " + animal_2 + ". " +
               animal_1 + " " + animal_relations_1[2] + " " + animal_3 + ". " +
               animal_1 + " " + animal_relations[0] + " " + animal_attributes_3[0] + ". " +
               animal_1 + " " + animal_relations[0] + " " + animal_attributes_3[1] + ". " +
               animal_2 + " " + animal_relations[0] + " " + animal_attributes_1[0] + ". " +
               animal_2 + " " + animal_relations[0] + " " + animal_attributes_1[1] + ". " +
               animal_2 + " " + animal_relations[0] + " " + animal_attributes_1[2] + ". " +
               animal_3 + " " + animal_relations[0] + " " + animal_attributes_4[0] + ". " +
               animal_3 + " " + animal_relations[0] + " " + animal_attributes_4[1] + ". " +
               animal_3 + " " + animal_relations[0] + " " + animal_attributes_4[2] + ". " +
               animal_attributes_1[0].capitalize() + " animals are " + animal_attributes_4[0] + ". " +
               "If something is " + animal_attributes_2[1] + " then it " + animal_relations_1[
                   1] + " " + animal_2 + ". " +
               "If something " + animal_relations_1[1] + " " + animal_2 + " then it is " + animal_attributes_2[
                   4] + ". " +
               "If something is " + animal_attributes_2[0] + " and " + animal_attributes_2[1] + " then it is " +
               animal_attributes_2[2] + ". " +
               "If something is " + animal_attributes_4[0] + " and " + animal_attributes_4[1] + " then it is " +
               animal_attributes_4[3] + ". " +
               "If something is " + animal_attributes_3[0] + " and " + animal_attributes_3[1] + " then it is " +
               animal_attributes_3[3] + ". " +
               "If something is " + animal_attributes_2[2] + " then it is " + animal_attributes_2[3] + ". " +
               "All " + animal_attributes_2[3] + " animals are " + animal_attributes_2[5] + ". " +
               "If something is " + animal_attributes_4[0] + " then it is " + animal_attributes_4[1] + ". " +
               "All " + animal_attributes_4[1] + " animals are " + animal_attributes_4[2] + ". " +
               "If something is " + animal_attributes_3[3] + " then it is " + animal_attributes_3[4] + ". " +
               "All " + animal_attributes_3[4] + " animals are " + animal_attributes_3[5] + ". " +
               "If something is " + animal_attributes_4[3] + " then it is " + animal_attributes_4[4] + ". " +
               "All " + animal_attributes_4[4] + " animals are " + animal_attributes_4[5] + ". " +
               "All " + animal_attributes_2[4] + " animals are " + animal_attributes_3[2] + "."
               ]
    question0 = animal_2 + " " + animal_relations[0] + " " + animal_attributes_4[2] + "."
    label0 = "true"
    QDep0 = "3"
    question0_0 = animal_2 + " " + animal_relations[1] + " " + animal_attributes_4[2] + "."
    label0_0 = "false"
    QDep0_0 = "3"
    question1 = animal + " " + animal_relations[0] + " " + animal_attributes_2[5] + "."
    label1 = "true"
    QDep1 = "3"
    question1_1 = animal + " " + animal_relations[1] + " " + animal_attributes_2[5] + "."
    label1_1 = "false"
    QDep1_1 = "3"
    question2 = animal_1 + " " + animal_relations[0] + " " + animal_attributes_3[5] + "."
    label2 = "true"
    QDep2 = "3"
    question2_2 = animal_1 + " " + animal_relations[1] + " " + animal_attributes_3[5] + "."
    label2_2 = "false"
    QDep2_2 = "3"
    question3 = animal_3 + " " + animal_relations[0] + " " + animal_attributes_4[5] + "."
    label3 = "true"
    QDep3 = "3"
    question3_3 = animal_3 + " " + animal_relations[1] + " " + animal_attributes_4[5] + "."
    label3_3 = "false"
    QDep3_3 = "3"
    question4 = animal + " " + animal_relations[0] + " " + animal_attributes_3[2] + "."
    label4 = "true"
    QDep4 = "3"
    question4_4 = animal + " " + animal_relations[1] + " " + animal_attributes_3[2] + "."
    label4_4 = "false"
    QDep4_4 = "3"

    test_dict = {
        'id': "NonNegationRule-Animal-D3-" + str(id),
        'context': context[0],
        'questions': [{
            'id': "NonNegationRule-Animal-D3-" + str(id) + "1",
            'text': question0,
            'label': label0,
            'meta': {
                "QDep": QDep0,
                "QCat": "0"
            }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "2",
             'text': question0_0,
             'label': label0_0,
             'meta': {
                 "QDep": QDep0_0,
                 "QCat": "0_0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "3",
             'text': question1,
             'label': label1,
             'meta': {
                 "QDep": QDep1,
                 "QCat": "0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "4",
             'text': question1_1,
             'label': label1_1,
             'meta': {
                 "QDep": QDep1_1,
                 "QCat": "0_0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "5",
             'text': question2,
             'label': label2,
             'meta': {
                 "QDep": QDep2,
                 "QCat": "0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "6",
             'text': question2_2,
             'label': label2_2,
             'meta': {
                 "QDep": QDep2_2,
                 "QCat": "0_0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "7",
             'text': question3,
             'label': label3,
             'meta': {
                 "QDep": QDep3,
                 "QCat": "0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "8",
             'text': question3_3,
             'label': label3_3,
             'meta': {
                 "QDep": QDep3_3,
                 "QCat": "0_0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "9",
             'text': question4,
             'label': label4,
             'meta': {
                 "QDep": QDep4,
                 "QCat": "0"
             }},
            {'id': "NonNegationRule-Animal-D3-" + str(id) + "10",
             'text': question4_4,
             'label': label4_4,
             'meta': {
                 "QDep": QDep4_4,
                 "QCat": "0_0"
             }}
        ]
    }

    whole_dict.append(test_dict)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


with open('NonNegationRule-Animal-D3.jsonl', 'w') as f:
    for index in whole_dict:
        json.dump(index, f, default=set_default)
        f.write('\n')
