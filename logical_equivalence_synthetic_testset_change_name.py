import pandas as pd
import random
from nltk.corpus import wordnet
from itertools import combinations, permutations
data = []
df_output = pd.DataFrame(data,columns=['ID','Sentence1','Sentence2', 'Label', 'Tag'])

subject_list = ['the sheep', 'the kitten', 'the Garfield', 'the lion', 'the goat', 'the bull', 'the cow',
                'the elephant', 'the butterfly', 'the fish',
                'Peter', 'Bill', 'Tom', 'Amy', 'Charles', 'Tim', 'Lucy', 'John']
verb_list = ['is']
adjective_list = ['kind', 'quiet', 'round', 'nice', 'smart', 'clever',
                  'dull', 'rough', 'lazy', 'slow', 'sleepy', 'boring', 'tired', 'reckless',
                  'furry', 'small', 'cute', 'lovely', 'beautiful', 'funny',
                  'big', 'strong', 'awful', 'fierce', 'heavy', 'horrible', 'powerful', 'angry',
                  'high', 'huge',
                  'short', 'thin', 'little', 'tiny',
                  'wealthy', 'poor', 'dull', 'rough', 'bad', 'sad']

def double_negation(df_output, id, sub, verb, adj):
    ##Logical equivalence example
    s1 = sub + " " + verb + " " + adj + "."
    syn = wordnet.synsets(adj)[0]
    good = wordnet.synset(syn.name())
    antonym = good.lemmas()[0].antonyms()
    if len(antonym) > 0:
        s2 = sub + " " + verb + " not " + antonym[0].name() + "."
        df_output = df_output.append(
            {'ID': id,
             'Sentence1': s1,
             'Sentence2': s2,
             'Label': 1,
             'Tag': "Double Negation"}, ignore_index=True)

    s3 = sub + " " + verb + " not " + adj + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Double Negation"}, ignore_index=True)

    return df_output

def commutative(df_output, id, sub1, verb1, adj1, sub2, verb2, adj2):
    ##Logical equivalence example
    s1 = sub1 + " " + verb1 + " " + adj1
    s2 = sub2 + " " + verb2 + " " + adj2
    s3 = sub1 + " " + verb1 + " not " + adj1

    concat_sen = s1 + " and " + s2 + "."
    concat_sen2 = s2 + " and " + s1 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': concat_sen,
         'Sentence2': concat_sen2,
         'Label': 1,
         'Tag': "Commutative"}, ignore_index=True)

    concat_sen3 = s1 + " and " + s3 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': concat_sen,
         'Sentence2': concat_sen3,
         'Label': 0,
         'Tag': "Commutative"}, ignore_index=True)

    return df_output

def contraposition(df_output, id, sub1, sub2, verb, adj1, adj2):
    ##Logical equivalence example
    s1 = "If " + sub1 + " " + verb + " " + adj1 + ", then " + sub2 + " " + verb + " " + adj2 + "."
    s2 = "If " + sub2 + " " + verb + " not " + adj2 + ", then " + sub1 + " " + verb + " not " + adj1 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s2,
         'Label': 1,
         'Tag': "Contraposition"}, ignore_index=True)
    s3 = "If " + sub1 + " " + verb + " " + adj1 + ", then " + sub2 + " " + verb + " not " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Contraposition"}, ignore_index=True)

    s1 = "If " + sub1 + " " + verb + " " + adj1 + ", then " + sub2 + " " + verb + " not " + adj2 + "."
    s2 = "If " + sub2 + " " + verb + " " + adj2 + ", then " + sub1 + " " + verb + " not " + adj1 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s2,
         'Label': 1,
         'Tag': "Contraposition"}, ignore_index=True)
    s3 = "If " + sub1 + " " + verb + " " + adj1 + ", then " + sub2 + " " + verb + " " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Contraposition"}, ignore_index=True)

    s1 = "If " + sub1 + " " + verb + " not " + adj1 + ", then " + sub2 + " " + verb + " " + adj2 + "."
    s2 = "If " + sub2 + " " + verb + " not " + adj2 + ", then " + sub1 + " " + verb + " " + adj1 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s2,
         'Label': 1,
         'Tag': "Contraposition"}, ignore_index=True)
    s3 = "If " + sub1 + " " + verb + " not " + adj1 + ", then " + sub2 + " " + verb + " not " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Contraposition"}, ignore_index=True)

    s1 = "If " + sub1 + " " + verb + " not " + adj1 + ", then " + sub2 + " " + verb + " not " + adj2 + "."
    s2 = "If " + sub2 + " " + verb + " not " + adj2 + ", then " + sub1 + " " + verb + " not " + adj1 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s2,
         'Label': 1,
         'Tag': "Contraposition"}, ignore_index=True)
    s3 = "If " + sub1 + " " + verb + " not " + adj1 + ", then " + sub2 + " " + verb + " " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Contraposition"}, ignore_index=True)

    return df_output


def implication(df_output, id, sub1, sub2, verb, adj1, adj2):
    ##Logical equivalence example
    s1 = "If " + sub1 + " " + verb + " " + adj1 + ", then " + sub2 + " " + verb + " " + adj2 + "."
    s2 = sub1 + " " + verb + " not " + adj1 + " or " + sub2 + " " + verb + " " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s2,
         'Label': 1,
         'Tag': "Contraposition"}, ignore_index=True)
    s3 = sub1 + " " + verb + " not " + adj1 + " or " + sub2 + " " + verb + " not " + adj2 + "."
    df_output = df_output.append(
        {'ID': id,
         'Sentence1': s1,
         'Sentence2': s3,
         'Label': 0,
         'Tag': "Contraposition"}, ignore_index=True)

    return df_output

id = 0

for i in range(100):
    random_number_1 = random.randint(0, len(subject_list) - 1)
    random_number_2 = random.randint(0, len(adjective_list) - 1)
    subject_1 = subject_list[random_number_1]
    adjective_1 = adjective_list[random_number_1]
    df_output = double_negation(df_output, id, subject_1, verb_list[0], adjective_1)
    id = id + 1

for i in range(100):
    random_number_1 = random.randint(0, len(subject_list) - 1)
    random_number_2 = random.randint(0, len(adjective_list) - 1)
    subject_1 = subject_list[random_number_1]
    adjective_1 = adjective_list[random_number_1]
    random_number_3 = random.randint(0, len(subject_list) - 1)
    random_number_4 = random.randint(0, len(adjective_list) - 1)
    while random_number_3 == random_number_1:
        random_number_3 = random.randint(0, len(subject_list) - 1)
    while random_number_4 == random_number_2:
        random_number_4 = random.randint(0, len(adjective_list) - 1)
    subject_3 = subject_list[random_number_3]
    adjective_4 = adjective_list[random_number_4]
    df_output = commutative(df_output, id, subject_1, verb_list[0], adjective_1, subject_3, verb_list[0], adjective_4)
    id = id + 1

for i in range(100):
    random_number_1 = random.randint(0, len(subject_list) - 1)
    random_number_2 = random.randint(0, len(subject_list) - 1)
    while random_number_1 == random_number_2:
        random_number_2 = random.randint(0, len(subject_list) - 1)
    subject_1 = subject_list[random_number_1]
    subject_2 = subject_list[random_number_2]

    random_number_1 = random.randint(0, len(adjective_list) - 1)
    random_number_2 = random.randint(0, len(adjective_list) - 1)
    adjective_1 = adjective_list[random_number_1]
    adjective_2 = adjective_list[random_number_2]

    ## comtraposition logical equivalence/inequivalence
    df_output = contraposition(df_output, id, subject_1, subject_2, verb_list[0], adjective_1, adjective_2)
    id = id + 1

for i in range(100):
    random_number_1 = random.randint(0, len(subject_list) - 1)
    random_number_2 = random.randint(0, len(subject_list) - 1)
    while random_number_1 == random_number_2:
        random_number_2 = random.randint(0, len(subject_list) - 1)
    subject_1 = subject_list[random_number_1]
    subject_2 = subject_list[random_number_2]

    random_number_1 = random.randint(0, len(adjective_list) - 1)
    random_number_2 = random.randint(0, len(adjective_list) - 1)
    adjective_1 = adjective_list[random_number_1]
    adjective_2 = adjective_list[random_number_2]

    ## comtraposition logical equivalence/inequivalence
    df_output = implication(df_output, id, subject_1, subject_2, verb_list[0], adjective_1, adjective_2)
    id = id + 1

df_output.to_csv("./synthetic_logical_equivalence_sentence_pair_testset_change_name.csv",index = None,encoding = 'utf8')