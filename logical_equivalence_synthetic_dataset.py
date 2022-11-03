import pandas as pd
import random
import string
punctuation_string = string.punctuation
data = []
data2 = []
df_output = pd.DataFrame(data,columns=['Sentences', 'Logic-words'])
df_output_2 = pd.DataFrame(data2,columns=['Sentences', 'Logic-words'])

subject_list = ['the bald eagle', 'the tiger', 'the bear', 'the lion', 'the wolf', 'the crocodile', 'the dinosaur', 'the snake', 'the leopard',
                'the cat', 'the dog', 'the mouse', 'the rabbit', 'the squirrel',
                'Anne', 'Alan', 'Bob', 'Charlie', 'Dave', 'Erin', 'Harry', 'Gary', 'Fiona']
verb_list = ['is']
adjective_list = ['kind', 'quiet', 'round', 'nice', 'smart', 'clever',
                  'dull', 'rough', 'lazy', 'slow', 'sleepy', 'boring', 'tired', 'reckless',
                  'furry', 'small', 'cute', 'lovely', 'beautiful', 'funny',
                  'big', 'strong', 'awful', 'fierce', 'heavy', 'horrible', 'powerful', 'angry',
                  'high', 'huge',
                  'short', 'thin', 'little', 'tiny',
                  'wealthy', 'poor', 'dull', 'rough', 'bad', 'sad']


for subject in subject_list:
    for verb in verb_list:
        for adjective in adjective_list:
            temp_sentence = subject+" "+verb+" "+adjective + "."
            df_output_2 = df_output_2.append({'Sentences': temp_sentence,
                 'Logic-words': 'None'}, ignore_index=True)
            df_output = df_output.append(
                {'Sentences': temp_sentence,
                 'Logic-words': 'positive'}, ignore_index=True)
            temp_sentence_negation = subject + " " + verb + " not " + adjective + "."
            df_output = df_output.append(
                {'Sentences': temp_sentence_negation,
                 'Logic-words': 'negative'}, ignore_index=True)


df_output_if_then = pd.DataFrame(data,columns=['Sentences', 'Logic-words'])
for index, item in df_output.iterrows():
    sentence1 = df_output.loc[index, 'Sentences']
    for i in punctuation_string:
        sentence1 = sentence1.replace(i, '')
    random_number = random.randint(0, df_output.shape[0]-1)
    while index == random_number:
        random_number = random.randint(0,df_output.shape[0]-1)
    sentence2 = df_output.loc[random_number, 'Sentences']
    temp_sentence = "If " + sentence1 + ", then " + sentence2
    df_output_if_then = df_output_if_then.append(
        {'Sentences': temp_sentence,
         'Logic-words': 'if,then'}, ignore_index=True)

df_output_and_or = pd.DataFrame(data,columns=['Sentences', 'Logic-words'])
for index, item in df_output.iterrows():
    sentence1 = df_output.loc[index, 'Sentences']
    for i in punctuation_string:
        sentence1 = sentence1.replace(i, '')
    random_number = random.randint(0, df_output.shape[0]-1)
    while index == random_number:
        random_number = random.randint(0,df_output.shape[0]-1)
    sentence2 = df_output.loc[random_number, 'Sentences']
    temp_sentence = sentence1 + " and " + sentence2
    df_output_and_or = df_output_and_or.append(
        {'Sentences': temp_sentence,
         'Logic-words': 'and'}, ignore_index=True)

    if "not" in sentence1 and "not" not in sentence2:
        temp_sentence = sentence2 + " or " + sentence1
    else:
        temp_sentence = sentence1 + " or " + sentence2
    df_output_and_or = df_output_and_or.append(
        {'Sentences': temp_sentence,
         'Logic-words': 'or'}, ignore_index=True)

df_output = df_output.append(df_output_if_then)
df_output = df_output.append(df_output_and_or)

df_output.to_csv("./output_result/synthetic_sentences.csv",index = None,encoding = 'utf8')
df_output_2.to_csv("./output_result/synthetic_single_no_logic_words_sentences.csv",index = None,encoding = 'utf8')