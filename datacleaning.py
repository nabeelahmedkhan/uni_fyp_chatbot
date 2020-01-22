# from __future__ import print_function
# from keras.utils.vis_utils import plot_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import re
import nltk
path = "dataset/datasetupdate.csv"
df = pd.read_csv(path)
df.index += 1 
questions = df['questions']
answers = df['answers']
answerstag = df['answerstag']
new_stopwords = ('php','pdf','http','https','pk','edu','admission@smiu.edu.pk','www.smiu.edu.pk','http://www.smiu.edu.pk/admissions.php',
                 'http://cs.smiu.edu.pk/bscs_program.php','http://business.smiu.edu.pk/bba-program.php','http://media.smiu.edu.pk/bs-media-studies-new.php',
                 'http://edu.smiu.edu.pk/bs-education.php','http://env.smiu.edu.pk/bs-environmental-sciences.php','http://business.smiu.edu.pk/accounting-finance.php',
                 'http://business.smiu.edu.pk/mba-program.php','http://business.smiu.edu.pk/ms-management.php','http://business.smiu.edu.pk/public-administration.php',
                 'http://cs.smiu.edu.pk/mscs_program.php','http://cs.smiu.edu.pk/PhD_program.php','http://edu.smiu.edu.pk/ms-education.php',
                 'http://env.smiu.edu.pk/ms-environmental-sciences.php','http://media.smiu.edu.pk/ms-media-studies-new.php','http://media.smiu.edu.pk/ms-social-sciences.php',
                 '+92 21 99217501-3','http://lms.smiu.edu.pk:8012/','http://smiu.fm/ FM 96.6','https://www.youtube.com/channel/UCFgimy2ABAwoT723zpBhMcA',
                 'info@smiu.edu.pk','http://www.smiu.edu.pk/academic-policies.php','http://www.smiu.edu.pk/admissions-criteria.php',
                 'http://www.smiu.edu.pk/images/Admissions-Examinations-Policy.pdf','http://business.smiu.edu.pk/bba2-program.php')


punctuation = ('!','#','$','%','&','(',')','*','+','-','/','.',':',';','}','{',']','[','=','`','~','?','<','>','"','|','_',';',',')
preprocessing=new_stopwords + punctuation
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question).strip())



# word2count = {}
# for question in clean_questions:
#     for word in question.split():
#         if word not in word2count:
#             word2count[word] = 1
#         else:
#             word2count[word] += 1
# for answer in answers:
#     for word in answer.split():
#         if word not in word2count:
#             word2count[word] = 1
#         else:
#             word2count[word] += 1
# # Creating two dictionaries that map the questions words and the answers words to a unique integer
# threshold_questions = 1
# questionswords2int = {}
# word_number = 0
# for word, count in word2count.items():
#     if count >= threshold_questions:
#         questionswords2int[word] = word_number
#         word_number += 1
# threshold_answers = 1
# answerswords2int = {}
# word_number = 0
# for word, count in word2count.items():
#     if count >= threshold_answers:
#         answerswords2int[word] = word_number
#         word_number += 1


# print(len(word2count.items()))
# print(len(answerswords2int.items()))
# print(len(questionswords2int.items()))
# print(answerswords2int)

# # Adding the last tokens to these two dictionaries
# tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
# for token in tokens:
#     questionswords2int[token] = len(questionswords2int) + 1 
# for token in tokens:
#     answerswords2int[token] = len(answerswords2int) + 1

# # Creating the inverse dictionary of the answerswords2int dictionary
# answersints2word = {w_i: w for w, w_i in answerswords2int.items()}


# # Adding the End Of String token to the end of every answer
# clean_answers = dataset['answers']
# for i in range(len(clean_answers)):
#       clean_answers[i] += ' <EOS>'
# print(clean_answers[:2])
# # Translating all the questions and the answers into integers
# # and Replacing all the words that were filtered out by <OUT> 
# questions_into_int = []
# for question in clean_questions:
#     ints = []
#     for word in question.split():
#         if word not in questionswords2int:
#             ints.append(questionswords2int['<OUT>'])
#         else:
#             ints.append(questionswords2int[word])
#     questions_into_int.append(ints)
# print(questions_into_int[:5])
# answers_into_int = []
# for answer in clean_answers:
#     ints = []
#     for word in answer.split():
#         if word not in answerswords2int:
#             ints.append(answerswords2int['<OUT>'])
#         else:
#             ints.append(answerswords2int[word])
#     answers_into_int.append(ints)
# print(answers_into_int[:5]) 
# sorted_clean_questions = []
# sorted_clean_answers = []
# for length in range(1, 25 + 1):
#     for i in enumerate(questions_into_int):
#         if len(i[1]) == length:
#             sorted_clean_questions.append(questions_into_int[i[0]])
#             sorted_clean_answers.append(answers_into_int[i[0]])
# print(sorted_clean_questions[-3:])


