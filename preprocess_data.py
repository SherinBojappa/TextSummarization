import numpy as np
import pandas as pd
import re
import pickle

#N = 50000
reviews = pd.read_csv("Reviews.csv")
#reviews = reviews.head(N)

# remove entries where any of the labels have null values
review_data = reviews.dropna(axis = 0, how='any' )
review_data = review_data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'], axis=1)
# Get continuous indices by using reset_index
review_data = review_data.reset_index(drop = True)
review_data.head()

from string import lower
# Convert to lower case
review_data['Summary'] = review_data['Summary'].apply(lower)
review_data['Text'] = review_data['Text'].apply(lower)

# Remove characters which do not add to the summary or text
review_data['Summary'] = review_data['Summary'].map(lambda x: re.sub(r'\W+', ' ', x))
review_data['Text'] = review_data['Text'].map(lambda x: re.sub(r'\W+', ' ', x))
review_data.head()

# extract the word embeddings from ConceptNet
embedding_file = open('numberbatch-en.txt')
cn_embedding = {}
i = 0
for line in embedding_file:
    embed = line.split(' ')
    cn_embedding[embed[0]] = embed[1:]     

from nltk import word_tokenize

def tockenize(text):
    try:
        w = word_tokenize(text)[:100]
    except:
        return []
    return w

review_summary_tockenized = review_data['Summary'].apply(tockenize)
review_text_tockenized = review_data['Text'].apply(tockenize)

new_words = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

# Add the above new words to the existing word embedding with random embeddings drawn 
# from a uniform distribution
for word in new_words:
    cn_embedding[word] = np.array(np.random.uniform(-1.0, 1.0, len(cn_embedding["tripolitan"])))

# count the number of times each word occurs in the text and summary. For this a dictionary 
# with key as words and value as number of words will be created
word_frequency_summary = {}
word_frequency_text = {}

for i in range(len(review_summary_tockenized)):
    for word in review_summary_tockenized[i]:
        if word in word_frequency_summary:
            word_frequency_summary[word] += 1
        else:
            word_frequency_summary[word] = 1
            
for i in range(len(review_text_tockenized)):
    for word in review_text_tockenized[i]:
        if word in word_frequency_text:
            word_frequency_text[word] += 1
        else:
            word_frequency_text[word] = 1

# New words which frequently occur, with a frequency greater than min_freq, in the reviews 
# and the summaries but not in the word embeddings will be added to embeddings with random 
# weights
min_freq = 25

for word in cn_embedding.keys():
    if (word not in word_frequency_summary) and (word not in word_frequency_text) and (word not in new_words):
        del cn_embedding[word]
        
for word, count in word_frequency_summary.iteritems():
    if (word not in cn_embedding) and (count >= min_freq):
        cn_embedding[word] = np.array(np.random.uniform(-1.0, 1.0, 
                                                        len(cn_embedding["<UNK>"])))

for word, count in word_frequency_text.iteritems():
    if (word not in cn_embedding) and (count >= min_freq):
        cn_embedding[word] = np.array(np.random.uniform(-1.0, 1.0, 
                                                        len(cn_embedding["<UNK>"])))

word_to_num = {}
num_to_word = {}

num = 0
for word, _ in cn_embedding.iteritems():
    word_to_num[word] = num
    num_to_word[num] = word
    num += 1

# set <PAD> num_to_word to 0 by swapping with word having id 0
word_at_0 = num_to_word[0]
pad_id = word_to_num["<PAD>"]

# swapping
num_to_word[0] = "<PAD>"
num_to_word[pad_id] = word_at_0
word_to_num["<PAD>"] = 0
word_to_num[word_at_0] = pad_id

# Set batch size and others
vocab_size = len(cn_embedding)
embedding_size = len(cn_embedding["<UNK>"])

EOS = word_to_num["<EOS>"]
GO = word_to_num["<GO>"]

cn_embedding_matrix_numpy = np.zeros((vocab_size, embedding_size), dtype=np.float32)

for word, _ in cn_embedding.iteritems():
    cn_embedding_matrix_numpy[word_to_num[word]] = cn_embedding[word]

# Represent each sentence as a sequence of numbers based on the integer value of the sentence
# in the word embedding. Also compute the length of each sentence and store it in another list

def convert_text_to_num(text, append_eos = False):
    text_to_num = []
    text_length = []
    for sentence in text:
        sentence_to_num = []
        text_length.append(len(sentence))
        # sentence to num has integer representations of the word. If the word doesnt have a 
        # word embedding then that word is replaced by the embedding of the unkown word.
        for word in sentence:
            if word in cn_embedding:
                sentence_to_num.append(word_to_num[word])
            else:
                sentence_to_num.append(word_to_num["<UNK>"])
        # This indicates the end of the sentence
        if append_eos:
            sentence_to_num.append(word_to_num["<EOS>"])
        text_to_num.append(sentence_to_num)
    return text_to_num, text_length
        
# Convert review text to be represented as numbers
review_text_to_num, review_text_sentence_length = convert_text_to_num(review_text_tockenized ,
                                                                      append_eos = False)

# Convert review summaries to be represented as numbers 
review_summary_to_num, review_summary_sentence_length = convert_text_to_num(review_summary_tockenized)

sorted_text_indices = np.argsort(review_text_sentence_length)

preprocessed_data = {}
preprocessed_data["cn_embedding_matrix_numpy"] = cn_embedding_matrix_numpy
preprocessed_data["word_to_num"] = word_to_num
preprocessed_data["num_to_word"] = num_to_word
preprocessed_data["sorted_text_indices"] = sorted_text_indices
preprocessed_data["review_text_to_num"] = review_text_to_num
preprocessed_data["review_summary_to_num"] = review_summary_to_num

pickle.dump(preprocessed_data, open('preprocessed_data.pkl', 'w'))

