from preprocess import load_set, load_clean_descriptions
from preprocess import load_photo_features, to_lines, create_tokenizer, max_length
from keras.preprocessing.sequence import pad_sequences
from decoder import CaptionModel
import numpy as np
import torch
from torch import nn
import os
from nltk.translate.bleu_score import corpus_bleu
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",
                    default=None,
                    type=str,
                    required=True,
                    help="Saved model name.")

args = parser.parse_args()

#loading the training dataset to get the same tokenizer that we used during the training

# loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocab_size = 7579
max_length = 34


# load testing dataset (6K)
filename = 'text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# map an integer to a word
def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
                if index == integer:
                        return word
        return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        sequence = torch.from_numpy(sequence)
        yhat = model(photo.to(device), sequence.type(torch.LongTensor).to(device))
        # convert probability to integer
        yhat = np.argmax(yhat.to("cpu").detach().numpy())
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
          break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
          break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print("\n\n\n\n")
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CaptionModel(vocab_size).to(device)
model.load_state_dict(torch.load(args.checkpoint)) 
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)