import torchvision.transforms as transforms
from preprocess import load_image, normalize_batch
from keras.preprocessing.sequence import pad_sequences
from decoder import CaptionModel
from encoder import Encoder
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint",
                    default=None,
                    type=str,
                    required=True,
                    help="Saved model name.")

args = parser.parse_args()


# loading the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocab_size = 7579
max_length = 34


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_model = Encoder()
decoder_model = CaptionModel(vocab_size).to(device)
decoder_model.load_state_dict(torch.load(args.checkpoint))
decoder_model.eval()
for image_name in os.listdir("evaluate/images"):
    print(image_name)
    image = load_image(os.path.join("evaluate/images/",image_name), size=224)
    # convert the image pixels to a numpy array
    image = transforms.ToTensor()(image)
    # reshape data for the model
    image = image.unsqueeze(0)
    # prepare the image for the VGG model
    image = normalize_batch(image)
    features = encoder_model(image)
    predicted_sentence = generate_desc(decoder_model, tokenizer, features, max_length)
    img = plt.imread(os.path.join("evaluate/images/",image_name))
    plt.imshow(img)
    plt.axis('off')
    plt.title(predicted_sentence)
    plt.savefig(os.path.join("evaluate/results/",
                                image_name+'_result.jpg'))

print("Complete")