import torchvision.transforms as transforms
from preprocess import load_image, normalize_batch, load_doc, load_descriptions, clean_descriptions
from preprocess import to_vocabulary, save_descriptions, load_set, load_clean_descriptions
from preprocess import load_photo_features, to_lines, create_tokenizer, max_length
from encoder import Encoder
from decoder import CaptionModel
from os import listdir
from pickle import dump, load
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import argparse
import torch
from torch import nn
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs",
                    default=2,
                    type=int,
                    required=False,
                    help="Number of epoch to run.")
parser.add_argument("--checkpoint",
                    default=None,
                    type=str,
                    required=False,
                    help="Saved model name.")

args = parser.parse_args()


# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = Encoder()
    # model.to(device)
    model.eval()
    # extract features from each photo
    features = dict()
    for i, name in enumerate(listdir(directory)):
        # load an image from file
        filename = directory + '/' + name
        image = load_image(filename, size=224)
        # convert the image pixels to a numpy array
        image = transforms.ToTensor()(image)
        # reshape data for the model
        image = image.unsqueeze(0)
        # prepare the image for the VGG model
        image = normalize_batch(image)
        # get features
        feature = model(image)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
#         print('>%s' % name)
        if i%50==0:
            print("{} image done.".format(i))
    return features

#Uncomment the below code to extract the features of images from VGG16 as I already have the file I have commented it

# # extract features from all images
# directory = 'Flicker8k_Dataset'
# features = extract_features(directory)
# print('Extracted Features: %d' % len(features))
# # save to file
# dump(features, open('features.pkl', 'wb'))
# print("Features are extracted and saved into the file")


filename = 'text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')

# load training dataset (6K)
filename = 'text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset for training: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

# load testing dataset (6K)
filename = 'text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: train=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: train=%d' % len(test_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved")

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            # out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(torch.from_numpy(in_seq).unsqueeze(0))
            y.append(out_seq)
    return torch.cat(X1, dim=0), torch.cat(X2, dim=0), y

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    for key, desc_list in descriptions.items():
        # retrieve the photo feature
        photo = photos[key]
        in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
        yield (in_img, in_seq, out_word)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CaptionModel(vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if args.checkpoint != None:
    print("Loading the checkpoint")
    model.load_state_dict(torch.load(args.checkpoint))

print("Number of epochs ", args.num_epochs)
for epoch in range(args.num_epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    test_generator = data_generator(test_descriptions, test_features, tokenizer, max_length, vocab_size)
    tr_loss, test_loss = 0, 0
    training_examples, test_examples = 0, 0
    model.train()
    for batch, data in enumerate(generator):
        image, caption, target_word = data

        out = model(image.to(device), caption.type(torch.LongTensor).to(device))
        loss = loss_fn(out, torch.from_numpy(np.array(target_word)).to(device))
        
        tr_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        training_examples += image.size(0)
        if (batch+1)%200 == 0:
            print("Epoch: {}, Batch: {}, loss: {}, avg loss: {}".format(epoch+1, batch+1, loss.item(), tr_loss/(training_examples)))
            
        if (batch+1)%400 == 0:
            model.eval().cpu()
            ckpt_model_path = os.path.join('results', 'ckpt_epoch_{}_batch_{}.pth'.format(epoch+1, batch+1))
            torch.save(model.state_dict(), ckpt_model_path)
            model.to(device).train()

    model.eval()
    for test_batch, data in enumerate(test_generator):
        image, caption, target_word = data
        out = model(image.to(device), caption.type(torch.LongTensor).to(device))
        loss = loss_fn(out, torch.from_numpy(np.array(target_word)).to(device))
        
        test_loss += loss.item()
        test_examples += image.size(0)
    
    print("Epoch {}, Training loss: {}, Test loss: {}".format(epoch+1, tr_loss/training_examples, test_loss/test_examples))
            

print("Training Complete") 