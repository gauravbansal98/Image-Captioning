import torch
import torch.nn  as nn

class CaptionModel(nn.Module):
    
    def __init__(self, vocab_size):
        super(CaptionModel, self).__init__()
        
        # Input from the encoder decoder
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('FC', nn.Linear(in_features=4096, out_features=256))
        self.feature_extractor.add_module('activation', nn.ReLU())
        
        # Sequence Model
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256, padding_idx=0)
        self.emb_drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, num_layers=1)
        
        # Decode Model
        self.decoder = nn.Sequential()
        self.decoder.add_module('FC1', nn.Linear(in_features=256, out_features=256))
        self.decoder.add_module('activation', nn.ReLU())
        self.decoder.add_module('FC2', nn.Linear(in_features=256, out_features=vocab_size))
        
    def forward(self, image, caption):
        image = self.feature_extractor(image)
        embd = self.embeddings(caption)
        embd_drop = self.emb_drop(embd)
        outputs, (hidden, _) = self.lstm(embd_drop)
        hidden = hidden.view(hidden.size(1), -1)
        decoder_features = image + hidden
        out = self.decoder(decoder_features)
        return out