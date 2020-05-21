import torch
import torch.nn  as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
    Encodes the input image to a vector.
    # """
    def __init__(self):
        super(Encoder, self).__init__()

        vgg = models.vgg16(pretrained=True)
        
        model = torch.nn.Sequential()
        for name, child in vgg.named_children():
            if isinstance(child, torch.nn.Sequential):
                for cnt, layer in child.named_children():
                    layer_name = name + str(cnt)
                    model.add_module(layer_name, layer)
            else:
                model.add_module(name, child)
                model.add_module('flatten', nn.Flatten())
                
        # remove last two layers
        modules = list(model.children())[:-2]
        self.enc_model = nn.Sequential(*modules)
        for p in self.enc_model.parameters():
            p.requires_grad = False

    def forward(self, images):
        encoded_out = self.enc_model(images)
        return encoded_out