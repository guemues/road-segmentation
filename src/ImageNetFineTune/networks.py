import torch
import torch.nn as nn
import torchvision

class resnet_and_fc(nn.Module):
    def __init__(self, pretrained_model, hidden_nodes=256):
        super(resnet_and_fc, self).__init__()
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        num_ftrs = self.pretrained_model.fc.in_features
        # change the last layer of resnet
        self.pretrained_model.fc = nn.Linear(num_ftrs, hidden_nodes)
        # add one more fully connected layer
        self.last_layer = torch.nn.Linear(hidden_nodes, 2)


    def forward(self, x):
        # run resnet and one the last year
        return self.last_layer(self.pretrained_model(x))



class resnet_multi_scale(nn.Module):
    def __init__(self, pretrained_model, hidden_nodes=256, require_grad=False):
        super(resnet_multi_scale, self).__init__()
        self.pretrained_model = pretrained_model

        if not require_grad:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        num_ftrs = self.pretrained_model.fc.in_features
        # change the last layer of resnet
        self.pretrained_model.fc = nn.Linear(num_ftrs, hidden_nodes)
        # add one more fully connected layer
        self.last_layer = torch.nn.Linear(3*hidden_nodes, 2)


    def forward(self, x_tuple):
        # run resnet and one the last year
        x, x1, x2 = x_tuple
        fc1 = self.pretrained_model(x)
        fc2 = self.pretrained_model(x1)
        fc3 = self.pretrained_model(x2)
        # concatenate three different scale features
        fc = torch.cat([fc1, fc2, fc3], dim=1)
        return self.last_layer(fc)