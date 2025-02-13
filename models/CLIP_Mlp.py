import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers,dropout_prob):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size)).half()
           # self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_prob)).half()

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size)).half()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            #x = self.batch_norms[i](x)
            x = nn.functional.relu(x)

        # Output layer without activation
        x = self.layers[-1](x)
        return x

class CLIP_MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout) -> None:
        super().__init__()
        self.projection_model = MLP(input_size=512, hidden_size=hidden_dim, output_size=512, num_layers=num_layers, dropout_prob=dropout)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    # LLaVA-CLIP Loss
    def CLIP_MLP_loss(self, logits: torch.Tensor, labels):
        labels=labels.to(self.device)
        loss = self.criterion(logits,labels)

        return loss
    # MLP_CLIP  Accuracy

    def CLIP_MLP_acc(self, logits, target_ind):

        predicted_index = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, img_features, txt_features, target_ind,t=0):
        # Similarity

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)
        logits=self.projection_model(similarity_clip)

        loss = self.CLIP_MLP_loss(logits, target_ind)
        acc = self.CLIP_MLP_acc(logits, target_ind)
        return loss, acc

class model_MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pretrained, pretrained_path="") -> None:
        super().__init__()
        self.projection_model = MLP(input_size=512, hidden_size=hidden_dim, output_size=16, num_layers=num_layers, dropout_prob=dropout)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    # LLaVA-CLIP Loss
    def model_MLP_loss(self, logits: torch.Tensor, labels):
        labels=labels.to(self.device)
        loss = self.criterion(logits,labels)

        return loss
    # MLP_CLIP  Accuracy

    def model_MLP_acc(self, logits, target_ind):

        predicted_index = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, img_features,txt_features,target_ind,t=0):
        logits=self.projection_model(img_features)

        loss = self.model_MLP_loss(logits, target_ind)
        acc = self.model_MLP_acc(logits, target_ind)
        return loss, acc