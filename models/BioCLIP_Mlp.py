import numpy as np
import torch
import torch.nn as nn
from models.CATALOG_Base import MLP as MLP_projection,Adapter



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers,dropout_prob):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_prob))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = nn.functional.relu(x)

        # Output layer without activation
        x = self.layers[-1](x)
        return x


class CLIP_MLP(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim, num_layers, dropout) -> None:
        super().__init__()
        self.mlp = MLP_projection(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def contrastive_loss(self, logits: torch.Tensor,label,t) :
        loss_i=0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i+=torch.log(num/dem)
        loss=-loss_i/len(logits)
        return loss

    def model_acc(self, img_feat,text_feat,target_ind):
        sim_clip= img_feat @ text_feat
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)

        predicted_index = torch.argmax(sim_clip, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc


    def forward(self, img_features, txt_features, target_ind,t=0):
        # Similarity
        mlp_img_features = self.mlp(img_features)
        mlp_img_features = mlp_img_features / mlp_img_features.norm(dim=-1, keepdim=True)


        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (mlp_img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        loss = self.contrastive_loss(similarity_clip, target_ind,t)
        acc = self.model_acc(mlp_img_features,txt_features, target_ind)
        return loss, acc


class CLIP_Adapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim, alpha=0.5):
        super(CLIP_Adapter, self).__init__()

        # Feature adapter (Av) - Adapt CLIP's image features
        self.visual_adapter = Adapter(feature_dim,hidden_dim)
        # Learnable residual blending factors
        self.alpha = nn.Parameter(torch.tensor(alpha))  # α trainable

        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def contrastive_loss(self, logits: torch.Tensor,label,t) :
        loss_i=0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i+=torch.log(num/dem)
        loss=-loss_i/len(logits)
        return loss

    def model_acc(self, img_feat,text_feat,target_ind):
        sim_clip= img_feat @ text_feat
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)

        predicted_index = torch.argmax(sim_clip, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, img_features, txt_features, target_ind,t=0):
        # Similarity
        adapter_img_features = self.visual_adapter(img_features)
        img_adapted = self.alpha * img_features + (1 - self.alpha) * adapter_img_features


        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_adapted @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        loss = self.contrastive_loss(similarity_clip, target_ind,t)
        acc = self.model_acc(img_adapted,txt_features, target_ind)
        return loss, acc

class Linear_probe(nn.Module):
    def __init__(self, hidden_dim,output_dim, num_layers, dropout) -> None:
        super().__init__()
        self.projection_model = MLP(input_size=512, hidden_size=hidden_dim, output_size=output_dim, num_layers=num_layers, dropout_prob=dropout)

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