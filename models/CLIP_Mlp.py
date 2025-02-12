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


class MLP_CLIP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pretrained, pretrained_path="") -> None:
        super().__init__()
        self.projection_model_img = MLP(input_size=512, hidden_size=hidden_dim, output_size=512, num_layers=num_layers, dropout_prob=dropout)

        self.projection_model_txt = MLP(input_size=512, hidden_size=hidden_dim, output_size=512, num_layers=num_layers, dropout_prob=dropout)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    # LLaVA-CLIP Loss
    def MLP_CLIP_loss(self, logits: torch.Tensor, labels, t):

        loss_i = 0
        for b in range(len(logits)):
            num = torch.exp(logits[b][labels[b]]/t)
            dem = torch.sum(torch.exp(logits[b] / t))
            loss_i += torch.log(num / dem)
        loss = -loss_i / len(logits)

        return loss

    def MLP_CLIP_loss2(self, logits: torch.Tensor, labels, t):
        temperature = t
        # creo un dict para saber en el batch cuales inputs tienen los mismos target labels de salida
        inputs_expected_class = {}
        for index in range(len(labels)):
            clas = labels[index]
            if not (clas in inputs_expected_class.keys()):
                inputs_expected_class[clas] = [index]
            else:
                inputs_expected_class[clas].append(index)

        # Iterar sobre tod.o el batch
        loss = 0.00
        for category in inputs_expected_class.keys():
            # iterar sobre inputs con mismo label
            aux_loss = 0.00
            for inputs_index in inputs_expected_class[category]:
                num = torch.exp((logits[inputs_index][labels[inputs_index]]) / temperature)
                dem = torch.sum(torch.exp(logits[inputs_index] / temperature))
                aux_loss += torch.log(num / dem)
            loss += -aux_loss / len(inputs_expected_class[category])

        return loss

    # MLP_CLIP  Accuracy

    def MLP_CLIP_acc(self, img_feat, text_feat, target_ind):
        sim_clip = img_feat @ text_feat.t()
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)


        predicted_index = torch.argmax(sim_clip, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, img_features, txt_features, target_ind, temp):

        # Proyections
        p_img_features = self.projection_model_img(img_features)
        p_txt_features = self.projection_model_txt(txt_features.t())

        # Similarity

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (p_img_features @ p_txt_features.t()) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        out_logits = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        loss = self.MLP_CLIP_loss(out_logits, target_ind, temp)
        acc = self.MLP_CLIP_acc(p_img_features, p_txt_features, target_ind)
        return loss, acc

class CLIP_MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pretrained, pretrained_path="") -> None:
        super().__init__()
        self.projection_model = MLP(input_size=16, hidden_size=hidden_dim, output_size=16, num_layers=num_layers, dropout_prob=dropout)

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