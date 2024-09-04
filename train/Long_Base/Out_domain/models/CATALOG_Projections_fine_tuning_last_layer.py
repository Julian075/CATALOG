import torch.nn as nn
import torch
import numpy as np

import clip
from PIL import Image

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False).half()
        self.linear2 = nn.Linear(d_out, d_out, bias=False).half()
        self.layer_norm = nn.LayerNorm(d_out).half()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(nn.functional.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.linears2 = nn.ModuleList()
        self.gelu = QuickGELU()
        self.num_layers = num_layers

        if num_layers > 1:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.linears2.append(nn.Linear(hidden_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                self.linears2.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=True))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))
        else:
            self.linears.append(nn.Linear(input_dim, output_dim))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))

        self.lns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(output_dim))
        # The log softmax layer
        self.softmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for lin2 in self.linears2:
            lin2.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        if self.num_layers > 1:
            for i, (lin, lin2) in enumerate(zip(self.linears[:-1], self.linears2[:-1])):  # for i, lin in enumerate(self.linears[:-1]):
                embed1 = lin(x)
                embed2 = self.drop(lin2(self.gelu(embed1)))
                x = self.lns[i](embed1 + embed2)


            embed1 = self.linears[-1](x)
            embed2 = self.drop(self.linears2[-1](self.gelu(embed1)))
            x = self.lns[-1](embed1 + embed2)
        else:
            embed1 = self.linears[0](x)
            embed2 = self.drop(self.linears2[0](self.gelu(embed1)))
            x = self.lns[0](embed1 + embed2)

        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


class LLaVA_CLIP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, device) -> None:
        super().__init__()
        self.description_encoder = MLP(input_dim=768, hidden_dim=hidden_dim, output_dim=512, num_layers=num_layers,
                                       dropout=dropout, return_embeds=True)
        self.proyection_Img_CLIP=Projection(d_in=512,d_out=512,p=dropout)
        self.proyection_txt_CLIP = Projection(d_in=512, d_out=512, p=dropout)

        self.model_clip, _ = clip.load(f'ViT-B/16', device)
        self.model_clip.to(device)

        self.device = device

        for param in self.model_clip.parameters():
            param.requires_grad = False

        self.model_clip.visual.proj.requires_grad = True

        self.criterion=nn.CrossEntropyLoss(reduction="mean")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_LLaVA = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    # LLaVA-CLIP Loss
    def LLaVA_CLIP_loss(self, logits: torch.Tensor,labels,t):

        loss_i = 0
        for b in range(len(logits)):
            num = torch.exp(logits[b][labels[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i += torch.log(num / dem)
        loss = -loss_i / len(logits)

        return loss

    def LLaVA_CLIP_loss2(self, logits: torch.Tensor,labels,t):
        temperature=t

        inputs_expected_class={}
        for index in range(len(labels)):
            clas=labels[index]
            if not (clas in inputs_expected_class.keys()):
                inputs_expected_class[clas]=[index]
            else:
                inputs_expected_class[clas].append(index)


        loss=0.00
        for category in inputs_expected_class.keys():

            aux_loss=0.00
            for inputs_index in inputs_expected_class[category]:
                num = torch.exp((logits[inputs_index][labels[inputs_index]])/temperature)
                dem = torch.sum(torch.exp(logits[inputs_index]/temperature))
                aux_loss+=torch.log(num/dem)
            loss+=-aux_loss/len(inputs_expected_class[category])

        return loss


    # LLaVA-CLIP  Accuracy

    def LLaVA_CLIP_acc(self, img_feat,description_feat,text_feat,weight_p,target_ind):
        sim_clip= img_feat @ text_feat.t()
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)

        sim_bert = description_feat.half()  @ text_feat.t()
        sim_bert = sim_bert / sim_bert.norm(dim=-1, keepdim=True)

        sim_total=sim_clip*weight_p+sim_bert*(1-weight_p)
        sim_total = sim_total / sim_total.norm(dim=-1, keepdim=True)
        
        predicted_index = torch.argmax(sim_total, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, embeddings, img, txt_features, weight_p, target_ind, temp):

        #Train last Layer CLIP

        zeroshot_weights = []
        for i in range(len(txt_features)):
            txt=clip.tokenize(txt_features[i]).to(self.device)
            class_embeddings = self.model_clip.encode_text(txt)  # embed with text encoder
            class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)

        img_features = self.model_clip.encode_image(img)
        img_features = img_features/ img_features.norm(dim=-1, keepdim=True)


        # Proyections
        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)
        p_img_features = self.proyection_Img_CLIP(img_features)
        p_txt_features = self.proyection_txt_CLIP(zeroshot_weights.t())


        # Similarity

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (p_img_features @ p_txt_features.t()) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()

        similarity_bert = (description_features.half() @ p_txt_features.t()) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)



        loss = self.LLaVA_CLIP_loss(out_logits,target_ind,temp)
        #loss = self.LLaVA_CLIP_loss2(out_logits, target_ind,temp)
        acc = self.LLaVA_CLIP_acc(p_img_features,description_features,p_txt_features,weight_p,target_ind)
        return loss, acc, torch.argmax(out_logits, dim=1)

    def predict (self, embeddings, img_features, txt_features, weight_p, target_ind, temp):

        # Proyections
        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)
        p_img_features = self.proyection_Img_CLIP(img_features)
        p_txt_features = self.proyection_txt_CLIP(txt_features.t())

        # Similarity

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (p_img_features @ p_txt_features.t()) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()

        similarity_bert = (description_features.half() @ p_txt_features.t()) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)
        return torch.argmax(out_logits, dim=1)

    def accuracy_top_3(self,output, target):
        topk = 3  # Queremos el top 3
        pred = output.topk(topk, 1, True, True)[1].t()
        pred=pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        return correct_k

    def predict_top_3 (self, embeddings, img_features, txt_features, weight_p, target_ind, temp):

        # Proyections
        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)
        p_img_features = self.proyection_Img_CLIP(img_features)
        p_txt_features = self.proyection_txt_CLIP(txt_features.t())

        # Similarity

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (p_img_features @ p_txt_features.t()) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()

        similarity_bert = (description_features.half() @ p_txt_features.t()) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)

        acc = self.accuracy_top_3(out_logits, target_ind)

        return acc