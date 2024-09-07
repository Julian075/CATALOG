import torch.nn as nn
import torch
import numpy as np

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

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
        # Skip classification layer and return node embeddings
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
            for i, (lin, lin2) in enumerate( zip(self.linears[:-1], self.linears2[:-1])):
                embed1 = lin(x)
                embed2 = self.drop(lin2(self.gelu(embed1)))
                x = self.lns[i](embed1 + embed2)


            embed1 = self.linears[-1](x)
            embed2 = self.drop(self.linears2[-1](self.gelu(embed1)))
            x = self.lns[-1](embed1 + embed2)
        else:
            print(f"Input shape: {x.shape}")  # Shape of the input to the linear layer
            print(f"Weight shape: {self.linears[0].weight.shape}")  # Shape of the first linear layer's weight
            embed1 = self.linears[0](x)
            embed2 = self.drop(self.linears2[0](self.gelu(embed1)))


            x = self.lns[0](embed1 + embed2)

        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


class LLaVA_CLIP(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pretrained,pretrained_path="") -> None:
        super().__init__()
        self.description_encoder = MLP(input_dim=512, hidden_dim=hidden_dim, output_dim=512, num_layers=num_layers,
                                       dropout=dropout, return_embeds=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_LLaVA = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if pretrained:
            self.pesos_preentrenados = torch.load(pretrained_path)#('weights/best_model_params_path.pth')
            self.description_encoder.load_state_dict(self.pesos_preentrenados)

    # LLaVA-CLIP Loss
    def LLaVA_CLIP_loss(self, logits: torch.Tensor,label,t) :
        loss_i=0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]]/t)
            dem = torch.sum(torch.exp(logits[b]/t))
            loss_i+=torch.log(num/dem)
        loss=-loss_i/len(logits)
        return loss

    def LLaVA_CLIP_loss2(self, logits: torch.Tensor, labels,t):
        temperature=t
        inputs_expected_class = {}
        for index in range(len(labels)):
            clas = labels[index]
            if not (clas in inputs_expected_class.keys()):
                inputs_expected_class[clas] = [index]
            else:
                inputs_expected_class[clas].append(index)


        loss = 0.00
        for category in inputs_expected_class.keys():
            aux_loss = 0.00
            for inputs_index in inputs_expected_class[category]:
                num = torch.exp((logits[inputs_index][labels[inputs_index]])/temperature)
                dem = torch.sum(torch.exp(logits[inputs_index]/temperature))
                aux_loss += torch.log(num / dem)
            loss += -aux_loss / len(inputs_expected_class[category])

        return loss

        # LLaVA-CLIP  Accuracy

    def LLaVA_CLIP_acc(self, img_feat,description_feat,text_feat,weight_p,target_ind):
        sim_clip= img_feat @ text_feat
        sim_clip = sim_clip / sim_clip.norm(dim=-1, keepdim=True)

        sim_bert = description_feat.half()  @ text_feat
        sim_bert = sim_bert / sim_bert.norm(dim=-1, keepdim=True)

        sim_total=sim_clip*weight_p+sim_bert*(1-weight_p)
        sim_total= sim_total / sim_total.norm(dim=-1, keepdim=True)

        predicted_index = torch.argmax(sim_total, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, embeddings, img_features, txt_features, weight_p,target_ind,temp):

        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()
        similarity_bert = (description_features.half() @ txt_features) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)



        loss = self.LLaVA_CLIP_loss(out_logits,target_ind,temp)
        #loss = self.LLaVA_CLIP_loss2(out_logits, target_ind,temp)
        acc = self.LLaVA_CLIP_acc(img_features,description_features,txt_features,weight_p,target_ind)
        return loss, acc,torch.argmax(out_logits, dim=1)

    def accuracy_top_3(self,output, target):
        topk = 3  # Queremos el top 3
        pred = output.topk(topk, 1, True, True)[1].t()
        pred=pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        return correct_k

    def predict(self, embeddings, img_features, txt_features, weight_p,target_ind,temp):

        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()
        similarity_bert = (description_features.half() @ txt_features) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)

        max_values, max_indices = torch.max(out_logits, dim=1)
        return max_values, max_indices

    def predict_top_3(self, embeddings, img_features, txt_features, weight_p,target_ind,temp):

        description_features = self.description_encoder(embeddings)
        description_features = description_features / description_features.norm(dim=-1, keepdim=True)

        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity_clip = (img_features @ txt_features) * logit_scale_CLIP
        similarity_clip = similarity_clip / similarity_clip.norm(dim=-1, keepdim=True)

        logit_scale_LLaVA = self.logit_scale_LLaVA.exp()
        similarity_bert = (description_features.half() @ txt_features) * logit_scale_LLaVA
        similarity_bert = similarity_bert / similarity_bert.norm(dim=-1, keepdim=True)

        similarity = (similarity_clip * weight_p + similarity_bert * (1 - weight_p))
        out_logits = similarity / similarity.norm(dim=-1, keepdim=True)

        acc=self.accuracy_top_3(out_logits,target_ind)

        return acc