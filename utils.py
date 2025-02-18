import clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from feature_extraction.CATALOG_feature_extraction import extract_features

#Datasets and Dataloaders
class BaselineDataset(Dataset):
    def __init__(self,_dict):
        self.data_dict = _dict
        self.samples = self._load_samples()


    def _load_samples(self):
        samples=[]
        for key in self.data_dict.keys():
            if len(self.data_dict[key])==3:
                samples.append([self.data_dict[key]['image_features'][0],self.data_dict[key]['description_embeddings'][0],self.data_dict[key]['target_index']])
            else:
                samples.append([self.data_dict[key]['image_features'][0],self.data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples[idx])==3:
            image_features, description_embeddings, target_index= self.samples[idx]
            return image_features, description_embeddings, target_index
        else:
            image_features, target_index = self.samples[idx]
            return image_features, target_index

def dataloader_baseline(root_dir, batch_size,BaselineDataset):
    dataset = BaselineDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)






#Tuning version
class TuningDataset(Dataset):
    def __init__(self,_dict):

        self.data_dict = _dict
        self.samples = self._load_samples()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess_clip = clip.load('ViT-B/16', self.device)


    def _load_samples(self):
        samples=[]

        for key in self.data_dict.keys():
            samples.append([self.data_dict[key]['image_features'],self.data_dict[key]['description_embeddings'][0],self.data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, description_embeddings, target_index = self.samples[idx]
        image_features = self.preprocess_clip(Image.open(image_features).convert("RGB")).unsqueeze(0)  # .to(self.device)
        image_features = image_features[0]
        return image_features, description_embeddings, target_index

def dataloader_Tuning(root_dir, batch_size,TuningDataset):
    dataset = TuningDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def build_optimizer( projection_model, optimizer, learning_rate, momentum, version,en_att=0):
    if version!='CLIP_MLP':
        params1 = {"params": projection_model.description_encoder.parameters(), "lr": learning_rate,
                    "momentum": momentum}

        if not en_att:
            params2 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}
            params3 = {"params": projection_model.logit_scale_LLaVA, "lr": learning_rate, "momentum": momentum}
        else:
            params2 = {"params": projection_model.cross_attention.parameters(), "lr": learning_rate, "momentum": momentum}
            params3 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}

    scheduler = None  # Inicializa el scheduler como None


    if optimizer == "sgd":
        if version == 'base':
            optimizer = optim.SGD([params1, params2, params3], lr=learning_rate, momentum=momentum)
        elif version == 'fine_tuning':
            params6 = {"params": projection_model.model_clip.visual.parameters(), "lr": learning_rate,
                       "momentum": momentum}
            optimizer = optim.SGD([params1, params2, params3, params6], lr=learning_rate,momentum=momentum)#, weight_decay=0.2
            T_max = 50
            eta_min = learning_rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        else:
            params_MLP = {"params": projection_model.parameters(), "lr": learning_rate, "momentum": momentum}
            optimizer = optim.SGD([params_MLP], lr=learning_rate, momentum=momentum)

    return optimizer,scheduler

def feature_extraction_(model_version,dataset,LLM):
    if model_version=='Base_long':
        extract_features(model_version=model_version, dataset=dataset, type_clip='longclip-B', LLM=LLM, only_text=0)
    elif  "BioCLIP" in model_version :
        extract_features(model_version=model_version, dataset=dataset, type_clip='BioCLIP', LLM=LLM, only_text=0)
    elif not('Long' in model_version):
        extract_features(model_version=model_version, dataset=dataset, type_clip='16', LLM=LLM, only_text=0)
    else:
        extract_features(model_version=model_version, dataset=dataset, type_clip='longclip-B', LLM=LLM, only_text=0)

