import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import model_V2 as md
import numpy as np
import random
from PIL import Image
from transformers import BertModel, BertTokenizer
import clip
import json
import pandas as pd


if __name__ == "__main__":

    # Parameters for training
    parser = argparse.ArgumentParser(description='Descripción del programa')

    parser.add_argument('--ruta_features_train', type=str,
                        default="Features_Serengeti/Features_LLaVA_CLIP_train_16.pt",
                        help='Training path')
    parser.add_argument('--ruta_features_val', type=str,
                        default="Features_Serengeti/Features_LLaVA_CLIP_val_16.pt",
                        help='Validation path')
    parser.add_argument('--ruta_features_test', type=str,
                        default="Features_Serengeti/Features_LLaVA_CLIP_test_16.pt",
                        help='Test path')
    parser.add_argument('--text_features', type=str,
                        default="Features_Serengeti/Text_features_16.pt",
                        help='Text features for training')  # Path for the features of the descriptions categories (Extracted by CLIP text encoder)

    parser.add_argument('--weight_Clip', type=float, default=0.60855, help='Alpha')
    parser.add_argument('--num_epochs', type=int, default=86, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=26, help='batch size')
    parser.add_argument('--pretrained', type=int, default=0, help='pretrained ')
    parser.add_argument('--num_layers', type=int, default=4, help='num_layers ')
    parser.add_argument('--dropout', type=float, default=0.381, help='dropout ')
    parser.add_argument('--hidden_dim', type=int, default=1743, help='hidden_dim ')
    parser.add_argument('--lr', type=float, default=0.0956, help='learning rate ')
    parser.add_argument('--t', type=float, default=0.1, help='temperature ')
    parser.add_argument('--momentum', type=float, default=0.8162, help='momentum ')
    parser.add_argument('--patience', type=int, default=20, help='patience ')

    args = parser.parse_args()

    ruta_features_train = args.ruta_features_train
    ruta_features_val = args.ruta_features_val
    ruta_features_test = args.ruta_features_test
    weight_Clip = args.weight_Clip
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    pretrained = args.pretrained
    num_layers = args.num_layers
    dropout = args.dropout
    hidden_dim = args.hidden_dim
    lr = args.lr
    t = args.t
    momentum = args.momentum
    patience = args.patience
    path_text_feat = args.text_features


    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def build_optimizer(projection_model, optimizer, learning_rate, momentum):
        params1 = {"params": projection_model.description_encoder.parameters(), "lr": learning_rate,
                   "momentum": momentum}
        params2 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}
        params3 = {"params": projection_model.logit_scale_LLaVA, "lr": learning_rate, "momentum": momentum}

        if optimizer == "sgd":
            optimizer = optim.SGD([params1, params2, params3], lr=learning_rate, momentum=momentum)
        elif optimizer == "adam":
            optimizer = optim.Adam([params1, params2, params3], lr=learning_rate)

        return optimizer



    def pru_ind(model_params_path,image_path,description_path):

        device = "cuda" if torch.cuda.is_available() else "cpu"


        # Initialize your models, tokenizer, etc.
        tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
        model_Bert = BertModel.from_pretrained('bert-base-uncased')
        model_Bert.to(device)

        model_clip, preprocess_clip = clip.load('ViT-B/16', device)
        model_clip.to(device)

        text_features = torch.load(path_text_feat)
        text_features = text_features.to(device)

        projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                         pretrained=0, pretrained_path="")
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()

        root_dir = os.path.join(image_path)

        outputs={}
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                # path where is located the images descriptions generated by LLaVA
                json_path = os.path.join(description_path, category, img_name[:-4] + '.json')

                target_index = int(category)
                #target_index = class_indices[category.lower()]

                images = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model_clip.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # images = images.unsqueeze(0)[0]
                f = open(json_path)
                data = json.load(f)
                description = data['description']
                f.close()
                tokens = tokenizer_Bert.tokenize(description)
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
                token_ids = tokenizer_Bert.convert_tokens_to_ids(tokens)

                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
                token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                with torch.no_grad():
                    output_bert = model_Bert(token_ids, attention_mask=attention_mask)
                    description_embeddings = output_bert.pooler_output

                image_features = image_features.to(device)
                description_embeddings = description_embeddings.to(device)

                pred = projection_model.predict(description_embeddings,image_features, text_features,
                                                                               weight_Clip, target_index, t)
                outputs[img_name[:-4]]={'class':target_index,'Prediction':pred.item()}
        df=pd.DataFrame(outputs)
        df.to_csv('Predictions_Serengeti_test.csv')

    image_path='D:/Udea/Maestria/Experimentos/scripts/Codigos_WACV/Test_examples/Serengeti'
    model_params_path = 'Models/CATALOG_V2_Serengeti.pth'
    description_path ='D:/Udea/Maestria/Bases_de_datos/snaphsot_serengeti_cropped_single_animals/organizado/descriptions_serengetti/test'
    pru_ind(model_params_path,image_path,description_path)

