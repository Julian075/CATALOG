import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import argparse
import datetime
from models import CATALOG_Base_fine_tuning as md
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random
from PIL import Image
import clip
import wandb



class CustomDataset(Dataset):
    def __init__(self,json_path):
        self.root_dir = json_path
        self.samples = self._load_samples()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess_clip = clip.load('ViT-B/16', self.device)


    def _load_samples(self):
        samples=[]
        data_dict=torch.load(self.root_dir)
        for key in data_dict.keys():
            samples.append([data_dict[key]['image_features'],data_dict[key]['description_embeddings'][0],data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, description_embeddings, target_index= self.samples[idx]
        image_features= self.preprocess_clip(Image.open(image_features).convert("RGB")).unsqueeze(0)#.to(self.device)
        image_features=image_features[0]
        return image_features, description_embeddings, target_index


# Define your DataLoader
def get_dataloader(root_dir, batch_size):
    dataset = CustomDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)









if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Descripción del programa')


    parser.add_argument('--ruta_features_train', type=str,
                        default="../../../features/Features_serengeti/finetuning_features/Features_CATALOG_train_16.pt",
                        help='Training path')
    parser.add_argument('--ruta_features_val', type=str,
                        default="../../../features/Features_serengeti/finetuning_features/Features_CATALOG_val_16.pt",
                        help='Validation path')
    parser.add_argument('--ruta_features_test', type=str,
                        default="../../../features/Features_serengeti/finetuning_features/Features_CATALOG_test_16.pt",
                        help='Test path')
    parser.add_argument('--text_features', type=str,
                        default="../../../features/Features_serengeti/finetuning_features/Text_features_16.pt",
                        help='Text features for training')  # Path for the features of the descriptions categories (Extracted by CLIP text encoder)

    parser.add_argument('--weight_Clip', type=float, default=0.6, help='Alpha')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--pretrained', type=int, default=0, help='pretrained ')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers ')
    parser.add_argument('--dropout', type=float, default=0.27822, help='dropout ')
    parser.add_argument('--hidden_dim', type=int, default=1045, help='hidden_dim ')
    parser.add_argument('--lr', type=float, default=0.07641, help='learning rate ')
    parser.add_argument('--t', type=float, default=0.1, help='temperature ')
    parser.add_argument('--momentum', type=float, default=0.8409, help='momentum ')
    parser.add_argument('--patience', type=int, default=5, help='patience ')

    # python Train_CATALOG_V2_Serengeti.py --ruta_features_train "Features_Serengeti/Features_CATALOG_train_16_mod.pt" --ruta_features_val "Features_Serengeti/Features_CATALOG_val_16_mod.pt" --ruta_features_test "Features_Serengeti/Features_CATALOG_test_16_mod.pt"  --text_features "Features_Serengeti/Text_features_16.pt" --weight_Clip 0.60855 --num_epochs 86 --batch_size 26 --pretrained 0 --num_layers 4 --dropout 0.381 --hidden_dim 1743 --lr 0.0956 --t 0.1 --momentum 0.8162 --patience 20

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
    path_text_feat=args.text_features

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def build_optimizer(projection_model, optimizer, learning_rate, momentum):
        params1 = {"params": projection_model.description_encoder.parameters(), "lr": learning_rate,
                   "momentum": momentum}
        params2 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}
        params3 = {"params": projection_model.logit_scale_LLaVA, "lr": learning_rate, "momentum": momentum}
        params4 = {"params": projection_model.model_clip.visual.parameters(), "lr": learning_rate,
                   "momentum": momentum}

        if optimizer == "sgd":
            optimizer = optim.SGD([params1, params2, params3, params4], lr=learning_rate, momentum=momentum)
        elif optimizer == "adam":
            optimizer = optim.Adam([params1, params2, params3, params4], lr=learning_rate)

        return optimizer

    def train():
        set_seed(42)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features=torch.load(path_text_feat)
        #text_features = text_features.to(device)



        projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,device=device)
        projection_model = projection_model.to(device)

        #DataLoader
        dataloader = get_dataloader(ruta_features_train, batch_size)
        dataloader_val = get_dataloader(ruta_features_val,batch_size)
        dataloader_test = get_dataloader(ruta_features_test, batch_size)

        optimizer = build_optimizer(projection_model,'sgd',lr,momentum)
        acc_best = 0
        counter = 0
        for epoch in range(num_epochs):
            print(epoch)
            # Training
            projection_model.train()
            time_in = time.time()
            running_loss = 0.0
            running_corrects = 0.0
            size=0
            for batch in dataloader:
                image_features, description_embeddings, target_index = batch
                size+=len(description_embeddings)
                image_features=image_features.to(device)
                description_embeddings = description_embeddings.to(device)


                loss, acc,_ = projection_model(description_embeddings, image_features, text_features, weight_Clip,target_index,t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # statistics Train
                running_loss += loss.item()
                running_corrects += float(acc)

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = (running_corrects / size) * 100

            # validation
            projection_model.eval()

            running_loss_val = 0
            running_corrects_val = 0.0
            size_val=0
            
            with torch.no_grad():

                for batch_val in dataloader_val:
                    image_features_val, description_embeddings_val, target_index_val = batch_val
                    size_val+=len(description_embeddings_val)
                    image_features_val = image_features_val.to(device)
                    description_embeddings_val = description_embeddings_val.to(device)



                    loss_val, acc_val,_ = projection_model(description_embeddings_val, image_features_val, text_features,
                                                 weight_Clip,target_index_val,t)

                    running_loss_val += loss_val.item()
                    running_corrects_val += float(acc_val)

                

            epoch_loss_val = running_loss_val / len(dataloader_val)
            epoch_acc_val = (running_corrects_val / size_val) * 100

            wandb.log({"acc_train": epoch_acc, "loss_train": epoch_loss, "acc_val": epoch_acc_val, "loss_val": epoch_loss_val})


            time_end = time.time()
            total_time = time_end - time_in
            # Print the loss at every nth epoch
            print_every = 1
            if epoch % print_every == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}]")
                print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print('Val loss: {:.4f},Val acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
                print(f"Time for epoch [{total_time}]")
                if epoch_acc_val > acc_best:
                    print('Save model')
                    acc_best = epoch_acc_val
                    counter = 0

                    # Create a directory for each training session based on the unique identifier
                    os.makedirs(f'Best_serengeti/training_{unique_id}', exist_ok=True)
                    # Save the model parameters within the corresponding directory
                    model_params_path = f'Best_serengeti/training_{unique_id}/best_model_params_{num_layers}_{hidden_dim}.pth'
                    torch.save(projection_model.state_dict(), model_params_path)

                else:
                    counter = counter + 1
                    print("The acc don't increase")

            if epoch==(num_epochs-1) or counter >= patience:
                projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                                 device=device)
                projection_model.load_state_dict(torch.load(model_params_path))
                projection_model = projection_model.to(device)
                projection_model.eval()

                running_loss_test = 0
                running_corrects_test = 0.0
                size_test = 0


                # Variables to calculate confusion matrix
                all_preds = []
                all_labels = []


                with torch.no_grad():

                    for batch_test in dataloader_test:
                        image_features_test, description_embeddings_test, target_index_test = batch_test
                        size_test += len(description_embeddings_test)
                        image_features_test = image_features_test.to(device)
                        description_embeddings_test = description_embeddings_test.to(device)


                        loss_test, acc_test,preds_test = projection_model(description_embeddings_test,
                                                                     image_features_test, text_features,
                                                                     weight_Clip, target_index_test, t)

                        all_preds.extend(preds_test.cpu().numpy())
                        all_labels.extend(target_index_test.cpu().numpy())

                        running_loss_test += loss_test.item()
                        running_corrects_test += float(acc_test)


                epoch_loss_test = running_loss_test / len(dataloader_test)
                epoch_acc_test = (running_corrects_test / size_test) * 100

                print('Test loss: {:.4f},Test acc: {:.4f}'.format(epoch_loss_test, epoch_acc_test))
                wandb.log({"acc_test": epoch_acc_test, "loss_test": epoch_loss_test})

                # Calculate confusion matrix
                conf_matrix = confusion_matrix(all_labels, all_preds)
                df_conf_matrix = pd.DataFrame(conf_matrix)
                df_conf_matrix.to_csv('conf_matrix_Base_serengeti.csv', index=False)

            # Check early stopping condition
            if counter >= patience:
                print(f'Validation acc has not improved for {patience} epochs. Stopping training.')
                break
    def prueba_model(model_params_path):# to calculate the acc in test for a saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(path_text_feat)
        text_features = text_features.to(device)

        
        dataloader_test = get_dataloader(ruta_features_test, batch_size)


        projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                         device=device)
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()
        
        running_loss_test = 0
        running_corrects_test = 0.0
        size_test = 0
        
        # Variables to calculate confusion matrix
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_test in dataloader_test:
                image_features_test, description_embeddings_test, target_index_test = batch_test
                size_test += len(description_embeddings_test)
                image_features_test = image_features_test.to(device)
                description_embeddings_test = description_embeddings_test.to(device)

                loss_test, acc_test,preds_test  = projection_model(description_embeddings_test,
                                                               image_features_test, text_features,
                                                               weight_Clip, target_index_test, t)

                all_preds.extend(preds_test.cpu().numpy())
                all_labels.extend(target_index_test.cpu().numpy())

                running_loss_test += loss_test.item()
                running_corrects_test += float(acc_test)
                
        epoch_loss_test = running_loss_test / len(dataloader_test)
        epoch_acc_test = (running_corrects_test / size_test) * 100

        unique_preds = np.unique(all_preds)
        print(unique_preds)

        print('Test loss: {:.4f}, Test acc: {:.4f}'.format(epoch_loss_test, epoch_acc_test))

        conf_matrix = confusion_matrix(all_labels, all_preds)
        df_conf_matrix = pd.DataFrame(conf_matrix)
        df_conf_matrix.to_csv('conf_matrix_Base_serengeti.csv')

        print("Confusion Matrix for Test:")
        print(conf_matrix)
        print("\nClassification Report for Test:")
        print(classification_report(all_labels, all_preds))


    def prueba_model_top_3(model_params_path):  # to calculate the acc in test for a saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(path_text_feat)
        text_features = text_features.to(device)

        dataloader_test = get_dataloader(ruta_features_test, batch_size)

        projection_model = md.LLaVA_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                                         device=device)
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()

        running_corrects_test = 0.0
        size_test = 0


        with torch.no_grad():
            for batch_test in dataloader_test:
                image_features_test, description_embeddings_test, target_index_test = batch_test
                size_test += len(description_embeddings_test)
                image_features_test = image_features_test.to(device)
                description_embeddings_test = description_embeddings_test.to(device)

                acc_test = projection_model.predict_top_3(description_embeddings_test,
                                                                   image_features_test, text_features,
                                                                   weight_Clip, target_index_test, t)


                running_corrects_test += float(acc_test)

        epoch_acc_test = (running_corrects_test / size_test) * 100
        print('Test acc Top 3: {:.4f}'.format(epoch_acc_test))




    model_params_path = '../../../models/CATALOG_finetuning_Base_Serengeti.pth'
    #prueba_model_top_3(model_params_path)
    #prueba_model(model_params_path)
    wandb.login(key="282780c770de0083eddfa3c56402f555ee60e108")
    wandb.init(
        project="Train_CLIP_V1_in_domain_Serengeti",
    )
    train()
    wandb.finish()