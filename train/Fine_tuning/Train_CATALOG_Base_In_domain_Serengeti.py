import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import argparse
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random
from PIL import Image
import clip
import wandb



class CATALOG_base_In_domain_serengeti:
    def __init__(self,weight_Clip,num_epochs,batch_size,num_layers,dropout,hidden_dim,lr,t,momentum,patience,model,Dataset,Dataloader,version,ruta_features_train,ruta_features_val,ruta_features_test,path_text_feat,build_optimizer,exp_name):
        self.ruta_features_train = ruta_features_train
        self.ruta_features_val   = ruta_features_val
        self.ruta_features_test  = ruta_features_test
        self.path_text_feat= path_text_feat
        self.weight_Clip=weight_Clip
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.hidden_dim=hidden_dim
        self.lr=lr
        self.t=t
        self.momentum=momentum
        self.patience=patience
        self.md=model
        self.dataset=Dataset
        self.dataloader=Dataloader
        self.version=version
        self.build_optimizer=build_optimizer
        self.exp_name=exp_name

    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def train(self):
        self.set_seed(42)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features=torch.load(self.path_text_feat)
        text_features = text_features.to(device)



        projection_model = self.md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,device=device)
        projection_model = projection_model.to(device)

        #DataLoader
        dataloader = self.dataloader(self.ruta_features_train, self.batch_size,self.dataset)
        dataloader_val = self.dataloader(self.ruta_features_val,self.batch_size,self.dataset)
        dataloader_test = self.dataloader(self.ruta_features_test, self.batch_size,self.dataset)

        optimizer,scheduler = self.build_optimizer(projection_model,'sgd',self.lr,self.momentum,self.version)
        acc_best = 0
        counter = 0
        for epoch in range(self.num_epochs):
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


                loss, acc,_ = projection_model(description_embeddings, image_features, text_features, self.weight_Clip,target_index,self.t)

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
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
                                                 self.weight_Clip,target_index_val,self.t)

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
                print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print('Val loss: {:.4f},Val acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
                print(f"Time for epoch [{total_time}]")
                if epoch_acc_val > acc_best:
                    print('Save model')
                    acc_best = epoch_acc_val
                    counter = 0

                    # Create a directory for each training session based on the unique identifier
                    os.makedirs(f'Best/{self.exp_name}/training_{unique_id}', exist_ok=True)
                    # Save the model parameters within the corresponding directory
                    model_params_path = f'Best/{self.exp_name}/training_{unique_id}/best_model_params_{self.num_layers}_{self.hidden_dim}.pth'
                    torch.save(projection_model.state_dict(), model_params_path)

                else:
                    counter = counter + 1
                    print("The acc don't increase")

            if epoch==(self.num_epochs-1) or counter >= self.patience:
                projection_model = self.md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
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
                                                                     self.weight_Clip, target_index_test, self.t)

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
            if counter >= self.patience:
                print(f'Validation acc has not improved for {self.patience} epochs. Stopping training.')
                break
    def prueba_model(self,model_params_path):# to calculate the acc in test for a saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_text_feat)
        text_features = text_features.to(device)

        
        dataloader_test = self.dataloader(self.ruta_features_test, self.batch_size,self.dataset)


        projection_model = self.md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
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
                                                               self.weight_Clip, target_index_test, self.t)

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


    def prueba_model_top_3(self,model_params_path):  # to calculate the acc in test for a saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_text_feat)
        text_features = text_features.to(device)

        dataloader_test = self.dataloader(self.ruta_features_test, self.batch_size,self.dataset)

        projection_model = self.md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
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
                                                                   self.weight_Clip, target_index_test, self.t)


                running_corrects_test += float(acc_test)

        epoch_acc_test = (running_corrects_test / size_test) * 100
        print('Test acc Top 3: {:.4f}'.format(epoch_acc_test))




    #model_params_path = '../../../models/CATALOG_finetuning_Base_Serengeti.pth'
    #prueba_model_top_3(model_params_path)
    #prueba_model(model_params_path)
    #wandb.login(key="282780c770de0083eddfa3c56402f555ee60e108")
    #wandb.init(
    #    project="Train_CLIP_V1_in_domain_Serengeti",
    #)
    #train()
    #wandb.finish()