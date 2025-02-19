import os
import torch
import time
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random

import wandb



class CATALOG_base_In_domain:
    def __init__(self,model, Dataset,Dataloader, version,build_optimizer):
        #
        self.md = model
        self.dataset = Dataset
        self.dataloader = Dataloader
        self.version = version
        self.build_optimizer = build_optimizer

        # default
        self.wnb = 0
        # data_dict=torch.load(self.root_dir)
        self.path_features_D = None
        self.path_prompts_D =  None

        self.weight_Clip = None
        self.num_epochs =  None
        self.batch_size =  None
        self.num_layers =  None
        self.dropout =     None
        self.hidden_dim =  None
        self.lr = None
        self.t = None
        self.momentum = None
        self.patience = None
        self.exp_name = None

    def set_parameters(self, weight_Clip, num_epochs, batch_size, num_layers, dropout, hidden_dim, lr, t, momentum, patience,path_features_D,path_prompts_D,exp_name,sup_loss=0,wnb=0):

        self.wnb=wnb
        #data_dict=torch.load(self.root_dir)
        self.path_features_D=  path_features_D
        self.path_prompts_D =  path_prompts_D

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
        self.exp_name=exp_name
        self.sup_loss=sup_loss

    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def train(self,seed=1064200250,test=1):
        self.set_seed(seed)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_prompts_D)
        text_features = text_features.to(device)


        projection_model = self.md.LLaVA_CLIP_long(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,device=device)
        projection_model = projection_model.to(device)

        #DataLoader
        # Get your DataLoader
        if self.wnb == 1:  # randon search and monte carlo partition activate
            dataset_D = self.path_features_D
        else:
            dataset_D = torch.load(self.path_features_D)

        dataloader = self.dataloader(dataset_D['train'], self.batch_size, self.dataset)
        dataloader_val = self.dataloader(dataset_D['val'], self.batch_size, self.dataset)

        if test:
            dataloader_test = self.dataloader(dataset_D['test'], self.batch_size, self.dataset)

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


                loss, acc,_ = projection_model(description_embeddings, image_features, text_features, self.weight_Clip,target_index,self.t,self.sup_loss)

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

            if self.wnb:
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})

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
                                                 self.weight_Clip,target_index_val,self.t,self.sup_loss)

                    running_loss_val += loss_val.item()
                    running_corrects_val += float(acc_val)

                

            epoch_loss_val = running_loss_val / len(dataloader_val)
            epoch_acc_val = (running_corrects_val / size_val) * 100

            if self.wnb:
                wandb.log({"val_loss": epoch_loss_val, "val_acc": epoch_acc_val})


            time_end = time.time()
            total_time = time_end - time_in
            # Print the loss at every nth epoch
            print_every = 1
            if epoch % print_every == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print('Val loss: {:.4f},Val acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
                print(f"Time for epoch [{total_time}]")
                if epoch_acc_val <1 or np.isnan(epoch_acc_val):
                    print(f'💀 This model is officially trash. Accuracy: {epoch_acc_val}. Let’s not waste more compute. Training stopped.')
                    if epoch==0:
                        test=0
                    break
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
                if test:
                    projection_model = self.md.LLaVA_CLIP_long(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
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
                                                                         self.weight_Clip, target_index_test, self.t,self.sup_loss)

                            all_preds.extend(preds_test.cpu().numpy())
                            all_labels.extend(target_index_test.cpu().numpy())

                            running_loss_test += loss_test.item()
                            running_corrects_test += float(acc_test)


                    epoch_loss_test = running_loss_test / len(dataloader_test)
                    epoch_acc_test = (running_corrects_test / size_test) * 100

                    if self.wnb:
                        wandb.log({"loss_test": epoch_loss_test, "acc_test": epoch_acc_test})

                    print('Test loss: {:.4f},Test acc: {:.4f}'.format(epoch_loss_test, epoch_acc_test))

                    # Calculate confusion matrix
                    conf_matrix = confusion_matrix(all_labels, all_preds)
                    df_conf_matrix = pd.DataFrame(conf_matrix)
                    df_conf_matrix.to_csv('conf_matrix_Base.csv', index=False)

            # Check early stopping condition
            if counter >= self.patience:
                print(f'Validation acc has not improved for {self.patience} epochs. Stopping training.')
                break
        if test:
            return epoch_loss_test, epoch_acc_test
        elif test==0 and self.wnb==1:
            return acc_best
        else:
            return None
    def prueba_model(self,model_params_path):# to calculate the acc in test for a saved model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_prompts_D)
        text_features = text_features.to(device)

        dataset_D = torch.load(self.path_features_D)
        dataloader_test = self.dataloader(dataset_D['test'], self.batch_size, self.dataset)


        projection_model = self.md.LLaVA_CLIP_long(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
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
                                                               self.weight_Clip, target_index_test, self.t,self.sup_loss)

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
        df_conf_matrix.to_csv('conf_matrix_Base.csv')

        print("Confusion Matrix for Test:")
        print(conf_matrix)
        print("\nClassification Report for Test:")
        print(classification_report(all_labels, all_preds))




