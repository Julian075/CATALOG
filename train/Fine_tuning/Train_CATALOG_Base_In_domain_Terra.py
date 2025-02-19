import os
import torch
import time
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random

import wandb


class CATALOG_base_In_domain_terra:
    def __init__(self, model, Dataset, Dataloader, version, build_optimizer):
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
        self.path_prompts_D = None

        self.weight_Clip = None
        self.num_epochs = None
        self.batch_size = None
        self.num_layers = None
        self.dropout = None
        self.hidden_dim = None
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
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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

        # Get your DataLoader
        if self.wnb == 1:  # randon search and monte carlo partition activate
            dataset_D = self.path_features_D
            dataloader = self.dataloader(dataset_D['train'], self.batch_size, self.dataset)
            dataloader_cis_val = self.dataloader(dataset_D['val'], self.batch_size, self.dataset)
        else:
            dataset_D = torch.load(self.path_features_D)

            dataloader = self.dataloader(dataset_D['train'], self.batch_size, self.dataset)
            dataloader_cis_val = self.dataloader(dataset_D['cis_val'], self.batch_size, self.dataset)
            dataloader_trans_val = self.dataloader(dataset_D['trans_val'], self.batch_size, self.dataset)

        if test:
            dataloader_cis_test = self.dataloader(dataset_D['cis_test'], self.batch_size, self.dataset)
            dataloader_trans_test = self.dataloader(dataset_D['trans_test'], self.batch_size, self.dataset)



        #Configurate optimazer for training
        optimizer,scheduler = self.build_optimizer(projection_model,'sgd',self.lr,self.momentum, self.version)
        acc_best = 0 #Variable to check the best model
        counter = 0  #Variable to verify the number of epoch without an improve in the val acc
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
                size+=len(image_features)
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

            if self.wnb == 1:
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})

            # validation
            projection_model.eval()

            running_loss_cis_val = 0
            running_corrects_cis_val = 0.0
            size_cis_val=0

            running_loss_trans_val = 0
            running_corrects_trans_val = 0.0
            size_trans_val = 0
            
            with torch.no_grad():
                for batch_cis_val in dataloader_cis_val:
                    image_features_cis_val, description_embeddings_cis_val, target_index_cis_val = batch_cis_val
                    size_cis_val+=len(image_features_cis_val)
                    image_features_cis_val = image_features_cis_val.to(device)
                    description_embeddings_cis_val = description_embeddings_cis_val.to(device)



                    loss_val, acc_val,_ = projection_model(description_embeddings_cis_val, image_features_cis_val, text_features,
                                                 self.weight_Clip,target_index_cis_val,self.t,self.sup_loss)

                    running_loss_cis_val += loss_val.item()
                    running_corrects_cis_val += float(acc_val)

            epoch_loss_cis_val = running_loss_cis_val / len(dataloader_cis_val)
            epoch_acc_cis_val = (running_corrects_cis_val / size_cis_val) * 100

            if self.wnb == 1:
                wandb.log({"epoch_loss_val": epoch_loss_cis_val, "epoch_acc_val": epoch_acc_cis_val})

            if self.wnb == 0:
                with torch.no_grad():
                    for batch_trans_val in dataloader_trans_val:
                        image_features_trans_val, description_embeddings_trans_val, target_index_trans_val = batch_trans_val
                        size_trans_val+=len(image_features_trans_val)
                        image_features_trans_val = image_features_trans_val.to(device)
                        description_embeddings_trans_val = description_embeddings_trans_val.to(device)



                        loss_val, acc_val,_ = projection_model(description_embeddings_trans_val, image_features_trans_val, text_features,
                                                     self.weight_Clip,target_index_trans_val,self.t,self.sup_loss)

                        running_loss_trans_val += loss_val.item()
                        running_corrects_trans_val += float(acc_val)

                epoch_loss_trans_val = running_loss_trans_val / len(dataloader_trans_val)
                epoch_acc_trans_val = (running_corrects_trans_val / size_trans_val) * 100


            time_end = time.time()
            total_time = time_end - time_in
            # Print the loss at every nth epoch
            print_every = 1
            if epoch % print_every == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print('Val Cis loss: {:.4f},Val Cis acc: {:.4f}'.format(epoch_loss_cis_val, epoch_acc_cis_val))
                if self.wnb == 0:
                    print('Val Trans loss: {:.4f},Val acc: {:.4f}'.format(epoch_loss_trans_val, epoch_acc_trans_val))
                print(f"Time for epoch [{total_time}]")
                
                if epoch_acc_cis_val <1 or np.isnan(epoch_acc_cis_val):
                    print(f'💀 This model is officially trash. Accuracy: {epoch_acc_cis_val}. Let’s not waste more compute. Training stopped.')
                    break
                
                if epoch_acc_cis_val > acc_best:
                    print('Save model')
                    acc_best = epoch_acc_cis_val
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
                    projection_model = self.md.LLaVA_CLIP_long(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,device=device)
                    projection_model.load_state_dict(torch.load(model_params_path))
                    projection_model = projection_model.to(device)
                    projection_model.eval()

                    running_loss_cis_test = 0
                    running_corrects_cis_test = 0.0
                    size_cis_test = 0

                    running_loss_trans_test = 0
                    running_corrects_trans_test = 0.0
                    size_trans_test = 0

                    # Variables to calculate the confusion matrix
                    all_preds_cis = []
                    all_labels_cis = []
                    all_preds_trans = []
                    all_labels_trans = []


                    with torch.no_grad():

                        for batch_cis_test in dataloader_cis_test:
                            image_features_cis_test, description_embeddings_cis_test, target_index_cis_test = batch_cis_test
                            size_cis_test += len(image_features_cis_test)
                            image_features_cis_test = image_features_cis_test.to(device)
                            description_embeddings_cis_test = description_embeddings_cis_test.to(device)

                            loss_cis_test, acc_cis_test,preds_cis_test = projection_model(description_embeddings_cis_test,
                                                                         image_features_cis_test, text_features,
                                                                         self.weight_Clip, target_index_cis_test, self.t,self.sup_loss)

                            # Save predictions and targets
                            all_preds_cis.extend(preds_cis_test.cpu().numpy())
                            all_labels_cis.extend(target_index_cis_test.cpu().numpy())

                            running_loss_cis_test += loss_cis_test.item()
                            running_corrects_cis_test += float(acc_cis_test)

                        for batch_trans_test in dataloader_trans_test:
                            image_features_trans_test, description_embeddings_trans_test, target_index_trans_test = batch_trans_test
                            size_trans_test += len(image_features_trans_test)
                            image_features_trans_test = image_features_trans_test.to(device)
                            description_embeddings_trans_test = description_embeddings_trans_test.to(device)

                            loss_trans_test, acc_trans_test,preds_trans_test = projection_model(description_embeddings_trans_test,
                                                                             image_features_trans_test, text_features,
                                                                             self.weight_Clip, target_index_trans_test, self.t,self.sup_loss)

                            # Save predictions and targets
                            all_preds_trans.extend(preds_trans_test.cpu().numpy())
                            all_labels_trans.extend(target_index_trans_test.cpu().numpy())

                            running_loss_trans_test += loss_trans_test.item()
                            running_corrects_trans_test += float(acc_trans_test)

                    epoch_loss_cis_test = running_loss_cis_test / len(dataloader_cis_test)
                    epoch_acc_cis_test = (running_corrects_cis_test / size_cis_test) * 100

                    epoch_loss_trans_test = running_loss_trans_test / len(dataloader_trans_test)
                    epoch_acc_trans_test = (running_corrects_trans_test / size_trans_test) * 100

                    if self.wnb == 1:
                        wandb.log({"epoch_loss_cis_test": epoch_loss_cis_test, "epoch_acc_cis_test": epoch_acc_cis_test,
                                   "epoch_loss_trans_test": epoch_loss_trans_test, "epoch_acc_trans_test": epoch_acc_trans_test})

                    print('Cis Test loss: {:.4f}, Cis Test acc: {:.4f}'.format(epoch_loss_cis_test, epoch_acc_cis_test))
                    print('Trans Test loss: {:.4f}, Trans Test acc: {:.4f}'.format(epoch_loss_trans_test, epoch_acc_trans_test))


                    # Calculate the confusion matrix
                    conf_matrix_cis = confusion_matrix(all_labels_cis, all_preds_cis)
                    df_conf_matrix_cis = pd.DataFrame(conf_matrix_cis)
                    df_conf_matrix_cis.to_csv('conf_matrix_cis_serengeti.csv', index=False)
                    conf_matrix_trans = confusion_matrix(all_labels_trans, all_preds_trans)
                    df_conf_matrix_trans = pd.DataFrame(conf_matrix_trans)
                    df_conf_matrix_trans.to_csv('conf_matrix_trans_serengeti.csv', index=False)

            # Check early stopping condition
            if counter >= self.patience:
                print(f'Validation acc has not improved for {self.patience} epochs. Stopping training.')
                break
        if test:
            return epoch_loss_cis_test, epoch_acc_cis_test, epoch_loss_trans_test, epoch_acc_trans_test
        elif test==0 and self.wnb==1:
            return acc_best
        else:
            return None
    def prueba_model(self,model_params_path):# to calculate the acc in test for a saved model

        dataset_D = torch.load(self.path_features_D)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_prompts_D)
        text_features = text_features.to(device)

        dataloader_cis_test = self.dataloader(dataset_D['cis_test'], self.batch_size, self.dataset)
        dataloader_trans_test = self.dataloader(dataset_D['trans_test'], self.batch_size, self.dataset)


        projection_model = self.md.LLaVA_CLIP_long(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,device=device)
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()

        running_loss_cis_test = 0
        running_corrects_cis_test = 0.0
        size_cis_test = 0

        running_loss_trans_test = 0
        running_corrects_trans_test = 0.0
        size_trans_test = 0

        # Variables to calculate confusion matrix
        all_preds_cis = []
        all_labels_cis = []
        all_preds_trans = []
        all_labels_trans = []

        with torch.no_grad():
            for batch_cis_test in dataloader_cis_test:
                image_features_cis_test, description_embeddings_cis_test, target_index_cis_test = batch_cis_test
                size_cis_test += len(image_features_cis_test)
                image_features_cis_test = image_features_cis_test.to(device)
                description_embeddings_cis_test = description_embeddings_cis_test.to(device)

                loss_cis_test, acc_cis_test,preds_cis_test  = projection_model(description_embeddings_cis_test,
                                                               image_features_cis_test, text_features,
                                                               self.weight_Clip, target_index_cis_test, self.t,self.sup_loss)

                all_preds_cis.extend(preds_cis_test.cpu().numpy())
                all_labels_cis.extend(target_index_cis_test.cpu().numpy())

                running_loss_cis_test += loss_cis_test.item()
                running_corrects_cis_test += float(acc_cis_test)
            for batch_trans_test in dataloader_trans_test:
                image_features_trans_test, description_embeddings_trans_test, target_index_trans_test = batch_trans_test
                size_trans_test += len(image_features_trans_test)
                image_features_trans_test = image_features_trans_test.to(device)
                description_embeddings_trans_test = description_embeddings_trans_test.to(device)

                loss_trans_test, acc_trans_test,preds_trans_test  = projection_model(description_embeddings_trans_test,
                                                                   image_features_trans_test, text_features,
                                                                   self.weight_Clip, target_index_trans_test, self.t,self.sup_loss)

                all_preds_trans.extend(preds_trans_test.cpu().numpy())
                all_labels_trans.extend(target_index_trans_test.cpu().numpy())

                running_loss_trans_test += loss_trans_test.item()
                running_corrects_trans_test += float(acc_trans_test)


        epoch_loss_cis_test = running_loss_cis_test / len(dataloader_cis_test)
        epoch_acc_cis_test = (running_corrects_cis_test / size_cis_test) * 100
        epoch_loss_trans_test = running_loss_trans_test / len(dataloader_trans_test)
        epoch_acc_trans_test = (running_corrects_trans_test / size_trans_test) * 100
        print('Cis Test loss: {:.4f}, Cis Test acc: {:.4f}'.format(epoch_loss_cis_test, epoch_acc_cis_test))
        print('Trans Test loss: {:.4f}, Trans Test acc: {:.4f}'.format(epoch_loss_trans_test, epoch_acc_trans_test))

        # Calculate confusion matrix
        conf_matrix_cis = confusion_matrix(all_labels_cis, all_preds_cis)
        df_conf_matrix_cis = pd.DataFrame(conf_matrix_cis)
        df_conf_matrix_cis.to_csv('conf_matrix_cis_serengeti.csv', index=False)

        conf_matrix_trans = confusion_matrix(all_labels_trans, all_preds_trans)
        df_conf_matrix_trans = pd.DataFrame(conf_matrix_trans)
        df_conf_matrix_trans.to_csv('conf_matrix_trans_serengeti.csv', index=False)

        print("Confusion Matrix for Cis Test:")
        print(conf_matrix_cis)
        print("\nClassification Report for Cis Test:")
        print(classification_report(all_labels_cis, all_preds_cis))

        print("Confusion Matrix for Trans Test:")
        print(conf_matrix_trans)
        print("\nClassification Report for Trans Test:")
        print(classification_report(all_labels_trans, all_preds_trans))