import os
import torch
import torch.optim as optim
import time
import argparse
import datetime
from models import CATALOG_Base as md
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random
import wandb






class CATALOG_base:
    def __init__(self, weight_Clip, num_epochs, batch_size, num_layers, dropout, hidden_dim, lr, t, momentum, patience,model, Dataset, Dataloader,version,ruta_features_train,ruta_features_val,ruta_features_test1,ruta_features_test2,path_text_feat1,path_text_feat2,build_optimizer,exp_name):
        self.ruta_features_train  = ruta_features_train  #  "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
        self.ruta_features_val    = ruta_features_val    #  "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
        self.ruta_features_test1  = ruta_features_test1  #  "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
        self.ruta_features_test2  = ruta_features_test2  #  "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
        self.path_text_feat1      = path_text_feat1      # "features/Features_serengeti/standard_features/Text_features_16.pt"
        self.path_text_feat2      = path_text_feat2      #  "features/Features_terra/standard_features/Text_features_16.pt"
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
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def train(self):
        self.set_seed(42)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_text_feat1)
        text_features = text_features.to(device)

        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)



        projection_model = md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,pretrained=0)
        projection_model = projection_model.to(device)

        # Get your DataLoader
        dataloader = self.dataloader(self.ruta_features_train, self.batch_size,self.dataset)
        dataloader_val = self.dataloader(self.ruta_features_val,self.batch_size,self.dataset)
        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size,self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size,self.dataset)

        #Configurate optimazer for training
        optimizer,scheduler = self.build_optimizer(projection_model,'sgd',self.lr,self.momentum,self.version)
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
                    size_val+=len(image_features_val)
                    image_features_val = image_features_val.to(device)
                    description_embeddings_val = description_embeddings_val.to(device)



                    loss_val, acc_val,_ = projection_model(description_embeddings_val, image_features_val, text_features,
                                                 self.weight_Clip,target_index_val,self.t)

                    running_loss_val += loss_val.item()
                    running_corrects_val += float(acc_val)

            epoch_loss_val = running_loss_val / len(dataloader_val)
            epoch_acc_val = (running_corrects_val / size_val) * 100

            wandb.log({"acc_train": epoch_acc, "loss_train": epoch_loss, "acc_val": epoch_acc_val,"loss_val": epoch_loss_val})

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
                projection_model = md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                                 pretrained=0, pretrained_path="")
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
                                                                     image_features_cis_test, text_features2,
                                                                     self.weight_Clip, target_index_cis_test, self.t)

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
                                                                         image_features_trans_test, text_features2,
                                                                         self.weight_Clip, target_index_trans_test, self.t)

                        # Save predictions and targets
                        all_preds_trans.extend(preds_trans_test.cpu().numpy())
                        all_labels_trans.extend(target_index_trans_test.cpu().numpy())

                        running_loss_trans_test += loss_trans_test.item()
                        running_corrects_trans_test += float(acc_trans_test)

                epoch_loss_cis_test = running_loss_cis_test / len(dataloader_cis_test)
                epoch_acc_cis_test = (running_corrects_cis_test / size_cis_test) * 100

                epoch_loss_trans_test = running_loss_trans_test / len(dataloader_trans_test)
                epoch_acc_trans_test = (running_corrects_trans_test / size_trans_test) * 100

                wandb.log({"acc_cis_test": epoch_acc_cis_test, "loss_cis_test": epoch_loss_cis_test,
                           "acc_trans_test": epoch_acc_trans_test, "loss_trans_test": epoch_loss_trans_test})

                print('Cis Test loss: {:.4f}, Cis Test acc: {:.4f}'.format(epoch_loss_cis_test, epoch_acc_cis_test))
                print('Trans Test loss: {:.4f}, Trans Test acc: {:.4f}'.format(epoch_loss_trans_test, epoch_acc_trans_test))

                # Calculate the confusion matrix
                conf_matrix_cis = confusion_matrix(all_labels_cis, all_preds_cis)
                df_conf_matrix_cis = pd.DataFrame(conf_matrix_cis)
                df_conf_matrix_cis.to_csv('conf_matrix_cis_Base.csv', index=False)
                conf_matrix_trans = confusion_matrix(all_labels_trans, all_preds_trans)
                df_conf_matrix_trans = pd.DataFrame(conf_matrix_trans)
                df_conf_matrix_trans.to_csv('conf_matrix_trans_Base.csv', index=False)

            # Check early stopping condition
            if counter >= self.patience:
                print(f'Validation acc has not improved for {self.patience} epochs. Stopping training.')
                break
    def prueba_model(self,model_params_path):# to calculate the acc in test for a saved model

        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)

        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size,self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size,self.dataset)


        projection_model = md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                         pretrained=0, pretrained_path="")
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
                                                               image_features_cis_test, text_features2,
                                                               self.weight_Clip, target_index_cis_test, self.t)

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
                                                                   image_features_trans_test, text_features2,
                                                                   self.weight_Clip, target_index_trans_test, self.t)

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
        df_conf_matrix_cis.to_csv('conf_matrix_cis_Base.csv', index=False)

        conf_matrix_trans = confusion_matrix(all_labels_trans, all_preds_trans)
        df_conf_matrix_trans = pd.DataFrame(conf_matrix_trans)
        df_conf_matrix_trans.to_csv('conf_matrix_trans_Base.csv', index=False)

        print("Confusion Matrix for Cis Test:")
        print(conf_matrix_cis)
        print("\nClassification Report for Cis Test:")
        print(classification_report(all_labels_cis, all_preds_cis))

        print("Confusion Matrix for Trans Test:")
        print(conf_matrix_trans)
        print("\nClassification Report for Trans Test:")
        print(classification_report(all_labels_trans, all_preds_trans))

    def prueba_model_top_3(self,model_params_path):# to calculate the acc in test for a saved model

        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)

        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size,self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size,self.dataset)


        projection_model = md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                         pretrained=0, pretrained_path="")
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()


        running_corrects_cis_test = 0.0
        size_cis_test = 0


        running_corrects_trans_test = 0.0
        size_trans_test = 0


        with torch.no_grad():
            for batch_cis_test in dataloader_cis_test:
                image_features_cis_test, description_embeddings_cis_test, target_index_cis_test = batch_cis_test
                size_cis_test += len(image_features_cis_test)
                image_features_cis_test = image_features_cis_test.to(device)
                description_embeddings_cis_test = description_embeddings_cis_test.to(device)

                acc_top_3_cis  = projection_model.predict_top_3(description_embeddings_cis_test,
                                                               image_features_cis_test, text_features2,
                                                               self.weight_Clip, target_index_cis_test, self.t)

                running_corrects_cis_test += float(acc_top_3_cis)
            for batch_trans_test in dataloader_trans_test:
                image_features_trans_test, description_embeddings_trans_test, target_index_trans_test = batch_trans_test
                size_trans_test += len(image_features_trans_test)
                image_features_trans_test = image_features_trans_test.to(device)
                description_embeddings_trans_test = description_embeddings_trans_test.to(device)

                acc_top_3_trans  = projection_model.predict_top_3(description_embeddings_trans_test,
                                                                   image_features_trans_test, text_features2,
                                                                   self.weight_Clip, target_index_trans_test, self.t)


                running_corrects_trans_test += float(acc_top_3_trans)


        epoch_acc_cis_test = (running_corrects_cis_test / size_cis_test) * 100

        epoch_acc_trans_test = (running_corrects_trans_test / size_trans_test) * 100
        print(' Cis Test acc Top 3: {:.4f}'.format( epoch_acc_cis_test))
        print(' Trans Test acc Top 3: {:.4f}'.format( epoch_acc_trans_test))

    #model_params_path = '../../../models/CATALOG_Base.pth'
    #prueba_model_top_3(model_params_path)
    #prueba_model(model_params_path)
    #train()