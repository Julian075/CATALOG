import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.optim as optim
import time
import argparse
import datetime
import wandb
import random



class CLIP_MLP_train:
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
        self.path_features_S = None
        self.path_prompts_S =  None

        self.num_epochs =  None
        self.batch_size =  None
        self.num_layers =  None
        self.dropout =     None
        self.hidden_dim =  None
        self.lr = None
        self.momentum = None
        self.patience = None
        self.exp_name = None

    def set_parameters(self, num_epochs, batch_size, num_layers, dropout, hidden_dim, lr, momentum, patience,path_features_D, path_prompts_D, path_features_S=None, path_prompts_S=None, exp_name="MLP",wnb=0):
        self.wnb = wnb
        self.path_features_D = path_features_D
        self.path_prompts_D = path_prompts_D
        self.path_features_S = path_features_S
        self.path_prompts_S = path_prompts_S

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.exp_name = exp_name

    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train_out_domain(self, seed=42, test=1):
        self.set_seed(seed)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        text_features = torch.load(self.path_prompts_D)
        text_features = text_features.to(device)
        text_features2 = torch.load(self.path_prompts_S)
        text_features2 = text_features2.to(device)

        projection_model=self.md.CLIP_MLP(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout)
        projection_model = projection_model.to(device)

        # Get your DataLoader
        if self.wnb == 1:  # randon search and monte carlo partition activate
            dataset_D = self.path_features_D
        else:
            dataset_D = torch.load(self.path_features_D)
        dataset_S = torch.load(self.path_features_S)
        dataloader = self.dataloader(dataset_D['train'], self.batch_size, self.dataset)
        dataloader_val = self.dataloader(dataset_D['val'], self.batch_size, self.dataset)
        if test:
            dataloader_cis_test = self.dataloader(dataset_S['cis_test'], self.batch_size, self.dataset)
            dataloader_trans_test = self.dataloader(dataset_S['trans_test'], self.batch_size, self.dataset)

        # Configurate optimazer for training
        optimizer, scheduler = self.build_optimizer(projection_model, 'sgd', self.lr, self.momentum, self.version,
                                                    self.en_att)
        acc_best = -float('inf')  # Variable to check the best model
        counter = 0  # Variable to verify the number of epoch without an improvement in the val acc
        for epoch in range(self.num_epochs):
            print(epoch)
            # Training
            projection_model.train()
            time_in = time.time()
            running_loss = 0.0
            running_corrects = 0.0
            size = 0
            for batch in dataloader:
                image_features, description_embeddings, target_index = batch
                size += len(image_features)
                image_features = image_features.to(device)
                description_embeddings = description_embeddings.to(device)

                loss, acc, _ = projection_model(description_embeddings, image_features, text_features, self.weight_Clip,
                                                target_index, self.t)

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
            size_val = 0

            with torch.no_grad():
                for batch_val in dataloader_val:
                    image_features_val, description_embeddings_val, target_index_val = batch_val
                    size_val += len(image_features_val)
                    image_features_val = image_features_val.to(device)
                    description_embeddings_val = description_embeddings_val.to(device)

                    loss_val, acc_val, _ = projection_model(description_embeddings_val, image_features_val,
                                                            text_features,
                                                            self.weight_Clip, target_index_val, self.t)

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

            if epoch == (self.num_epochs - 1) or counter >= self.patience:
                if test:
                    projection_model = self.md.LLaVA_CLIP(hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                                                          dropout=self.dropout, en_att=self.en_att, device=device)
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

                            loss_cis_test, acc_cis_test, preds_cis_test = projection_model(
                                description_embeddings_cis_test,
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

                            loss_trans_test, acc_trans_test, preds_trans_test = projection_model(
                                description_embeddings_trans_test,
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

                    if self.wnb:
                        wandb.log({"loss_cis_test": epoch_loss_cis_test, "acc_cis_test": epoch_acc_cis_test,
                                   "loss_trans_test": epoch_loss_trans_test, "acc_trans_test": epoch_acc_trans_test})

                    print('Cis Test loss: {:.4f}, Cis Test acc: {:.4f}'.format(epoch_loss_cis_test, epoch_acc_cis_test))
                    print('Trans Test loss: {:.4f}, Trans Test acc: {:.4f}'.format(epoch_loss_trans_test,
                                                                                   epoch_acc_trans_test))

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
        if test:
            return epoch_loss_cis_test, epoch_acc_cis_test, epoch_loss_trans_test, epoch_acc_trans_test
        elif test == 0 and self.wnb == 1:
            return acc_best
        else:
            return None





class CustomDataset(Dataset):
    def __init__(self,json_path):
        self.root_dir = json_path
        self.samples = self._load_samples()


    def _load_samples(self):
        samples=[]
        data_dict=torch.load(self.root_dir)
        for key in data_dict.keys():
            samples.append([data_dict[key]['image_features'][0],data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, target_index= self.samples[idx]
        return image_features, target_index


# Define your DataLoader
def get_dataloader(root_dir, batch_size):
    dataset = CustomDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":

    # Crear un objeto ArgumentParser
    parser = argparse.ArgumentParser(description='Descripción del programa')

    # Agregar los argumentos
    parser.add_argument('--ruta_features_train', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_16_MLP_train.pt",
                        help='Training path')
    parser.add_argument('--ruta_features_val1', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_16_MLP_cis_val.pt",
                        help='Validation path 1')
    parser.add_argument('--ruta_features_val2', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_16_MLP_trans_val.pt",
                        help='Validation path 2')
    parser.add_argument('--ruta_features_test1', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_16_MLP_cis_test.pt",
                        help='Test path 1')
    parser.add_argument('--ruta_features_test2', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_16_MLP_trans_test.pt",
                        help='Test path 2')
    # parser.add_argument('--num_epochs', type=int, default=30, help='Número de épocas')
    # parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--pretrained', type=int, default=0, help='pretrained ')
    # parser.add_argument('--num_layers', type=int, default=1, help='num_layers ')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout ')
    # parser.add_argument('--hidden_dim', type=int, default=512 * 4, help='hidden_dim ')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate ')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum ')
    # parser.add_argument('--patience', type=int, default=10, help='patience ')

    # Parsear los argumentos
    args = parser.parse_args()

    # Acceder a los valores de los argumentos
    ruta_features_train = args.ruta_features_train
    ruta_features_val1 = args.ruta_features_val1
    ruta_features_val2 = args.ruta_features_val2
    ruta_features_test1 = args.ruta_features_test1
    ruta_features_test2 = args.ruta_features_test2
    # num_epochs = args.num_epochs
    # batch_size = args.batch_size
    pretrained = args.pretrained
    # num_layers = args.num_layers
    # dropout = args.dropout
    # hidden_dim = args.hidden_dim
    # lr = args.lr
    # momentum = args.momentum
    # patience = args.patience

    # ruta_img_train='D:/Udea/Maestria/Bases_de_datos/eccv_18_all_images_sm/Organizado/categorias/train'
    # ruta_img_val = 'D:/Udea/Maestria/Bases_de_datos/eccv_18_all_images_sm/Organizado/categorias/cis_val'

    wandb.login(key="282780c770de0083eddfa3c56402f555ee60e108")

    # Configurar las settings de W&B
    sweep_config = {
        'method': 'random',
        'name': 'model_MLP_16',
    }
    metric = {
        'name': 'loss_trans_val',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'opt': {
            'values': ['sgd']
        },
        't': {
            'values': [0.1, 0.01, 0.001]
        },
    }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'momentum': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.99
        },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 4,
            'max': 256,
        },
        'hidden_dim': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1024,
            'max': 2048,
        },
        'dropout': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5,
        },
        'num_layers': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1,
            'max': 20,
        },
        'num_epochs': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'int_uniform',
            'min': 1,
            'max': 100,
        }
    })
    parameters_dict.update({
        'patience': {
            'value': 15}
    })
    sweep_id = wandb.sweep(sweep_config, project="CLIP_MLP")


    # Generate a unique identifier based on the current date and time
    def train(config=None):
        clip_version=16
        with wandb.init(config=config):
            config = wandb.config
            config.name = f"run_num_layers_{config.num_layers}_lr_{config.lr}_dropout_{config.dropout}"
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            dropout = config.dropout
            batch_size = config.batch_size
            opt = config.opt
            lr = config.lr
            momentum = config.momentum
            num_epochs = config.num_epochs
            patience = config.patience
            t = config.t

            unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "cpu:":
                # Get the number of available GPUs
                num_gpus = torch.cuda.device_count()
                # Create a list of devices
                devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
            else:
                # If CUDA is not available, just use CPU
                devices = [torch.device("cpu")]
            print('num_gpus: ', num_gpus, devices)

            text_features = torch.load(f'Text_features_{clip_version}.pt')
            text_features = text_features.to(device)
            #model_clip = md.MLP_CLIP(hidden_dim=hidden_dim,num_layers=num_layers,dropout=dropout,pretrained=pretrained,pretrained_path=pretrained)
            #model_clip = md.CLIP_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=pretrained, pretrained_path=pretrained)
            model_clip = md.model_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=pretrained, pretrained_path=pretrained)
            model_clip = model_clip.to(device)

            # Get your DataLoader
            dataloader = get_dataloader(ruta_features_train, batch_size)
            dataloader_cis_val = get_dataloader(ruta_features_val1, batch_size)
            dataloader_trans_val = get_dataloader(ruta_features_val2, batch_size)
            dataloader_cis_test = get_dataloader(ruta_features_test1, batch_size)
            dataloader_trans_test = get_dataloader(ruta_features_test2, batch_size)

            def build_optimizer(optimizer, learning_rate, momentum):
                # Definir los parámetros y configuraciones de optimización para cada conjunto de parámetros
                params1 = {"params": model_clip.parameters(), "lr": learning_rate,
                           "momentum": momentum}

                # Inicializar el optimizador con todos los conjuntos de parámetros
                if optimizer == "sgd":
                    optimizer = optim.SGD([params1], lr=learning_rate, momentum=momentum)
                elif optimizer == "adam":
                    optimizer = optim.Adam([params1], lr=learning_rate)

                return optimizer

            optimizer = build_optimizer(opt, lr, momentum)  # optim.SGD([params1, params2,params3])

            acc_best = 0
            counter = 0
            for epoch in range(num_epochs):
                print(epoch)
                # Training
                model_clip.train()
                time_in = time.time()
                running_loss = 0.0
                running_corrects = 0.0
                size = 0
                for batch in dataloader:
                    image_features, target_index = batch
                    size += len(image_features)
                    image_features = image_features.to(device)


                    loss, acc = model_clip( image_features, text_features, target_index, t)

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # Update the parameters
                    optimizer.step()
                    # Zero the gradients
                    optimizer.zero_grad()

                    # statistics Train
                    running_loss += loss.item()
                    running_corrects += float(acc)

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = (running_corrects / size) * 100

                # validation
                model_clip.eval()

                running_loss_cis_val = 0
                running_corrects_cis_val = 0.0
                size_cis_val = 0

                running_loss_trans_val = 0
                running_corrects_trans_val = 0.0
                size_trans_val = 0
                with torch.no_grad():

                    for batch_cis_val in dataloader_cis_val:
                        image_features_cis_val, target_index_cis_val = batch_cis_val
                        size_cis_val += len(image_features_cis_val)
                        image_features_cis_val = image_features_cis_val.to(device)

                        # batch_text_val = text_features.t()[target_index_val]

                        loss_cis_val, acc_cis_val = model_clip(image_features_cis_val, text_features, target_index_cis_val, t)

                        running_loss_cis_val += loss_cis_val.item()
                        running_corrects_cis_val += float(acc_cis_val)

                    for batch_trans_val in dataloader_trans_val:
                        image_features_trans_val, target_index_trans_val = batch_trans_val
                        size_trans_val += len(image_features_trans_val)
                        image_features_trans_val = image_features_trans_val.to(device)

                        # batch_text_val = text_features.t()[target_index_val]

                        loss_trans_val, acc_trans_val = model_clip(image_features_trans_val, text_features,target_index_trans_val, t)

                        running_loss_trans_val += loss_trans_val.item()
                        running_corrects_trans_val += float(acc_trans_val)

                epoch_loss_cis_val = running_loss_cis_val / len(dataloader_cis_val)
                epoch_acc_cis_val = (running_corrects_cis_val / size_cis_val) * 100

                epoch_loss_trans_val = running_loss_trans_val / len(dataloader_trans_val)
                epoch_acc_trans_val = (running_corrects_trans_val / size_trans_val) * 100

                # log metrics to wandb #
                wandb.log({"acc_train": epoch_acc, "loss_train": epoch_loss, "acc_cis_val": epoch_acc_cis_val,
                           "loss_cis_val": epoch_loss_cis_val, "acc_trans_val": epoch_acc_trans_val,
                           "loss_trans_val": epoch_loss_trans_val})

                time_end = time.time()
                tiempo_transcurrido = time_end - time_in
                # Print the loss at every nth epoch
                print_every = 1
                if epoch % print_every == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}]")
                    print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                    print('Cis Val loss: {:.4f}, Cis Val acc: {:.4f}'.format(epoch_loss_cis_val, epoch_acc_cis_val))
                    print('Trans Val loss: {:.4f}, Trans acc: {:.4f}'.format(epoch_loss_trans_val, epoch_acc_trans_val))
                    print(f"Time for epoch [{tiempo_transcurrido}]")
                    if epoch_acc_trans_val > acc_best:
                        print('Save model')
                        acc_best = epoch_acc_trans_val
                        counter = 0

                        # Create a directory for each training session based on the unique identifier
                        os.makedirs(f'training_{unique_id}', exist_ok=True)
                        # Save the model parameters within the corresponding directory
                        model_params_path = f'training_{unique_id}/best_model_params_{num_layers}_{hidden_dim}.pth'
                        torch.save(model_clip.state_dict(), model_params_path)

                    else:
                        counter = counter + 1
                        print("The acc don't increase")

                if epoch == num_epochs or counter >= patience:
                    #model_clip = md.MLP_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,  pretrained=0, pretrained_path="")
                    #model_clip = md.CLIP_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=0, pretrained_path="")
                    model_clip = md.model_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,  pretrained=0, pretrained_path="")
                    model_clip.load_state_dict(torch.load(model_params_path))
                    model_clip = model_clip.to(device)
                    model_clip.eval()

                    running_loss_cis_test = 0
                    running_corrects_cis_test = 0.0
                    size_cis_test = 0

                    running_loss_trans_test = 0
                    running_corrects_trans_test = 0.0
                    size_trans_test = 0
                    with torch.no_grad():

                        for batch_cis_test in dataloader_cis_test:
                            image_features_cis_test, target_index_cis_test = batch_cis_test
                            size_cis_test += len(image_features_cis_test)
                            image_features_cis_test = image_features_cis_test.to(device)

                            # batch_text_val = text_features.t()[target_index_val]

                            loss_cis_test, acc_cis_test = model_clip(image_features_cis_test, text_features, target_index_cis_test, t)

                            running_loss_cis_test += loss_cis_test.item()
                            running_corrects_cis_test += float(acc_cis_test)

                        for batch_trans_test in dataloader_trans_test:
                            image_features_trans_test, target_index_trans_test = batch_trans_test
                            size_trans_test += len(image_features_trans_test)
                            image_features_trans_test = image_features_trans_test.to(device)

                            # batch_text_val = text_features.t()[target_index_val]

                            loss_trans_test, acc_trans_test = model_clip(image_features_trans_test, text_features, target_index_trans_test, t)

                            running_loss_trans_test += loss_trans_test.item()
                            running_corrects_trans_test += float(acc_trans_test)

                    epoch_loss_cis_test = running_loss_cis_test / len(dataloader_cis_test)
                    epoch_acc_cis_test = (running_corrects_cis_test / size_cis_test) * 100

                    epoch_loss_trans_test = running_loss_trans_test / len(dataloader_trans_test)
                    epoch_acc_trans_test = (running_corrects_trans_test / size_trans_test) * 100

                    print('Cis Test loss: {:.4f}, Cis Test acc: {:.4f}'.format(epoch_loss_cis_test, epoch_acc_cis_test))
                    print('Trans Test loss: {:.4f}, Trans Test acc: {:.4f}'.format(epoch_loss_trans_test,
                                                                                   epoch_acc_trans_test))
                    wandb.log({"acc_cis_test": epoch_acc_cis_test, "loss_cis_test": epoch_loss_cis_test,
                               "acc_trans_test": epoch_acc_trans_test, "loss_trans_test": epoch_loss_trans_test})
                # Check early stopping condition
                if counter >= patience:
                    print(f'Validation acc has not improved for {patience} epochs. Stopping training.')
                    break


    wandb.agent(sweep_id, train, count=100)
    wandb.finish()
