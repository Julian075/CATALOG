import torch
from torch.utils.data import Dataset, DataLoader
import CLIP_Mlp as md
import os
import torch.optim as optim
import time
import argparse
import datetime
import wandb


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
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_32_MLP_train.pt",
                        help='Training path')
    parser.add_argument('--ruta_features_val1', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_32_MLP_cis_val.pt",
                        help='Validation path 1')
    parser.add_argument('--ruta_features_val2', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_32_MLP_trans_val.pt",
                        help='Validation path 2')
    parser.add_argument('--ruta_features_test1', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_32_MLP_cis_test.pt",
                        help='Test path 1')
    parser.add_argument('--ruta_features_test2', type=str,
                        default="D:/Udea/Maestria/Experimentos/scripts/Pruebas_Clip/Features_CLIP_32_MLP_trans_test.pt",
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
        'name': 'model_MLP_32',
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
        clip_version=32
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


                    loss, acc = model_clip( image_features, text_features,
                                                 target_index, t)

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
                    #model_clip = md.MLP_CLIP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=0, pretrained_path="")
                    #model_clip = md.CLIP_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=0, pretrained_path="")
                    model_clip = md.model_MLP(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,pretrained=0, pretrained_path="")

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
