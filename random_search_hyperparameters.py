import os
import csv
import wandb
from feature_extraction.Monte_carlo_partition import monte_carlo_partition
import numpy as np


def wandb_train(model, model_version,train_type,path_features, seeds, config=None):
    with wandb.init(config=config):
        config = wandb.config

        weight_clip = config.weight_Clip
        num_epochs = config.num_epochs
        batch_size = config.batch_size
        num_layers = config.num_layers
        dropout = config.dropout
        hidden_dim = config.hidden_dim
        learning_rate = config.lr
        temperature = config.t
        momentum = config.momentum

        results_val_seeds = []
        for seed in seeds:


            if train_type == 'Out_domain':
                features_D = [monte_carlo_partition(model_version, path_features[0][0], seed), path_features[0][1]]
                features_S = [monte_carlo_partition(model_version, path_features[1][1], seed), path_features[1][1]]
                features = [features_D, features_S]


            elif train_type == 'In_domain':
                features = [[monte_carlo_partition(model_version, path_features[0][0], seed), path_features[0][1]]]

            if model_version == 'Base':
                if train_type == 'Out_Domain':
                    model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                         num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                         t=temperature,
                                         momentum=momentum, patience=5, path_features_D=features[0][0],
                                         path_propmts_D=features[0][1], path_features_S=features[1][0],
                                         path_propmts_S=features[1][1], exp_name=f'{seed}_{model_version}_{train_type}', wnb=1)

            best_acc_val=model.train(seed=seed, test=0)
            results_val_seeds.append(best_acc_val)


        avg_acc_val=np.mean(results_val_seeds)
        std_acc_val=np.std(results_val_seeds)
        results_file='random_search_results.csv'
        results_exist = os.path.isfile(results_file)
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Escribir el encabezado solo si el archivo no existe
            if not results_exist:
                writer.writerow([
                    "avg_acc_val","std_acc_val", "weight_clip", "num_epochs", "batch_size",
                    "num_layers", "dropout", "hidden_dim", "learning_rate", "temperature", "momentum"
                ])
            # Escribir la configuración y el resultado
            writer.writerow([
                avg_acc_val,std_acc_val, weight_clip, num_epochs, batch_size,
                num_layers, dropout, hidden_dim, learning_rate, temperature, momentum
            ])
def random_search(path_features,train_type, model_version,model, name_exp, name_project, seeds):

        token =os.getenv("WandB_TOKE")
        wandb.login(key=token)
        sweep_config = {
                        'method': 'random', 'metric': {'goal': 'maximize','name': 'epoch_acc_val' },
                        'name': name_exp,
                        'parameters': {
                            'batch_size': { 'distribution': 'categorical', 'values': [2 ** i for i in range(2, 9)] },
                            'dropout': {'distribution': 'uniform','min': 0.1,'max': 0.5 },
                            'hidden_dim': {'distribution': 'categorical', 'values': [2**i for i in range(9, 12)]},
                            'lr': {'distribution': 'uniform', 'min': 1e-5, 'max': 0.1 },
                            'momentum': {'distribution': 'uniform','min': 0.8, 'max': 0.99 },
                            'num_epochs': {'distribution': 'int_uniform', 'min': 1, 'max': 200 },
                            'num_layers': {'distribution': 'int_uniform','min': 1,'max': 7 },
                            't': {'distribution': 'log_uniform', 'min': 0.01,'max': 1},
                            'weight_Clip': { 'distribution': 'uniform','min': 0.4,'max': 0.8}
                        }
                    }

        sweep_id = wandb.sweep(sweep_config, project=name_project)


        wandb.agent(sweep_id, function=lambda: wandb_train(model, model_version, train_type, path_features, seeds), count=100)