import os
import csv
import wandb
from feature_extraction.Monte_carlo_partition import monte_carlo_partition
import numpy as np


def wandb_train(model, model_version,train_type,path_features, name_exp, seeds, config=None):
    with wandb.init(config=config,name=name_exp):
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
                features_D = [monte_carlo_partition(path_features[0][0], seed), path_features[0][1]]
                features_S = [path_features[1][0], path_features[1][1]]
                features = [features_D, features_S]


            elif train_type == 'In_domain':
                features = [[monte_carlo_partition( path_features[0][0], seed), path_features[0][1]]]

            if model_version == 'Base':
                if train_type == 'Out_domain':
                    model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                         num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                         t=temperature,momentum=momentum, patience=5, path_features_D=features[0][0],
                                         path_prompts_D=features[0][1], path_features_S=features[1][0],
                                         path_prompts_S=features[1][1], exp_name=f'{seed}_{model_version}_{train_type}', wnb=1)

            if model_version == 'Fine_tuning':
                if train_type == 'In_domain':
                    model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                         num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                         t=temperature,momentum=momentum, patience=5, path_features_D=features[0][0],
                                         path_prompts_D=features[0][1], exp_name=f'{seed}_{model_version}_{train_type}', wnb=1)

            best_acc_val=model.train(seed=seed, test=0)
            results_val_seeds.append(best_acc_val)


        avg_acc_val=np.mean(results_val_seeds)
        std_acc_val=np.std(results_val_seeds)
        results_file=f'random_search_results_{name_exp}.csv'
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

        token ="282780c770de0083eddfa3c56402f555ee60e108"#os.getenv("WandB_TOKE")
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
                            't': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 1},
                            'weight_Clip': { 'distribution': 'uniform','min': 0.4,'max': 0.8}
                        }
                    }

        sweep_id = wandb.sweep(sweep_config, project=name_project)


        wandb.agent(sweep_id, function=lambda: wandb_train(model, model_version, train_type, path_features,name_exp, seeds), count=100)


def random_search2(path_features, train_type, model_version, model, name_exp, name_project, seeds):
    token = "282780c770de0083eddfa3c56402f555ee60e108"  # os.getenv("WandB_TOKE")
    wandb.login(key=token)
    sweep_config = {
        'method': 'random', 'metric': {'goal': 'maximize', 'name': 'epoch_acc_val'},
        'name': name_exp,
        'parameters': {
            'batch_size': {'distribution': 'categorical', 'values': [2 ** i for i in range(2, 9)]},
            'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'hidden_dim': {'distribution': 'uniform', 'min': 512, 'max': 1400},
            'lr': {'distribution': 'uniform', 'min': 1e-3, 'max': 0.1},
            'momentum': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'num_epochs': {'distribution': 'int_uniform', 'min': 1, 'max': 200},
            'num_layers': {'value': 1},
            't': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 1},
            'weight_Clip': {'distribution': 'uniform', 'min': 0.4, 'max': 0.6}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=name_project)

    wandb.agent(sweep_id,
                function=lambda: wandb_train(model, model_version, train_type, path_features, name_exp, seeds),
                count=100)

def test_best_model(path_features,train_type, model_version,model, name_exp,config, seeds):
    weight_clip = config['weight_Clip']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_layers = config['num_layers']
    dropout = config['dropout']
    hidden_dim = config['hidden_dim']
    learning_rate = config['lr']
    temperature = config['t']
    momentum = config['momentum']

    results_cis_test_seeds = []
    results_trans_test_seeds = []

    results_temporal = f"results_test_random_search_temporal_{name_exp}.csv"
    results_exist_temp = os.path.isfile(results_temporal)
    existing_seeds = set()

    # Read existing seeds from CSV if the file exists
    if results_exist_temp:
        with open(results_temporal, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                try:
                    seed = int(row[0])  # Intenta convertirlo a entero
                    existing_seeds.add(seed)
                except ValueError:
                    pass  # Ignora si no es un número

    for seed in seeds:
        if seed in existing_seeds:
            print(f"Skipping seed {seed}, already tested.")
            continue

        if train_type == 'Out_domain':
            features_D = [path_features[0][0], path_features[0][1]]
            features_S = [path_features[1][0], path_features[1][1]]
            features = [features_D, features_S]
        elif train_type == 'In_domain':
            features = [[path_features[0][0], path_features[0][1]]]

        if model_version == 'Base' and train_type == 'Out_domain':
            model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                 num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                 t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                 path_prompts_D=features[0][1], path_features_S=features[1][0],
                                 path_prompts_S=features[1][1], exp_name=f'{seed}_{model_version}_{name_exp}',
                                 wnb=0)

        elif model_version == 'Fine_tuning' and train_type == 'In_domain':
            model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                 num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                 t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                 path_prompts_D=features[0][1], exp_name=f'{seed}_{model_version}_{name_exp}',
                                 wnb=0)

        epoch_loss_cis_test, epoch_acc_cis_test, epoch_loss_trans_test, epoch_acc_trans_test = model.train(seed=seed,
                                                                                                           test=1)
        results_cis_test_seeds.append(epoch_acc_cis_test)
        results_trans_test_seeds.append(epoch_acc_trans_test)

        # Append new results to CSV
        with open(results_temporal, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not results_exist_temp or os.stat(results_temporal).st_size == 0:
                writer.writerow([
                    "seed", "acc_cis_test", "acc_trans_test", "weight_clip", "num_epochs", "batch_size",
                    "num_layers", "dropout", "hidden_dim", "learning_rate", "temperature", "momentum"
                ])
            writer.writerow([
                seed, epoch_acc_cis_test, epoch_acc_trans_test, weight_clip,
                num_epochs, batch_size, num_layers, dropout, hidden_dim, learning_rate, temperature, momentum
            ])

    avg_acc_cis_test = np.mean(results_cis_test_seeds) if results_cis_test_seeds else 0
    std_acc_cis_test = np.std(results_cis_test_seeds) if results_cis_test_seeds else 0
    avg_acc_trans_test = np.mean(results_trans_test_seeds) if results_trans_test_seeds else 0
    std_acc_trans_test = np.std(results_trans_test_seeds) if results_trans_test_seeds else 0

    results_file = "results_test_random_search.csv"
    results_exist = os.path.isfile(results_file)

    with open(results_file, mode='a', newline='') as file:

        writer = csv.writer(file)

        # Write header only if the file is empty or newly created

        if not results_exist or os.stat(results_file).st_size == 0:
            writer.writerow([

                "Experiment", "avg_acc_cis_val", "std_acc_cis_val", "avg_acc_trans_val", "std_acc_trans_val" ,"weight_clip", "num_epochs", "batch_size",
                "num_layers", "dropout", "hidden_dim", "learning_rate", "temperature", "momentum"
            ])

        # Append new experiment results with correct number of fields

        writer.writerow([ name_exp, avg_acc_cis_test, std_acc_cis_test,avg_acc_trans_test, std_acc_trans_test, weight_clip, num_epochs, batch_size,num_layers, dropout, hidden_dim, learning_rate, temperature, momentum])

