import os
import csv
import wandb
from feature_extraction.Monte_carlo_partition import monte_carlo_partition
import numpy as np


def combination(n_combination):
    parameters=[]
    np.random.seed(23)
    for _ in range(n_combination):
        parameters.append({
            'batch_size': int(np.random.choice(np.array([128, 256]), size=1)[0]),
            'hidden_dim': int(np.random.choice(np.arange(253, 973, 60))),
            'lr': float(np.round(np.random.choice(np.arange(0.01, 0.1, 0.01)), decimals=2)),
            'momentum': float(np.round(np.random.choice(np.arange(0.8, 1, 0.02)), decimals=2)),
            'num_epochs': int(np.random.randint(25, 101, size=1)[0]),
            't': float(np.random.choice(np.array([0.1, 0.01, 0.001]), size=1)[0]),
            'weight_Clip': float(np.random.choice(np.array([0.4, 0.5, 0.6]), size=1)[0])
        }
        )
    return parameters


def random_search_hyperparameters(path_features, train_type, model_version, model, name_exp, seeds,n_combination=30,sup_loss=0):
    token = os.getenv("WandB_TOKE")
    wandb.login(key=token)
    hyperparameters=combination(n_combination)

    for n in range (n_combination):
        batch_size= hyperparameters[n]['batch_size']
        hidden_dim = hyperparameters[n]['hidden_dim']
        lr = hyperparameters[n]['lr']
        momentum = hyperparameters[n]['momentum']
        num_epochs = hyperparameters[n]['num_epochs']
        t =hyperparameters[n]['t']
        weight_Clip = hyperparameters[n]['weight_Clip']

        results_val_seeds = []
        wandb.init(project=f'{name_exp}_{n}',config={"batch_size": batch_size,"hidden_dim":hidden_dim,"lr":lr,"momentum":momentum,"num_epochs":num_epochs,"t":t,"weight_Clip":weight_Clip})
        for seed in seeds:

            if train_type == 'Out_domain':
                    features_D = [monte_carlo_partition(path_features[0][0], seed), path_features[0][1]]
                    features_S = [path_features[1][0], path_features[1][1]]
                    features = [features_D, features_S]


            elif train_type == 'In_domain':
                features = [[monte_carlo_partition( path_features[0][0], seed), path_features[0][1]]]

            if model_version == 'Base' or model_version == 'Base_long':
                if train_type == 'Out_domain':
                    t = 0.1
                    model.set_parameters(weight_Clip=weight_Clip, num_epochs=num_epochs, batch_size=batch_size,
                                         num_layers="", dropout="", hidden_dim=hidden_dim, lr=lr,
                                         t=t,momentum=momentum, patience=5, path_features_D=features[0][0],
                                         path_prompts_D=features[0][1], path_features_S=features[1][0],
                                         path_prompts_S=features[1][1], exp_name=f'{seed}_{model_version}_{train_type}',sup_loss=sup_loss, wnb=1)

            if model_version == 'Fine_tuning':
                if train_type == 'In_domain':
                    lr=1e-3
                    t=0.1
                    model.set_parameters(weight_Clip=weight_Clip, num_epochs=num_epochs, batch_size=batch_size,
                                         num_layers="", dropout="", hidden_dim=hidden_dim, lr=lr,
                                         t=t, momentum=momentum, patience=5, path_features_D=features[0][0],
                                         path_prompts_D=features[0][1], exp_name=f'{seed}_{model_version}_{train_type}',sup_loss=sup_loss, wnb=1)


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
                    "avg_acc_val","std_acc_val", "weight_clip", "num_epochs", "batch_size", "hidden_dim", "learning_rate", "temperature", "momentum"
                ])
            # Escribir la configuración y el resultado
            writer.writerow([avg_acc_val,std_acc_val, weight_Clip, num_epochs, batch_size, hidden_dim, lr, t, momentum])


def test_best_model(path_features,train_type, model_version,model, name_exp,config, seeds,sup_loss=0,dataset='terra'):
    weight_clip = config['weight_Clip']
    dropout = config['dropout']
    num_layers = config['num_layers']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    hidden_dim = config['hidden_dim']
    learning_rate = config['lr']
    temperature = config['t']
    momentum = config['momentum']

    results_cis_test_seeds = []
    results_trans_test_seeds = []
    results_test_seeds = []

    results_temporal = f"results_test_random_search_temporal_{name_exp}.csv"
    results_exist_temp = os.path.isfile(results_temporal)
    existing_seeds = set()

    # Read existing seeds from CSV if the file exists
    with open(results_temporal, mode='w', newline='') as file:
        writer = csv.writer(file)
        if train_type == 'Out_domain' or (train_type == 'In_domain' and dataset=='terra'):
            writer.writerow(["seed", "acc_cis_test", "acc_trans_test"])
        else:
            writer.writerow(["seed", "acc_test"])

    for seed in seeds:
        if seed in existing_seeds:
            print(f"Skipping seed {seed}, already tested.")
            continue

        if train_type == 'Out_domain':
            features_D = [path_features[0][0], path_features[0][1]]
            features_S = [path_features[1][0], path_features[1][1]]
            features = [features_D, features_S]
        elif train_type == 'In_domain':
            features = [[path_features[0][0], path_features[0][1]],["",""]]

        if (model_version == 'Base' or model_version == 'Base_long') and train_type == 'Out_domain':
            model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                 num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                 t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                 path_prompts_D=features[0][1], path_features_S=features[1][0],
                                 path_prompts_S=features[1][1], exp_name=f'{seed}_{name_exp}',sup_loss=sup_loss,wnb=0)

        elif model_version == 'Fine_tuning' and train_type == 'In_domain':
            model.set_parameters(weight_Clip=weight_clip, num_epochs=num_epochs, batch_size=batch_size,
                                 num_layers=num_layers, dropout=dropout, hidden_dim=hidden_dim, lr=learning_rate,
                                 t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                 path_prompts_D=features[0][1], exp_name=f'{seed}_{name_exp}',sup_loss=sup_loss, wnb=0)
        elif 'MLP' in model_version :
                model.set_parameters(num_epochs=num_epochs, batch_size=batch_size,num_layers=num_layers, dropout=dropout,
                                     hidden_dim=hidden_dim, lr=learning_rate,t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                     path_prompts_D=features[0][1],path_features_S=features[1][0],
                                     path_prompts_S=features[1][1], exp_name=f'{seed}_{name_exp}',
                                     wnb=0)
        elif 'Adapter' in model_version:
                model.set_parameters(num_epochs=num_epochs, batch_size=batch_size,num_layers="", dropout="",hidden_dim=hidden_dim, lr=learning_rate,t=temperature, momentum=momentum, patience=5, path_features_D=features[0][0],
                                     path_prompts_D=features[0][1],path_features_S=features[1][0],path_prompts_S=features[1][1], exp_name=f'{seed}_{name_exp}',
                                     wnb=0)

        if train_type == 'Out_domain' or (train_type == 'In_domain' and dataset=='terra'):
            epoch_loss_cis_test, epoch_acc_cis_test, epoch_loss_trans_test, epoch_acc_trans_test = model.train(seed=seed, test=1)
            results_cis_test_seeds.append(epoch_acc_cis_test)
            results_trans_test_seeds.append(epoch_acc_trans_test)
        else:
            epoch_loss_test, epoch_acc_test = model.train_ID(seed=seed, test=1)
            results_test_seeds.append(epoch_acc_test)


        # Append new results to CSV
        with open(results_temporal, mode='a', newline='') as file:
            writer = csv.writer(file)
            if train_type == 'Out_domain' or (train_type == 'In_domain' and dataset=='terra'):
                writer.writerow([seed, epoch_acc_cis_test, epoch_acc_trans_test])
            else:
                writer.writerow([seed, epoch_acc_test])
    if train_type == 'Out_domain' or (train_type == 'In_domain' and dataset=='terra'):
        avg_acc_cis_test = np.mean(results_cis_test_seeds) if results_cis_test_seeds else 0
        std_acc_cis_test = np.std(results_cis_test_seeds) if results_cis_test_seeds else 0
        avg_acc_trans_test = np.mean(results_trans_test_seeds) if results_trans_test_seeds else 0
        std_acc_trans_test = np.std(results_trans_test_seeds) if results_trans_test_seeds else 0
        header=["Experiment", "avg_acc_cis_test", "std_acc_cis_test", "avg_acc_trans_test", "std_acc_trans_test"  ]
        line =[ name_exp, avg_acc_cis_test, std_acc_cis_test,avg_acc_trans_test, std_acc_trans_test]
    else:
        avg_acc_test = np.mean(results_test_seeds) if results_test_seeds else 0
        std_acc_test = np.std(results_test_seeds) if results_test_seeds else 0
        header = ["Experiment", "avg_acc_test", "std_acc_test"]
        line = [ name_exp, avg_acc_test, std_acc_test]


    results_file = f"results_test_random_search_{train_type}.csv"
    results_exist = os.path.isfile(results_file)

    with open(results_file, mode='a', newline='') as file:

        writer = csv.writer(file)
        if not results_exist:
            writer.writerow(header)

        # Append new experiment results with correct number of fields

        writer.writerow(line)
