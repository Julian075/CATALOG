import os.path


from random_search_hyperparameters import test_best_model,random_search_hyperparameters

from models import CATALOG_Base as base
from models import CATALOG_Base_long as base_long
from models import CATALOG_Base_fine_tuning as base_fine_tuning
from models import CLIP_Mlp as CLIP_MLP
from models import BioCLIP_Mlp as BioCLIP_MLP


import argparse
from utils import BaselineDataset,dataloader_baseline,TuningDataset,dataloader_Tuning,build_optimizer,feature_extraction_
from data.seeds import val_seeds, test_seeds, test_seeds_finetuning

from train.Train_CATALOG_Base_out_domain import CATALOG_base
from train.Train_CLIP_MLP import CLIP_MLP_train

from train.Fine_tuning.Train_CATALOG_Base_In_domain import CATALOG_base_In_domain
from train.Fine_tuning.Train_CATALOG_Base_In_domain_Terra import CATALOG_base_In_domain_terra


def mode_model(model,model_params_path,mode):
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.prueba_model(model_params_path=model_params_path)

    elif mode == 'test_top3':
        model.prueba_model_top_3(model_params_path=model_params_path)

model_type = {
            "Base": base,
            "Base_long": base_long,
            "Fine_tuning": base_fine_tuning,
            "CLIP_MLP": CLIP_MLP,
            "Long_CLIP_MLP": CLIP_MLP,
            "BioCLIP_MLP": BioCLIP_MLP,
            "Long_CLIP_Adapter": CLIP_MLP,
            "CLIP_Adapter": CLIP_MLP,
            "BioCLIP_Adapter": BioCLIP_MLP,
        }
type_feat = {
            "Base": 'standard_features',
            "Base_long": 'long_features',
            "Fine_tuning": 'finetuning_features',
            "CLIP_MLP": 'CLIP_MLP',
            "Long_CLIP_MLP": 'CLIP_MLP',
            "BioCLIP_MLP": 'CLIP_MLP',
            "CLIP_Adapter": 'CLIP_MLP',
            "Long_CLIP_Adapter": 'CLIP_MLP',
            "BioCLIP_Adapter": 'CLIP_MLP',
            }
ext_name_feats = {
            "Base": '',
            "Base_long": '',
            "Fine_tuning": '_longclip-B',
            "CLIP_MLP": '_16',
            "Long_CLIP_MLP": '_longclip-B',
            "BioCLIP_MLP": '_BioCLIP',
            "Long_CLIP_Adapter": '_longclip-B',
            "CLIP_Adapter": '_16',
            "BioCLIP_Adapter": '_BioCLIP',
            }
model_params_path = {
            "Base": 'models/CATALOG_BERT.pth',
            "Base_long": '/home/ids/jpabon/projects/CATALOG/Best/4258031807_Base_long_Out_domain/training_2025-02-19_09-47-16/best_model_params__613.pth',#'models/CATALOG_LongCLIP.pth',
            "Fine_tuning": {'serengeti':'models/CATALOG_finetuning_Base_Long_Serengeti.pth','terra':'models/CATALOG_finetuning_Base_Long_terra.pth'},
            "CLIP_MLP": 'models/CLIP_MLP.pth',
            "Long_CLIP_MLP": 'models/Long_CLIP_MLP.pth',
            "BioCLIP_MLP": 'models/BioCLIP_MLP.pth',
            "CLIP_Adapter": 'models/CLIP_Adapter.pth',
            "Long_CLIP_Adapter": 'models/Long_CLIP_Adapter.pth',
            "BioCLIP_Adapter": 'models/BioCLIP_Adapter.pth',
        }
config = {
            "Base": {"weight_Clip": 0.494, "num_epochs": 107, "batch_size": 128, "num_layers": "", "dropout": "", "hidden_dim": 913,"lr": 0.017475,"t": 0.0983,"momentum": 0.95166},
            "Base_long": {"weight_Clip": 0.5, "num_epochs": 107, "batch_size": 256, "num_layers": "", "dropout":"", "hidden_dim": 613,"lr": 0.09,"t": 0.1,"momentum": 0.98},
            "Fine_tuning": {'serengeti':{"weight_Clip": 0.6, "num_epochs": 1000, "batch_size": 100, "num_layers": "", "dropout": "", "hidden_dim": 913,"lr": 1e-3,"t": 0.1,"momentum": 0.8409},'terra':{"weight_Clip": 0.6, "num_epochs": 57, "batch_size": 256, "num_layers": "", "dropout": "", "hidden_dim": 733,"lr": 1e-3,"t": 0.1,"momentum": 0.82}},
            "CLIP_MLP":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": 1, "dropout": 0.42656, "hidden_dim": 913, "lr": 0.017475, "t": 0.0983,"momentum": 0.95166},          ##0.6,57,256,733,0.001,0.1,0.82
            "Long_CLIP_MLP":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": 1, "dropout": 0.42656, "hidden_dim": 913, "lr": 0.017475, "t": 0.0983,"momentum": 0.95166},
            "BioCLIP_MLP":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": 1, "dropout": 0.42656, "hidden_dim": 913, "lr": 0.017475, "t": 0.0983, "momentum": 0.95166},
            "CLIP_Adapter":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": "", "dropout": "", "hidden_dim": 256, "lr": 0.017475, "t": 0.0983, "momentum": 0.95166},
            "Long_CLIP_Adapter":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": "", "dropout": "", "hidden_dim": 256, "lr": 0.017475, "t": 0.0983,"momentum": 0.95166},
            "BioCLIP_Adapter":{"weight_Clip": "", "num_epochs": 107, "batch_size": 128, "num_layers": "", "dropout": "", "hidden_dim": 256, "lr": 0.017475, "t": 0.0983,"momentum": 0.95166},
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program description')

    parser.add_argument('--model_version', type=str, default="Base_long", help='Model version')
    parser.add_argument('--dataset', type=str, default="serengeti", help='dataset')
    parser.add_argument('--dataset2', type=str, default="terra", help='dataset')
    parser.add_argument('--mode', type=str, default="train", help='define if you want train or test or feature_extraction')
    parser.add_argument('--train_type', type=str, default="Out_domain", help='Type of training')
    parser.add_argument('--hyperparameterTuning_mode', type=int, default=0, help='Type of training')
    parser.add_argument('--feature_extraction', type=int, default=0, help='Type of training')
    parser.add_argument('--sup_loss', type=int, default=0, help='Enable the Attention layer')

    parser.add_argument('--LLM', type=str, default="ChatGPT_0.5", help='define LLM')
    args = parser.parse_args()

    model_version = args.model_version
    train_type = args.train_type
    dataset=args.dataset
    dataset2 = args.dataset2
    mode = args.mode
    hyperparameterTuning_mode=args.hyperparameterTuning_mode
    feature_extraction=args.feature_extraction
    LLM=args.LLM
    sup_loss=args.sup_loss

    if feature_extraction :
        feature_extraction_(model_version,dataset,LLM)
        
        if train_type=='Out_domain':
            feature_extraction_(model_version,dataset2,LLM)

    path_features_D = f"features/Features_{dataset}/{type_feat[model_version]}/Features{ext_name_feats[model_version]}_{dataset}.pt"
    path_prompts_D = f"features/Features_{dataset}/{type_feat[model_version]}/Prompts{ext_name_feats[model_version]}_{dataset}_{LLM}.pt"
    if train_type == "Out_domain":
        path_features_S = f"features/Features_{dataset2}/{type_feat[model_version]}/Features{ext_name_feats[model_version]}_{dataset2}.pt"
        path_prompts_S = f"features/Features_{dataset2}/{type_feat[model_version]}/Prompts{ext_name_feats[model_version]}_{dataset2}_{LLM}.pt"


    model_params_path=model_params_path[model_version]

    if model_version=="Base" or model_version=="Base_long":
        if train_type=="Out_domain":
            if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                seeds=val_seeds
                features_D=[path_features_D,path_prompts_D]
                features_S = [path_features_S, path_prompts_S]

                model = CATALOG_base( model=model_type[model_version], Dataset=BaselineDataset,Dataloader=dataloader_baseline, version='base',build_optimizer=build_optimizer)

                if hyperparameterTuning_mode == 1:
                    seeds = val_seeds
                    random_search_hyperparameters([features_D, features_S], train_type, model_version, model, f'{model_version}_{train_type}_{LLM}', seeds, n_combination=30, sup_loss=sup_loss)
                else:
                    seeds = test_seeds
                    test_best_model([features_D, features_S],train_type, model_version,model, f'{model_version}_{train_type}_{LLM}',config[model_version], seeds,sup_loss=sup_loss)

            else:
                model = CATALOG_base(model=model_type[model_version], Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='base',build_optimizer=build_optimizer)
                model.set_parameters(weight_Clip=config[model_version]['weight_Clip'], num_epochs=config[model_version]['num_epochs'], batch_size=config[model_version]['batch_size'],num_layers=config[model_version]['num_layers'], dropout=config[model_version]['dropout'], hidden_dim=config[model_version]['hidden_dim'], lr= config[model_version]['lr'],
                                     t=config[model_version]['t'],momentum=config[model_version]['momentum'], patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S=path_features_S,
                                     path_prompts_S=path_prompts_S, exp_name=f'{model_version}_{train_type}', sup_loss=sup_loss,wnb=0)

                mode_model(model, model_params_path, mode)
        elif train_type=='In_domain':
                model = CATALOG_base(model=model_type[model_version], Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='base',build_optimizer=build_optimizer)
                model.set_parameters(weight_Clip=config[model_version]['weight_Clip'], num_epochs=config[model_version]['num_epochs'], batch_size=config[model_version]['batch_size'],num_layers=config[model_version]['num_layers'], dropout=config[model_version]['dropout'], hidden_dim=config[model_version]['hidden_dim'], lr= config[model_version]['lr'],
                                     t=config[model_version]['t'],momentum=config[model_version]['momentum'], patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S="",
                                     path_prompts_S="", exp_name=f'{model_version}_{train_type}', sup_loss=sup_loss,wnb=0)

                if dataset != 'terra':
                    model.train_ID()
                else:
                    model.train_ID_terra()


    elif 'Fine_tuning' in model_version:
        model_params_path = model_params_path[dataset]

        if train_type=="In_domain":
            if dataset=="serengeti":
                    model = CATALOG_base_In_domain( model=model_type[model_version], Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)
                    model.set_parameters(weight_Clip=config[model_version][dataset]['weight_Clip'], num_epochs=config[model_version][dataset]['num_epochs'], batch_size=config[model_version][dataset]['batch_size'],num_layers=config[model_version][dataset]['num_layers'], dropout=config[model_version][dataset]['dropout'], hidden_dim=config[model_version][dataset]['hidden_dim'], lr= config[model_version][dataset]['lr'],
                                     t=config[model_version][dataset]['t'],momentum=config[model_version][dataset]['momentum'], patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D,exp_name=f'exp_{model_version}_{train_type}_{dataset}',sup_loss=sup_loss)

            elif dataset=="terra":
                    model = CATALOG_base_In_domain_terra( model=model_type[model_version], Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)

                    model.set_parameters(weight_Clip=config[model_version][dataset]['weight_Clip'], num_epochs=config[model_version][dataset]['num_epochs'], batch_size=config[model_version][dataset]['batch_size'],num_layers=config[model_version][dataset]['num_layers'], dropout=config[model_version][dataset]['dropout'], hidden_dim=config[model_version][dataset]['hidden_dim'], lr= config[model_version][dataset]['lr'],
                                     t=config[model_version][dataset]['t'],momentum=config[model_version][dataset]['momentum'], patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D,exp_name=f'exp_{model_version}_{train_type}_{dataset}',sup_loss=sup_loss)

            if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                seeds=val_seeds
                features_D=[path_features_D,path_prompts_D]
                
                if hyperparameterTuning_mode == 1:
                    seeds = val_seeds
                    random_search_hyperparameters([features_D], train_type, model_version, model, f'{model_version}_{train_type}_{LLM}_{dataset}', seeds, n_combination=30, sup_loss=sup_loss)
                else:
                    seeds = test_seeds_finetuning
                    test_best_model([features_D],train_type, model_version,model, f'{model_version}_{train_type}_{LLM}_{dataset}',config[model_version], seeds,sup_loss=sup_loss)

            else:
                mode_model(model, model_params_path, mode)
    elif "MLP" in model_version  or "Adapter" in model_version:
        if train_type=="Out_domain":
                model = CLIP_MLP_train(model=model_type[model_version], Dataset=BaselineDataset,Dataloader=dataloader_baseline,version=model_version,build_optimizer=build_optimizer)
                model.set_parameters(num_epochs=config[model_version]['num_epochs'], batch_size=config[model_version]['batch_size'],num_layers=config[model_version]['num_layers'], dropout=config[model_version]['dropout'], hidden_dim=config[model_version]['hidden_dim'], lr= config[model_version]['lr'],
                                     t=config[model_version]['t'],momentum=config[model_version]['momentum'],patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S=path_features_S,
                                     path_prompts_S=path_prompts_S, exp_name=f'{model_version}_{train_type}',wnb=0)
                mode_model(model, model_params_path, mode)

        elif train_type == "In_domain":
           model = CLIP_MLP_train(model=model_type[model_version], Dataset=BaselineDataset,Dataloader=dataloader_baseline,version=model_version,build_optimizer=build_optimizer)
           model.set_parameters(num_epochs=config[model_version]['num_epochs'], batch_size=config[model_version]['batch_size'],num_layers=config[model_version]['num_layers'], dropout=config[model_version]['dropout'], hidden_dim=config[model_version]['hidden_dim'], lr= config[model_version]['lr'],
                                 t=config[model_version]['t'],momentum=config[model_version]['momentum'],patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S="",path_prompts_S="", exp_name=f'{model_version}_{train_type}',wnb=0)

           if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
               seeds = val_seeds
               features_D = [path_features_D, path_prompts_D]

               if hyperparameterTuning_mode == 1:
                   seeds = val_seeds
                   random_search_hyperparameters([features_D], train_type, model_version, model,
                                                 f'{model_version}_{train_type}_{LLM}_{dataset}', seeds,
                                                 n_combination=30, sup_loss=sup_loss)
               else:
                   seeds = test_seeds_finetuning
                   test_best_model([features_D], train_type, model_version, model,
                                   f'{model_version}_{train_type}_{LLM}_{dataset}', config[model_version], seeds,
                                   sup_loss=sup_loss,dataset=dataset)

           elif dataset!='terra':
               model.train_ID()
           else:
               model.train_ID_terra()