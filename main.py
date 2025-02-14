import os.path

from feature_extraction.CATALOG_feature_extraction import extract_features
from random_search_hyperparameters import random_search,test_best_model,random_search2,random_search_MLP,random_search_Adapter

from models import CATALOG_Base as base
from models import CATALOG_Base_long as base_long
from models import CATALOG_Base_fine_tuning as base_fine_tuning
from models import CLIP_Mlp as CLIP_MLP


import argparse
from utils import BaselineDataset,dataloader_baseline,TuningDataset,dataloader_Tuning,build_optimizer
from data.seeds import val_seeds, test_seeds

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program description')

    parser.add_argument('--model_version', type=str, default="Base_long", help='Model version')
    parser.add_argument('--dataset', type=str, default="terra", help='dataset')
    parser.add_argument('--dataset2', type=str, default="terra", help='dataset')
    parser.add_argument('--mode', type=str, default="train", help='define if you want train or test or feature_extraction')
    parser.add_argument('--train_type', type=str, default="Out_domain", help='Type of training')
    parser.add_argument('--hyperparameterTuning_mode', type=int, default=0, help='Type of training')
    parser.add_argument('--feature_extraction', type=int, default=0, help='Type of training')#en_att
    parser.add_argument('--en_att', type=int, default=0, help='Enable the Attention layer')

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
    en_att=args.en_att

    if feature_extraction :
        if model_version=='Base_long':
            extract_features(model_version=model_version, dataset=dataset, type_clip='longclip-B', LLM=LLM, only_text=0)
        elif model_version == "BioCLIP_MLP":
            extract_features(model_version=model_version, dataset=dataset, type_clip='BioCLIP', LLM=LLM, only_text=0)
        else:
            extract_features(model_version=model_version, dataset=dataset, type_clip='16', LLM=LLM, only_text=0)
    else:

            if model_version=="Base" or model_version=="Base_long":
                if model_version == "Base":
                    model_type=base
                    type_feat="standard_features"
                    model_params_path = 'models/CATALOG_BERT.pth'
                    #model_params_path = 'models/CATALOG_LongCLIP_BERT.pth'
                else:
                    model_type=base_long
                    type_feat = "long_features"
                    model_params_path = 'models/CATALOG_LongCLIP_BERT.pth'

                path_features_D = f"features/Features_{dataset}/{type_feat}/Features_{dataset}.pt"
                path_prompts_D = f"features/Features_{dataset}/{type_feat}/Prompts_{dataset}_{LLM}.pt"


                path_features_S = f"features/Features_{dataset2}/{type_feat}/Features_{dataset2}.pt"
                path_prompts_S = f"features/Features_{dataset2}/{type_feat}/Prompts_{dataset2}_{LLM}.pt"


                if train_type=="Out_domain":
                    if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                        seeds=val_seeds
                        features_D=[path_features_D,path_prompts_D]
                        features_S = [path_features_S, path_prompts_S]

                        model = CATALOG_base( model=model_type, Dataset=BaselineDataset,Dataloader=dataloader_baseline, version='base',build_optimizer=build_optimizer)

                        if hyperparameterTuning_mode == 1:
                            seeds = val_seeds
                            random_search2([features_D, features_S], train_type, model_version,model, f'{model_version}2_{train_type}_{LLM}_ATT_{en_att}',f'Hp_{model_version}2_{LLM}_ATT_{en_att}',seeds,en_att=en_att)
                        else:
                            config = {"weight_Clip": 0.494, "num_epochs": 107, "batch_size": 128, "num_layers": 1, "dropout": 0.42656, "hidden_dim": 913,"lr": 0.017475,"t": 0.0983,"momentum": 0.95166}
                            seeds = test_seeds
                            test_best_model([features_D, features_S],train_type, model_version,model, f'{model_version}2_{train_type}_{LLM}_ATT_{en_att}',config, seeds,en_att=en_att)

                    else:
                        model = CATALOG_base(model=model_type, Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='base',build_optimizer=build_optimizer)
                        model.set_parameters(weight_Clip=0.494, num_epochs=107, batch_size=128,num_layers=1, dropout=0.42656, hidden_dim=913, lr=0.017475,
                                             t=0.0983,momentum=0.95166, patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S=path_features_S,
                                             path_prompts_S=path_prompts_S, exp_name=f'{model_version}_{train_type}', en_att=en_att,wnb=0)

                        mode_model(model, model_params_path, mode)


            elif model_version == "Fine_tuning":

                path_features_D = f"features/Features_{dataset}/finetuning_features/Features_{dataset}.pt"
                path_prompts_D = f"features/Features_{dataset}/finetuning_features/Prompts_{dataset}_{LLM}.pt"
                if not os.path.isfile(path_prompts_D):
                    path_prompts_D = f"features/Features_{dataset}/finetuning_features/Prompts_{dataset}.pt"

                if train_type=="In_domain":
                    if dataset=="serengeti":

                        if hyperparameterTuning_mode == 1:
                            seeds=val_seeds
                            features_D=[path_features_D,path_prompts_D]


                            model = CATALOG_base_In_domain( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)

                            random_search([features_D], train_type, model_version,model, f'{train_type}_{dataset}',f'Hp_{model_version}_{train_type}_{dataset}_{LLM}',seeds)
                        else:
                            model = CATALOG_base_In_domain( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)
                            model.set_parameters(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=4, dropout=0.4,hidden_dim=1743,lr=1e-3,t=0.1,momentum=0.8409
                                                                , patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D,exp_name=f'exp_{model_version}_{train_type}_{dataset}')


                            model_params_path = 'models/CATALOG_finetuning_Base_Serengeti.pth'
                            mode_model(model, model_params_path, mode)

                    elif dataset=="terra":

                        if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                            seeds = val_seeds
                            features_D = [path_features_D, path_prompts_D]

                            model = CATALOG_base_In_domain_terra( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)


                            if hyperparameterTuning_mode == 1:
                                seeds = val_seeds
                                random_search([features_D], train_type, model_version, model,f'{train_type}_{dataset}', f'Hp_{model_version}_{train_type}_{dataset}_{LLM}', seeds)
                            else:
                                config = {"weight_Clip": 0.391885, "num_epochs": 185, "batch_size": 32, "num_layers": 7,
                                          "dropout": 0.391855, "hidden_dim": 1024,
                                          "lr": 0.00034235, "t": 0.157429, "momentum": 0.8851568}
                                seeds = test_seeds
                                test_best_model([features_D], train_type, model_version, model,
                                                f'{train_type}_{dataset}_{LLM}', config, seeds)

                        else:
                            model = CATALOG_base_In_domain_terra( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)

                            model.set_parameters(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=1,
                                                dropout=0.5,hidden_dim=1045,lr=1e-4,t=0.1,momentum=0.8409, patience=5,
                                                 path_features_D=path_features_D, path_prompts_D=path_prompts_D,
                                                 exp_name=f'{model_version}_{train_type}', wnb=0)

                            model_params_path = 'models/CATALOG_finetuning_Base_Terra.pth'
                            mode_model(model, model_params_path, mode)

            elif model_version == "CLIP_MLP" or  model_version == "CLIP_Adapter"  or model_version == "BioCLIP_MLP":
                if model_version== "CLIP_MLP" or  model_version == "CLIP_Adapter":
                    path_features_D = f"features/Features_{dataset}/CLIP_MLP/Features_16_{dataset}.pt"
                    path_prompts_D = f"features/Features_{dataset}/CLIP_MLP/Prompts_16_{dataset}_{LLM}.pt"

                elif model_version== "BioCLIP_MLP":
                    path_features_D = f"features/Features_{dataset}/CLIP_MLP/Features_BioCLIP_{dataset}.pt"
                    path_prompts_D = f"features/Features_{dataset}/CLIP_MLP/Prompts_BioCLIP_{dataset}_{LLM}.pt"


                if train_type=="Out_domain":
                    model_params_path = 'models/CLIP_MLP_Out_Domain.pth'
                    if model_version == "CLIP_MLP" or  model_version == "CLIP_Adapter":
                        path_features_S = f"features/Features_{dataset2}/CLIP_MLP/Features_16_{dataset2}.pt"
                        path_prompts_S = f"features/Features_{dataset2}/CLIP_MLP/Prompts_16_{dataset2}_{LLM}.pt"

                    elif model_version == "BioCLIP_MLP":
                        path_features_S = f"features/Features_{dataset2}/CLIP_MLP/Features_BioCLIP_{dataset2}.pt"
                        path_prompts_S = f"features/Features_{dataset2}/CLIP_MLP/Prompts_BioCLIP_{dataset2}_{LLM}.pt"

                    if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                        seeds=val_seeds
                        features_D=[path_features_D,path_prompts_D]
                        features_S = [path_features_S, path_prompts_S]

                        model = CLIP_MLP_train( model=CLIP_MLP, Dataset=BaselineDataset,Dataloader=dataloader_baseline, version=model_version,build_optimizer=build_optimizer)

                        if hyperparameterTuning_mode == 1:
                            seeds = val_seeds
                            if model_version == 'CLIP_Apater':
                                random_search_Adapter([features_D, features_S], train_type, model_version,model, f'{model_version}_{train_type}_{LLM}',f'Hp_{model_version}_{LLM}',seeds)
                            elif model_version == 'CLIP_MLP':
                                random_search_MLP([features_D, features_S], train_type, model_version,model, f'{model_version}_{train_type}_{LLM}',f'Hp_{model_version}_{LLM}',seeds)
                        else:
                            config = {"weight_Clip": 0.494, "num_epochs": 107, "batch_size": 128, "num_layers": 1,"dropout": 0.42656, "hidden_dim": 913, "lr": 0.017475, "t": 0.0983,
                                      "momentum": 0.95166}

                            seeds = test_seeds
                            test_best_model([features_D, features_S],train_type, model_version,model, f'{model_version}_{train_type}_{LLM}',config, seeds)

                    else:
                        model = CLIP_MLP_train(model=CLIP_MLP, Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='base',build_optimizer=build_optimizer)
                        model.set_parameters(num_epochs=107, batch_size=128,num_layers=1, dropout=0.42656, hidden_dim=913, lr=0.017475,
                                             t=0.0983,momentum=0.95166, patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S=path_features_S,
                                             path_prompts_S=path_prompts_S, exp_name=f'{model_version}_{train_type}',wnb=0)

                        mode_model(model, model_params_path, mode)

                elif train_type == "In_domain":
                   print('Loading..')





