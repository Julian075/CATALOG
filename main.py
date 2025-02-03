import os.path

import numpy as np

from feature_extraction.Base.CATALOG_feature_extraction import extract_features
from random_search_hyperparameters import random_search,test_best_model,random_search2

from models import CATALOG_Base as base
from models import CATALOG_Base_fine_tuning as base_fine_tuning
from models import CATALOG_Base_fine_tuning_last_layer as base_fine_tuning_layer
from models import CATALOG_Base_long as base_long

from models import CATALOG_Projections as projections
from models import CATALOG_Projections_fine_tuning as projections_fine_tuning
from models import CATALOG_Projections_fine_tuning_last_layer as projections_fine_tuning_layer


import argparse
from utils import BaselineDataset,dataloader_baseline,TuningDataset,dataloader_Tuning,build_optimizer
from data.seeds import val_seeds, test_seeds


from train.Base.Train_CATALOG_Projections_Serengeti import CATALOG_projections_serengeti
from train.Base.Train_CATALOG_Projections_Terra import CATALOG_projections_terra
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base

from train.Fine_tuning.Train_CATALOG_Base_In_domain_Serengeti import CATALOG_base_In_domain_serengeti
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

    parser.add_argument('--model_version', type=str, default="Base", help='Model version')
    parser.add_argument('--dataset', type=str, default="serengeti", help='dataset')
    parser.add_argument('--mode', type=str, default="train", help='define if you want train or test or feature_extraction')
    parser.add_argument('--train_type', type=str, default="Out_domain", help='Type of training')
    parser.add_argument('--hyperparameterTuning_mode', type=int, default=0, help='Type of training')
    parser.add_argument('--feature_extraction', type=int, default=0, help='Type of training')

    parser.add_argument('--LLM', type=str, default="ChatGPT", help='define LLM')
    args = parser.parse_args()

    model_version = args.model_version
    train_type = args.train_type
    dataset=args.dataset
    mode = args.mode
    hyperparameterTuning_mode=args.hyperparameterTuning_mode
    feature_extraction=args.feature_extraction
    LLM=args.LLM

    if feature_extraction :
        extract_features(model_version=model_version,dataset=dataset,mode_clip='16',LLM=LLM,only_text=1,AB_omg=1)
    else:

            if model_version=="Base":
                path_features_D = f"features/Features_{dataset}/standard_features/Features_{dataset}.pt"
                path_prompts_D = f"features/Features_{dataset}/standard_features/Prompts_{dataset}_{LLM}.pt"
                if not os.path.isfile(path_prompts_D):
                    path_prompts_D = f"features/Features_{dataset}/standard_features/Prompts_{dataset}.pt"

                path_features_S = "features/Features_terra/standard_features/Features_terra.pt"
                path_prompts_S = f"features/Features_terra/standard_features/Prompts_terra_{LLM}.pt"
                if not os.path.isfile(path_prompts_S):
                    path_prompts_S = f"features/Features_terra/standard_features/Prompts_terra.pt"
                if train_type=="In_domain":
                    if dataset!="terra":

                        model=CATALOG_projections_serengeti(weight_Clip=0.60855,num_epochs=86,batch_size=26, num_layers=4,
                                                        dropout=0.381,hidden_dim=1743,lr=0.0956,t=0.1,momentum=0.8162
                                                        ,patience=20,model=projections,Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='projection',
                                                        ruta_features_train=ruta_features_train,ruta_features_val=ruta_features_val,ruta_features_test=ruta_features_test,
                                                            path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')

                        model_params_path = 'models/CATALOG_Projections_Serengeti.pth'
                        mode_model(model, model_params_path, mode)


                    elif dataset=="terra" :

                        model = CATALOG_projections_terra(weight_Clip=0.4848, num_epochs=74, batch_size=65, num_layers=4,
                                                          dropout=0.2658, hidden_dim=1863, lr=0.032, t=1, momentum=0.909
                                                          , patience=20, model=projections, Dataset=BaselineDataset,
                                                          Dataloader=dataloader_baseline, version='projection',
                                                          ruta_features_train=ruta_features_train,
                                                          ruta_features_val1=ruta_features_cis_val,
                                                          ruta_features_val2=ruta_features_trans_val,
                                                          ruta_features_test1=ruta_features_cis_test,
                                                          ruta_features_test2=ruta_features_trans_test,
                                                          path_text_feat=path_text_feat, build_optimizer=build_optimizer,
                                                          exp_name=f'exp_{model_version}_{train_type}_{dataset}')

                        model_params_path = 'models/CATALOG_Projections_Terra.pth'
                        mode_model(model, model_params_path, mode)


                if train_type=="Out_domain":
                    if hyperparameterTuning_mode == 1 or hyperparameterTuning_mode == 2:
                        seeds=val_seeds
                        features_D=[path_features_D,path_prompts_D]
                        features_S = [path_features_S, path_prompts_S]

                        model = CATALOG_base( model=base, Dataset=BaselineDataset,Dataloader=dataloader_baseline, version='base',build_optimizer=build_optimizer)

                        if hyperparameterTuning_mode == 1:
                            seeds = val_seeds
                            random_search2([features_D, features_S], train_type, model_version,model, f'{train_type}_{LLM}',f'Hp_{model_version}_{LLM}',seeds)
                        else:
                            config = {"weight_Clip": 0.494, "num_epochs": 107, "batch_size": 128, "num_layers": 1, "dropout": 0.42656, "hidden_dim": 913,"lr": 0.017475,"t": 0.0983,"momentum": 0.95166}
                            #if LLM=='ChatGPT':
                            #    config = {"weight_Clip": 0.494,"num_epochs": 107,"batch_size": 128,"num_layers": 1,"dropout": 0.42656, "hidden_dim": 913,
                            #            "lr": 0.017475,"t": 0.0983,"momentum": 0.95166}
                            #elif LLM=='ChatGPT_0.0':
                            #    config = {"weight_Clip": 0.5428,"num_epochs": 107,"batch_size": 128,"num_layers": 1,"dropout": 0.42656, "hidden_dim": 913,
                            #            "lr": 0.017475,"t": 0.0983,"momentum": 0.95166}
                            seeds = test_seeds
                            test_best_model([features_D, features_S],train_type, model_version,model, f'{train_type}_{LLM}',config, seeds)

                    else:
                        model = CATALOG_base(model=base, Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='base',build_optimizer=build_optimizer)
                        model.set_parameters(weight_Clip=0.494, num_epochs=107, batch_size=128,num_layers=1, dropout=0.42656, hidden_dim=913, lr=0.017475,
                                             t=0.0983,momentum=0.95166, patience=5, path_features_D= path_features_D, path_prompts_D=path_prompts_D, path_features_S=path_features_S,
                                             path_prompts_S=path_prompts_S, exp_name=f'{model_version}_{train_type}', wnb=0)

                        model_params_path =f'models/CATALOG_Base.pth'
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


                            model = CATALOG_base_In_domain_serengeti( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)

                            random_search([features_D], train_type, model_version,model, f'{train_type}_{dataset}',f'Hp_{model_version}_{train_type}_{dataset}_{LLM}',seeds)
                        else:
                            model = CATALOG_base_In_domain_serengeti( model=base_fine_tuning, Dataset=TuningDataset,Dataloader=dataloader_Tuning, version='fine_tuning',build_optimizer=build_optimizer)
                            model.set_parameters(weight_Clip=0.5026669097446104,num_epochs=47,batch_size=128, num_layers=4, dropout=0.25553407020377744,hidden_dim=1024,lr=1e-3,t=0.14970543696433683,momentum=0.8530664134160747
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

                            model.set_parameters(weight_Clip=0.391885, num_epochs=185, batch_size=32, num_layers=7,
                                                 dropout=0.391855, hidden_dim=1024, lr=0.00034235,
                                                 t=0.157429, momentum=0.8851568, patience=5,
                                                 path_features_D=path_features_D, path_prompts_D=path_prompts_D,
                                                 exp_name=f'{model_version}_{train_type}', wnb=0)

                            model_params_path = 'models/CATALOG_finetuning_Base_Terra.pth'
                            mode_model(model, model_params_path, mode)
                elif train_type=="Out_domain":
                    ruta_features_train = "features/Features_serengeti/finetuning_features/Features_CATALOG_train_16.pt"
                    ruta_features_val = "features/Features_serengeti/finetuning_features/Features_CATALOG_val_16.pt"
                    ruta_features_test1 = "features/Features_terra/finetuning_features/Features_CATALOG_cis_test_16.pt"
                    ruta_features_test2 = "features/Features_terra/finetuning_features/Features_CATALOG_trans_test_16.pt"
                    path_text_feat1 = "features/Features_serengeti/finetuning_features/Text_features_16.pt"
                    path_text_feat2 = "features/Features_terra/finetuning_features/Text_features_16.pt"
                    model = CATALOG_base(weight_Clip=0.6, num_epochs=1000, batch_size=100, num_layers=1,
                                         dropout=0.27822, hidden_dim=1045, lr=0.07641, t=0.1, momentum=0.8409
                                         , patience=20, model=base_fine_tuning, Dataset=TuningDataset,
                                         Dataloader=dataloader_Tuning, version='fine_tuning',
                                         ruta_features_train=ruta_features_train,
                                         ruta_features_val=ruta_features_val, ruta_features_test1=ruta_features_test1,
                                         ruta_features_test2=ruta_features_test2, path_text_feat1=path_text_feat1,
                                         path_text_feat2=path_text_feat2, build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}')

                    model_params_path = 'models/CATALOG_finetuning_Base_out_domain.pth'
                    mode_model(model, model_params_path, mode)


            elif model_version == "Fine_tuning_layer":
                if train_type=="In_domain":
                    if dataset=="serengeti":
                        ruta_features_train = "features/Features_serengeti/finetuning_features/Features_CATALOG_train_16.pt"
                        ruta_features_val = "features/Features_serengeti/finetuning_features/Features_CATALOG_val_16.pt"
                        ruta_features_test = "features/Features_serengeti/finetuning_features/Features_CATALOG_test_16.pt"
                        path_text_feat = "features/Features_serengeti/finetuning_features/Text_features_16.pt"
                        model=CATALOG_base_In_domain_serengeti(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=1,
                                                        dropout=0.27822,hidden_dim=1045,lr=1e-4,t=0.1,momentum=0.8409
                                                            , patience=20, model=base_fine_tuning_layer, Dataset=TuningDataset,
                                                            Dataloader=dataloader_Tuning, version='fine_tuning_last_layer',
                                                            ruta_features_train=ruta_features_train,
                                                            ruta_features_val=ruta_features_val,
                                                            ruta_features_test=ruta_features_test,
                                                            path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')

                        model_params_path = 'models/CATALOG_finetuning_layer_Base_Serengeti.pth'
                        mode_model(model, model_params_path, mode)


                    elif dataset=="terra":
                        ruta_features_train = "features/Features_terra/finetuning_features/Features_CATALOG_train_16.pt"
                        ruta_features_cis_val = "features/Features_terra/finetuning_features/Features_CATALOG_cis_val_16.pt"
                        ruta_features_trans_val = "features/Features_terra/finetuning_features/Features_CATALOG_trans_val_16.pt"
                        ruta_features_cis_test = "features/Features_terra/finetuning_features/Features_CATALOG_cis_test_16.pt"
                        ruta_features_trans_test = "features/Features_terra/finetuning_features/Features_CATALOG_trans_test_16.pt"
                        path_text_feat = "features/Features_terra/finetuning_features/Text_features_16.pt"

                        model=CATALOG_base_In_domain_terra(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=1,
                                                        dropout=0.27822,hidden_dim=1045,lr=1e-7,t=0.1,momentum=0.8409
                                                            , patience=20, model=base_fine_tuning_layer, Dataset=TuningDataset,
                                                            Dataloader=dataloader_Tuning, version='fine_tuning_last_layer',
                                                               ruta_features_train=ruta_features_train,
                                                               ruta_features_val1=ruta_features_cis_val,
                                                               ruta_features_val2=ruta_features_trans_val,
                                                               ruta_features_test1=ruta_features_cis_test,
                                                               ruta_features_test2=ruta_features_trans_test,
                                                            path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')

                        model_params_path = 'models/CATALOG_finetuning_layer_Base_Terra.pth'
                        mode_model(model, model_params_path, mode)

                elif train_type=="Out_domain":
                    ruta_features_train = "features/Features_serengeti/finetuning_features/Features_CATALOG_train_16.pt"
                    ruta_features_val = "features/Features_serengeti/finetuning_features/Features_CATALOG_val_16.pt"
                    ruta_features_test1 = "features/Features_terra/finetuning_features/Features_CATALOG_cis_test_16.pt"
                    ruta_features_test2 = "features/Features_terra/finetuning_features/Features_CATALOG_trans_test_16.pt"
                    path_text_feat1 = "features/Features_serengeti/finetuning_features/Text_features_16.pt"
                    path_text_feat2 = "features/Features_terra/finetuning_features/Text_features_16.pt"
                    model = CATALOG_base(weight_Clip=0.6, num_epochs=1000, batch_size=100, num_layers=1,
                                         dropout=0.27822, hidden_dim=1045, lr=1e-7, t=0.1, momentum=0.8409
                                         , patience=20, model=base_fine_tuning_layer, Dataset=TuningDataset,
                                         Dataloader=dataloader_Tuning, version='fine_tuning_last_layer',
                                         ruta_features_train=ruta_features_train,
                                         ruta_features_val=ruta_features_val, ruta_features_test1=ruta_features_test1,
                                         ruta_features_test2=ruta_features_test2, path_text_feat1=path_text_feat1,
                                         path_text_feat2=path_text_feat2, build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}')

                    model_params_path = 'models/CATALOG_finetuning_layer_Base_out_domain.pth'
                    mode_model(model, model_params_path, mode)

            elif model_version=="Base_long":

                if train_type=="Out_domain":
                    ruta_features_train  = "features/Features_serengeti/long_standard_features/Features_CATALOG_train_longclip-B.pt"
                    ruta_features_val    = "features/Features_serengeti/long_standard_features/Features_CATALOG_val_longclip-B.pt"
                    ruta_features_test1  = "features/Features_terra/long_standard_features/Features_CATALOG_cis_test_longclip-B.pt"
                    ruta_features_test2  = "features/Features_terra/long_standard_features/Features_CATALOG_trans_test_longclip-B.pt"
                    path_text_feat1      = "features/Features_serengeti/long_standard_features/Text_features_longclip-B.pt"
                    path_text_feat2      = "features/Features_terra/long_standard_features/Text_features_longclip-B.pt"
                    model = CATALOG_base(weight_Clip=0.6, num_epochs=100, batch_size=48, num_layers=4,
                                                          dropout=0.27822, hidden_dim=1045, lr=0.07641, t=0.1, momentum=0.8409
                                                          , patience=20, model=base_long, Dataset=BaselineDataset,
                                                          Dataloader=dataloader_baseline,version='base',ruta_features_train=ruta_features_train,
                                                          ruta_features_val=ruta_features_val,ruta_features_test1=ruta_features_test1,
                                                          ruta_features_test2=ruta_features_test2,path_text_feat1=path_text_feat1,
                                                          path_text_feat2=path_text_feat2,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}')

                    model_params_path = 'models/CATALOG_Base_long.pth'
                    mode_model(model, model_params_path, mode)


