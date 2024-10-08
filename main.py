from models import CATALOG_Base as base
from models import CATALOG_Base_fine_tuning as base_fine_tuning
from models import CATALOG_Base_fine_tuning_last_layer as base_fine_tuning_layer
from models import CATALOG_Base_long as base_long

from models import CATALOG_Projections as projections
from models import CATALOG_Projections_fine_tuning as projections_fine_tuning
from models import CATALOG_Projections_fine_tuning_last_layer as projections_fine_tuning_layer

import argparse
from utils import BaselineDataset,dataloader_baseline,TuningDataset,dataloader_Tuning,build_optimizer


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

    parser.add_argument('--model_version', type=str, default="Fine_tuning", help='Model version')
    parser.add_argument('--train_type', type=str, default="In_domain", help='Type of training')
    parser.add_argument('--dataset', type=str, default="serengeti", help='dataset')
    parser.add_argument('--mode', type=str, default="train", help='define if you want train or test')
    args = parser.parse_args()

    model_version = args.model_version
    train_type = args.train_type
    dataset=args.dataset
    mode = args.mode



    if model_version=="Base":
        if train_type=="In_domain":
            if dataset=="serengeti":
                ruta_features_train = "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
                ruta_features_val = "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
                ruta_features_test = "features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt"
                path_text_feat = "features/Features_serengeti/standard_features/Text_features_16.pt"
                model=CATALOG_projections_serengeti(weight_Clip=0.60855,num_epochs=86,batch_size=26, num_layers=4,
                                                dropout=0.381,hidden_dim=1743,lr=0.0956,t=0.1,momentum=0.8162
                                                ,patience=20,model=projections,Dataset=BaselineDataset,Dataloader=dataloader_baseline,version='projection',
                                                ruta_features_train=ruta_features_train,ruta_features_val=ruta_features_val,ruta_features_test=ruta_features_test,
                                                    path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')

                model_params_path = 'models/CATALOG_Projections_Serengeti.pth'
                mode_model(model, model_params_path, mode)


            elif dataset=="terra" :
                ruta_features_train = "features/Features_terra/standard_features/Features_CATALOG_train_16.pt"
                ruta_features_cis_val = "features/Features_terra/standard_features/Features_CATALOG_cis_val_16.pt"
                ruta_features_trans_val = "features/Features_terra/standard_features/Features_CATALOG_trans_val_16.pt"
                ruta_features_cis_test = "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
                ruta_features_trans_test = "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
                path_text_feat = "features/Features_terra/standard_features/Text_features_16.pt"

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
            ruta_features_train  = "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt"
            ruta_features_val    = "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt"
            ruta_features_test1  = "features/Features_terra/standard_features/Features_CATALOG_cis_test_16.pt"
            ruta_features_test2  = "features/Features_terra/standard_features/Features_CATALOG_trans_test_16.pt"
            path_text_feat1      = "features/Features_serengeti/standard_features/Text_features_16.pt"
            path_text_feat2      = "features/Features_terra/standard_features/Text_features_16.pt"
            model = CATALOG_base(weight_Clip=0.6, num_epochs=8, batch_size=48, num_layers=1,
                                                  dropout=0.27822, hidden_dim=1045, lr=0.07641, t=0.1, momentum=0.8409
                                                  , patience=5, model=base, Dataset=BaselineDataset,
                                                  Dataloader=dataloader_baseline,version='base',ruta_features_train=ruta_features_train,
                                                  ruta_features_val=ruta_features_val,ruta_features_test1=ruta_features_test1,
                                                  ruta_features_test2=ruta_features_test2,path_text_feat1=path_text_feat1,
                                                  path_text_feat2=path_text_feat2,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}')

            model_params_path = 'models/CATALOG_Base.pth'
            mode_model(model, model_params_path, mode)

    elif model_version == "Fine_tuning":
        if train_type=="In_domain":
            if dataset=="serengeti":
                ruta_features_train = "features/Features_serengeti/finetuning_features/Features_CATALOG_train_16.pt"
                ruta_features_val = "features/Features_serengeti/finetuning_features/Features_CATALOG_val_16.pt"
                ruta_features_test = "features/Features_serengeti/finetuning_features/Features_CATALOG_test_16.pt"
                path_text_feat = "features/Features_serengeti/finetuning_features/Text_features_16.pt"

                model=CATALOG_base_In_domain_serengeti(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=4,
                                                dropout=0.4,hidden_dim=1743,lr=1e-3,t=0.1,momentum=0.8409
                                                    , patience=20, model=base_fine_tuning, Dataset=TuningDataset,
                                                    Dataloader=dataloader_Tuning, version='fine_tuning',
                                                    ruta_features_train=ruta_features_train,
                                                    ruta_features_val=ruta_features_val,
                                                    ruta_features_test=ruta_features_test,
                                                    path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')



                model_params_path = 'models/CATALOG_finetuning_Base_Serengeti.pth'
                mode_model(model, model_params_path, mode)

            elif dataset=="terra":
                ruta_features_train = "features/Features_terra/finetuning_features/Features_CATALOG_train_16.pt"
                ruta_features_cis_val = "features/Features_terra/finetuning_features/Features_CATALOG_cis_val_16.pt"
                ruta_features_trans_val = "features/Features_terra/finetuning_features/Features_CATALOG_trans_val_16.pt"
                ruta_features_cis_test = "features/Features_terra/finetuning_features/Features_CATALOG_cis_test_16.pt"
                ruta_features_trans_test = "features/Features_terra/finetuning_features/Features_CATALOG_trans_test_16.pt"
                path_text_feat = "features/Features_terra/finetuning_features/Text_features_16.pt"
                #dropout=0.27822
                model=CATALOG_base_In_domain_terra(weight_Clip=0.6,num_epochs=1000,batch_size=100, num_layers=1,
                                                dropout=0.5,hidden_dim=1045,lr=1e-4,t=0.1,momentum=0.8409
                                                    , patience=20, model=base_fine_tuning, Dataset=TuningDataset,
                                                    Dataloader=dataloader_Tuning, version='fine_tuning',
                                                       ruta_features_train=ruta_features_train,
                                                       ruta_features_val1=ruta_features_cis_val,
                                                       ruta_features_val2=ruta_features_trans_val,
                                                       ruta_features_test1=ruta_features_cis_test,
                                                       ruta_features_test2=ruta_features_trans_test,
                                                    path_text_feat=path_text_feat,build_optimizer=build_optimizer,exp_name=f'exp_{model_version}_{train_type}_{dataset}')


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



