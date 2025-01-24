import torch
import random




def monte_carlo_partition(model_version,dataset,seed):

    if model_version=='Base':
        type_feats='standard_features'
    elif model_version=='Fine_tuning':
        type_feats='finetuning_features'
    elif model_version=='Base_long':
        type_feats='long_standard_features'
    else:
        type_feats = 'standard_features'

    random.seed(seed)
    features=torch.load(f'features/Features_{dataset}/{type_feats}/Features_{dataset}.pt')
    features_dev= {**features['train'], **features['val']}

    monte_carlo_partitions={}
    keys = list(features_dev.keys())
    random.shuffle(keys)

    for key in keys:
        chosen_partition = random.choices(['train','val'], [0.8,0.2])[0]
        if not(chosen_partition in monte_carlo_partitions.keys()):
            monte_carlo_partitions[chosen_partition]={}
        monte_carlo_partitions[chosen_partition][key]=features_dev[key]

    return monte_carlo_partitions
    #torch.save(monte_carlo_partitions, f'features/Features_{dataset}/{type_feats}/monte_carlo/Features_{dataset}.pt')


