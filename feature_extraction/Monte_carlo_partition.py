import torch
import random




def monte_carlo_partition(path_features,seed):

    random.seed(seed)
    features=torch.load(path_features)
    if len(features)==3:
        features_dev= {**features['train'], **features['val']}
    else:
        features_dev = {**features['train'], **features['trans_val']}

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


