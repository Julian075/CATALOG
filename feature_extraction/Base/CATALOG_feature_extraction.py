import os
import torch
from transformers import BertModel, BertTokenizer
import clip
import json
from PIL import Image


camera_trap_templates1 = [
        'a photo captured by a camera trap of a {}.',
        'a camera trap image showing multiple {}.',
        'a camera trap image of a {} in low light conditions.',
        'a camera trap image with low resolution showing the {}.',
        'a camera trap photo of the {} captured in poor conditions.',
        'a cropped camera trap image of the {}.',
        'a camera trap image of the {} captured in challenging conditions.',
        'a camera trap image featuring a bright view of the {}.',
        'a camera trap image of the {} captured in clean conditions.',
        'a camera trap image of the {} captured in dirty conditions.',
        'a camera trap image with low light conditions featuring the {}.',
        'a camera trap image of the {} showing cool conditions.',
        'a black and white camera trap image of the {}.',
        'a pixelated camera trap image of the {}.',
        'a camera trap image with bright conditions showing the {}.',
        'a cropped camera trap image of a {}.',
        'a blurry camera trap image of the {}.',
        'a camera trap image of the {}.',
        'a well-captured camera trap image of the {}.',
        'a camera trap image of a single {}.',
        'a camera trap image of a {}.',
        'a low resolution camera trap image of a {}.',
        'a camera trap image of a large {}.',
        'a rendition of a {} captured by a camera trap.',
        'a camera trap image of a nice {}.',
        'a camera trap image of a strange {}.',
        'a blurry camera trap image of a {}.',
        'a pixelated camera trap image of a {}.',
        'an image taken with a camera trap of the {}.',
        'a corrupted JPEG camera trap image of the {}.',
        'a well-captured camera trap image of a {}.',
        'a camera trap image of the nice {}.',
        'a camera trap image of the small {}.',
        'a camera trap image of the weird {}.',
        'a camera trap image of the large {}.',
        'a black and white camera trap image of a {}.',
        'a dark camera trap image of a {}.',
        'an image taken with a camera trap of a {}.',
        'an image taken with a camera trap of my {}.',
        'a camera trap image of a cool {}.',
        'a camera trap image of a small {}.',
    ]
def zeroshot_classifier(classnames, templates1, templates2,model_clip,device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates1]
            texts2 = [template for template in templates2[classname]]  # format with class
            texts = texts + texts2
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model_clip.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def extract_features(dataset,mode_clip):
    #path where is located the images
    #dataset='serengeti'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize your models, tokenizer, etc.
    tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_Bert = BertModel.from_pretrained('bert-base-uncased')
    model_Bert.to(device)

    model_clip, preprocess_clip = clip.load(f'ViT-B/{mode_clip}', device)
    model_clip.to(device)

    root=f'data/{dataset}/img'
    carpetas=os.listdir(root)

    f = open(f'data/info_{dataset}.json')
    data = json.load(f)
    camera_trap_templates2 = data['llm_descriptions']
    f.close()
    class_indices=list(camera_trap_templates2.keys())

    features_dataset = {}

    accepted_modes = ['train', 'test', 'val', 'cis_test', 'trans_test', 'cis_val', 'trans_val']
    for mode in carpetas:
        if mode in accepted_modes:

            #class_indices=['aardvark', 'aardwolf', 'baboon', 'batEaredFox', 'buffalo', 'bushbuck', 'caracal', 'cheetah', 'civet','dikDik',
            # 'eland', 'elephant', 'gazelleGrants', 'gazelleThomsons', 'genet', 'giraffe', 'guineaFowl', 'hare','hartebeest', 'hippopotamus',
            #               'honeyBadger', 'hyenaSpotted', 'hyenaStriped', 'impala', 'jackal', 'koriBustard', 'leopard', 'lionFemale',
            # 'lionMale', 'mongoose','ostrich', 'porcupine', 'reedbuck', 'reptiles', 'rhinoceros', 'rodents', 'secretaryBird', 'serval', 'topi',
            #               'vervetMonkey','warthog', 'waterbuck', 'wildcat', 'wildebeest', 'zebra', 'zorilla']

            data_dict={}
            root_dir=os.path.join(root,mode)
            for category in os.listdir(root_dir):
                category_path = os.path.join(root_dir, category)
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    # path where is located the images descriptions generated by LLaVA
                    json_path = os.path.join(f'data/{dataset}/descriptions/{mode}/', category, img_name[:-4] + '.json')

                    if category.isdigit():
                        target_index =int(category)
                    else:
                        target_index = class_indices.index(category)

                    images = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model_clip.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                    # images = images.unsqueeze(0)[0]
                    f = open(json_path)
                    data = json.load(f)
                    description = data['description']
                    f.close()
                    tokens = tokenizer_Bert.tokenize(description)
                    tokens = ['[CLS]'] + tokens + ['[SEP]']
                    attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
                    token_ids = tokenizer_Bert.convert_tokens_to_ids(tokens)

                    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
                    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output_bert = model_Bert(token_ids, attention_mask=attention_mask)
                        description_embeddings = output_bert.pooler_output

                    data_dict[img_name] = {
                        "image_features": image_features,
                        "description_embeddings": description_embeddings,
                        "target_index": target_index
                    }
            # Save the dict in a .pt file
            features_dataset[mode]=data_dict
            #torch.save(data_dict,f'features/Features_{dataset}/standard_features/Features_CATALOG_{mode}_{mode_clip}.pt')

    torch.save(features_dataset, f'features/Features_{dataset}/standard_features/Features_{dataset}.pt')
    zeroshot_weights = zeroshot_classifier(class_indices, camera_trap_templates1, camera_trap_templates2,model_clip,device)
    torch.save(zeroshot_weights,f'features/Features_{dataset}/standard_features/Prompts_{dataset}.pt')





