import os
import torch
from transformers import BertModel, BertTokenizer
import clip
from feature_extraction.long_Clip.model import longclip
import json
from PIL import Image
import open_clip


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
def zeroshot_classifier(classnames,model_clip,type_clip,device):
    if type_clip=='BioCLIP':
        biocllip_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = f'A photo of a {classname}'
            if type_clip == 'longclip-B':
                texts = longclip.tokenize(texts).to(device)
            elif type_clip == "BioCLIP":
                texts = biocllip_tokenizer(texts).to(device)
            else:
                texts = clip.tokenize(texts).to(device)  # tokenize
            class_embedding = model_clip.encode_text(texts)  # embed with text encoder
            class_embedding /= class_embedding.norm(dim=-1,keepdim=True)  # embed with text encoder
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.squeeze().T
def zeroshot_classifier_2(classnames, templates1, templates2,model_clip,device,type_clip,beta):
    if type_clip=='BioCLIP':
        biocllip_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates1]
            if type_clip == 'longclip-B':
                texts = longclip.tokenize(texts).to(device)
            elif type_clip == "BioCLIP":
                texts = biocllip_tokenizer(texts).to(device)
            else:
                texts = clip.tokenize(texts).to(device)
            template_embeddings = model_clip.encode_text(texts)  # embed with text encoder
            template_embeddings /= template_embeddings.norm(dim=-1, keepdim=True)
            template_embeddings = template_embeddings.mean(dim=0)
            template_embeddings /= template_embeddings.norm(dim=-1, keepdim=True)
            template_embeddings = template_embeddings.unsqueeze(0)

            texts2 = [template for template in templates2[classname]]  # format with class
            if type_clip == 'longclip-B':
                texts2 = longclip.tokenize(texts2).to(device)
            elif type_clip=="BioCLIP":
                texts2 =biocllip_tokenizer(texts2).to(device)
            else:
                texts2 = clip.tokenize(texts2).to(device)

            description_embeddings = model_clip.encode_text(texts2)  # embed with text encoder
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = description_embeddings.mean(dim=0)
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = description_embeddings.unsqueeze(0)

            class_embedding = torch.cat ((template_embeddings *(1-beta),description_embeddings * beta), dim = 0)
            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm()  # Normalize final embedding


            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def extract_features(model_version,dataset,type_clip,LLM='ChatGPT',only_text=0,beta=0.5):
    #path where is located the images
    #dataset='serengeti'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize your models, tokenizer, etc.
    tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_Bert = BertModel.from_pretrained('bert-base-uncased')
    model_Bert.to(device)

    if type_clip == 'longclip-B':
        model_clip, preprocess_clip = longclip.load(f'feature_extraction/long_Clip/checkpoints/{type_clip}.pt', device=device)
    elif type_clip=='BioCLIP':
        model_clip, _, preprocess_clip = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    else:
        model_clip, preprocess_clip = clip.load(f'ViT-B/{type_clip}', device)
    model_clip.to(device)

    root=f'data/{dataset}/img'
    carpetas=os.listdir(root)

    if 'ChatGPT' in LLM:
        f = open(f'data/info_{dataset}_ChatGPT.json')
    else:
        f = open(f'data/info_{dataset}_{LLM}.json')
    data = json.load(f)
    camera_trap_templates2 = data['llm_descriptions']
    f.close()
    class_indices=list(camera_trap_templates2.keys())

    features_dataset = {}
    if only_text==0:
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

                        if model_version == 'Base' or "Base_long":
                            images = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

                            with torch.no_grad():
                                image_features = model_clip.encode_image(images)
                                image_features /= image_features.norm(dim=-1, keepdim=True)

                        if not  ('MLP'in model_version) and not  ('Adapter'in model_version)and not ('zero_shot' in model_version):
                            f = open(json_path)
                            data = json.load(f)
                            description = data['description']
                            f.close()
                            if type_clip == 'longclip-B':
                                tokens = longclip.tokenize(description,truncate=True).to(device)
                                with torch.no_grad():
                                    description_embeddings = model_clip.encode_text(tokens)
                            else:
                                tokens = tokenizer_Bert.tokenize(description)
                                tokens = ['[CLS]'] + tokens + ['[SEP]']
                                attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
                                token_ids = tokenizer_Bert.convert_tokens_to_ids(tokens)

                                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
                                token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    output_bert = model_Bert(token_ids, attention_mask=attention_mask)
                                    description_embeddings = output_bert.pooler_output

                        if model_version == 'Base'or model_version=="Base_long":
                            data_dict[img_name] = {
                                "image_features": image_features,
                                "description_embeddings": description_embeddings,
                                "target_index": target_index
                            }
                        elif 'Fine_tuning' in model_version :
                            data_dict[img_name] = {
                                "image_features": img_path,
                                "description_embeddings": description_embeddings,
                                "target_index": target_index
                            }
                        elif 'MLP' in model_version or 'Adapter' in model_version or 'zero_shot' in model_version:
                            data_dict[img_name] = {
                                "image_features": image_features,
                                "target_index": target_index
                            }
                # Save the dict in a .pt file
                features_dataset[mode]=data_dict

        if dataset=='serengeti':
            #This is a particular case of serengeti dataset
            repeated_keys = features_dataset['train'].keys() & features_dataset['val'].keys()

            for key in repeated_keys:
                del features_dataset['train'][key]

        if model_version== 'Base':
            torch.save(features_dataset, f'features/Features_{dataset}/standard_features/Features_{dataset}.pt')
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2,model_clip,device,type_clip,0.5)
            torch.save(zeroshot_weights,f'features/Features_{dataset}/standard_features/Prompts_{dataset}_{LLM}.pt')

        if model_version== 'Base_long':
            torch.save(features_dataset, f'features/Features_{dataset}/long_features/Features_{dataset}.pt')
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2,model_clip,device,type_clip,0.5)
            torch.save(zeroshot_weights,f'features/Features_{dataset}/long_features/Prompts_{dataset}_{LLM}.pt')

        elif 'Fine_tuning' in model_version :
            torch.save(features_dataset, f'features/Features_{dataset}/finetuning_features/Features_{type_clip}_{dataset}.pt')
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2, model_clip, device, type_clip,0.5)
            torch.save(zeroshot_weights, f'features/Features_{dataset}/finetuning_features/Prompts_{type_clip}_{dataset}_{LLM}.pt')
        elif 'MLP' in model_version or 'Adapter' in model_version or 'zero_shot' in model_version:
            torch.save(features_dataset, f'features/Features_{dataset}/CLIP_MLP/Features_{type_clip}_{dataset}.pt')
            zeroshot_weights = zeroshot_classifier(class_indices, model_clip, type_clip,device)
            torch.save(zeroshot_weights, f'features/Features_{dataset}/CLIP_MLP/Prompts_{type_clip}_{dataset}_{LLM}.pt')
    else:

        if model_version== 'Base':
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2,model_clip,device,type_clip,beta=beta)
            torch.save(zeroshot_weights,f'features/Features_{dataset}/standard_features/Prompts_{dataset}_{LLM}_{beta}.pt')
        elif model_version=="Base_long":
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2, model_clip, device, type_clip,beta)
            torch.save(zeroshot_weights, f'features/Features_{dataset}/long_features/Prompts_{dataset}_{LLM}_{beta}.pt')
        elif 'Fine_tuning' in model_version :
            zeroshot_weights = zeroshot_classifier_2(class_indices, camera_trap_templates1, camera_trap_templates2, model_clip, device, type_clip,beta=beta)
            torch.save(zeroshot_weights, f'features/Features_{dataset}/finetuning_features/Prompts_{type_clip}_{dataset}_{LLM}_{beta}.pt')
        elif 'MLP' in model_version or 'Adapter' in model_version or 'zero_shot' in model_version:
            zeroshot_weights = zeroshot_classifier(class_indices, model_clip,type_clip, device)
            torch.save(zeroshot_weights, f'features/Features_{dataset}/CLIP_MLP/Prompts_{type_clip}_{dataset}.pt')





