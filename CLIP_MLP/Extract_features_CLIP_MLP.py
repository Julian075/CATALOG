import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import clip
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

root_dir='/export/jsantamaria/Bases_de_datos/Megadetector'
mode=['train','cis_val','trans_val','cis_test','trans_test']
class_indices = {'badger': 0, 'bird': 1, 'bobcat': 2, 'car': 3, 'cat': 4, 'coyote': 5, 'deer': 6, 'dog': 7, \
                     'empty': 8, 'fox': 9, 'opossum': 10, 'rabbit': 11, 'raccoon': 12, 'rodent': 13, 'skunk': 14,
                     'squirrel': 15}  # Your class indices dictionary

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cpu:":
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    # Create a list of devices
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
else:
    # If CUDA is not available, just use CPU
    devices = [torch.device("cpu")]
    num_gpus=0

print('num_gpus: ',num_gpus,devices)

model_clip, preprocess_clip = clip.load('ViT-B/16', device)
#model_clip = nn.DataParallel(model_clip, device_ids=devices)
model_clip.to(device)

#for mod in mode:
#    data_path=os.path.join(root_dir,mod)
#    data_dict={}
#    for category in os.listdir(data_path):
#        category_path = os.path.join(data_path, category)
#        for img_name in os.listdir(category_path):
#            img_path = os.path.join(category_path, img_name)
#            target_index = class_indices[category.lower()]
#            images = preprocess_clip(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
#
#            with torch.no_grad():
#                image_features = model_clip.encode_image(images)
#                image_features /= image_features.norm(dim=-1, keepdim=True)
#
#            data_dict[img_name] = {
#                "image_features": image_features,
#                "target_index": target_index
#            }
#    # Guarda el diccionario en un archivo .pt
#    torch.save(data_dict,f'Features_CLIP_32_MLP_{mod}.pt')
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
    'a camera trap image of the {} captured in dirty conditions.',
    'a blurry camera trap image of the {}.',
    'a camera trap image of the {}.',
    'a well-captured camera trap image of the {}.',
    'a camera trap image of a single {}.',
    'a camera trap image of a {}.',
    'a low resolution camera trap image of a {}.',
    'a camera trap image of the {} captured in clean conditions.',
    'a camera trap image of a large {}.',
    'a rendition of a {} captured by a camera trap.',
    'a camera trap image of a nice {}.',
    'a camera trap image of a strange {}.',
    'a blurry camera trap image of a {}.',
    'a cartoon version of the {} captured by a camera trap.',
    'art of a {} captured by a camera trap.',
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
camera_trap_templates2 = {

    "badger": [
        "A badger is a mammal with a stout body and short, sturdy legs.",
        "A badger's fur is coarse and typically grayish-black.",
        "badgers often feature a white stripe running from the nose to the back of the head, dividing into two stripes along the sides of the body to the base of the tail.",
        "badgers have broad, flat heads with small eyes and ears.",
        "badger noses are elongated and tapered, ending in a black muzzle.",
        "badger possess strong, well-developed claws adapted for digging burrows.",
        "Overall, badgers have a rugged and muscular appearance suited for their burrowing lifestyle."
    ],

    "bird": [
        "birds have feathers covering their bodies, providing insulation and enabling flight.",
        "birds exhibit a wide range of colors and patterns on their plumage, varying greatly between species.",
        "birds have two legs, adapted for walking, perching, and sometimes swimming.",
        "birds feet come in various shapes and sizes, depending on their ecological niche.",
        "birds possess wings, which are modified forelimbs, allowing them to fly in most species.",
        "The birds beaks or bills are adapted to their feeding habits, ranging from long and slender to short and stout.",
        "birds have keen eyesight, crucial for locating food, navigating, and avoiding predators.",
        "Many species of birds exhibit sexual dimorphism, with males and females having different plumage colors or patterns.",
        "The bird size varies greatly, from the tiny bee hummingbird to the towering ostrich.",
        "Overall, birds display remarkable diversity in appearance, behavior, and ecological adaptations."
    ],
    "bobcat": [
        "bobcats are medium-sized wildcats with a distinctive appearance.",
        "bobcats have short, tawny fur with black spots and streaks, aiding in camouflage.",
        "bobcats fur may also exhibit variations in color, from grayish-brown to reddish-brown.",
        "bobcats have tufted ears with black tips, providing keen hearing and aiding in communication.",
        "bobcats possess whiskers on their face, aiding in detecting prey and navigating their environment.",
        "bobcats have a short, stubby tail, typically with a black tip and barring along its length.",
        "The legs of Bobcat are relatively short compared to their body size, suited for agility and stealth.",
        "bobcats have sharp retractable claws, essential for climbing, hunting, and self-defense.",
        "bobcats eyes are yellowish or amber in color, with slit pupils for enhanced night vision.",
        "Overall, bobcats have a compact, muscular build, adapted for hunting small mammals and birds."
    ],

    "car": [
        "cars are wheeled vehicles designed for transportation.",
        "cars typically have a sleek, aerodynamic body shape for improved efficiency.",
        "cars come in various colors, ranging from vibrant hues to more muted tones.",
        "cars are equipped with headlights at the front for illumination during low-light conditions.",
        "cars also have taillights at the rear, which serve as indicators for braking and turning.",
        "Many cars feature a grille at the front, allowing airflow to cool the engine.",
        "cars are often equipped with side mirrors for the driver to monitor surrounding traffic.",
        "cars have windows made of transparent material, providing visibility for the occupants.",
        "cars come with wheels and tires for movement, with different designs and sizes available.",
        "cars may have decorative elements such as emblems, badges, and trim to enhance aesthetics."
    ],

    "cat": [
        "cats are small mammals known for their agile and graceful movements.",
        "cats typically have a slender body with soft fur covering their skin.",
        "cats come in various colors and patterns, including tabby, calico, and solid.",
        "cats have a distinct head with two pointed ears on top, often capable of rotating.",
        "cats possess sharp, retractable claws on their paws, aiding in climbing and hunting.",
        "cats have whiskers on their face, which are sensitive tactile hairs for navigation.",
        "cats have large, expressive eyes with vertical pupils, providing excellent night vision.",
        "cats have a flexible tail that helps with balance and communication.",
        "cats often groom themselves by licking their fur to keep clean and remove loose hair."
    ],
    "coyote": [
        "coyotes are medium-sized canids with a bushy tail and a pointed muzzle.",
        "coyotes typically have a gray or reddish-brown fur coat with lighter underparts.",
        "coyotes have a lean, athletic build with long legs adapted for running.",
        "coyotes possess keen senses, including acute hearing and a strong sense of smell.",
        "coyotes have a thick fur coat that provides insulation in colder climates.",
        "coyotes have triangular-shaped ears that are erect and highly mobile for detecting sounds.",
        "coyotes have a white throat and belly, which contrasts with the color of their fur.",
        "coyotes have sharp teeth adapted for tearing flesh and crunching bones.",
        "coyotes often travel and hunt in packs, displaying social behavior."
    ],

    "deer": [
        "deer are medium to large-sized ungulates with slender legs and bodies.",
        "deer typically have brown fur with variations in shade depending on the species and season.",
        "deer have a white underside, which contrasts with the color of their fur.",
        "deer possess a distinctive set of antlers, typically branched and regrown annually by males.",
        "deer eyes are large and dark, providing excellent vision, particularly in low light.",
        "deer have a short tail, often barely visible beneath their fur.",
        "deer have sensitive noses, allowing them to detect scents and perceive their surroundings.",
        "deer have hooves that are adapted for swift movement and agile navigation through various terrains.",
        "deer ears are relatively large and mobile, capable of swiveling to detect sounds from different directions."
    ],
    "dog": [
        "dogs come in various sizes, from small breeds like Chihuahuas to large ones like Great Danes.",
        "dogs have a wide range of coat colors and patterns, including black, brown, white, golden, and spotted.",
        "dogs have a furry coat that can be long, short, curly, or wiry depending on the breed.",
        "dogs ears can be erect like those of German Shepherds or floppy like those of Beagles.",
        "dogs have expressive eyes, typically dark brown or amber, reflecting their emotions.",
        "dogs possess a keen sense of smell, with a wet nose that aids in scent detection.",
        "dogs have a tail that varies in length and shape, often wagging to convey happiness or excitement.",
        "dogs teeth are adapted for tearing meat and grinding food, with sharp canines and molars.",
        "dogs have strong, muscular limbs, allowing them to run, jump, and play with ease.",
        "dogs exhibit a wide range of behaviors, from loyal companionship to protective instincts."
    ],
    "fox": [
        "foxes are small to medium-sized canines with slender bodies and pointed snouts.",
        "foxes typically have a coat of reddish-brown fur, although some species may display variations such as silver, gray, or black.",
        "foxes have a white or lighter-colored underside, often extending to their chin and throat.",
        "fox bushy tails are long and often tipped with white, serving various purposes including balance and communication.",
        "foxes possess sharp, pointed ears that are usually erect, aiding in acute hearing for detecting prey and predators.",
        "foxes have distinctive facial markings, including dark patches around the eyes and black accents on the ears and muzzle.",
        "foxes have keen eyesight, with large, amber-colored eyes that enable them to hunt effectively, especially during twilight hours.",
        "fox limbs are slender yet agile, allowing them to navigate through diverse habitats with ease.",
        "foxes exhibit a wide range of vocalizations, including barks, yips, and high-pitched screams, used for communication within their social groups.",
        "foxes are known for their intelligence and adaptability, thriving in various ecosystems from forests and grasslands to urban areas."
    ],

    "opossum": [
        "opossums are marsupials characterized by their rat-like appearance and long, hairless tails.",
        "opossums have pointed snouts and small, rounded ears, resembling rodents in their facial features.",
        "opossum fur is typically grayish-white, sometimes with a slightly yellowish tint, and can appear coarse and unkempt.",
        "opossums have sharp claws on their feet, particularly their hind feet, which they use for climbing and grasping.",
        "opossum tails are prehensile, meaning they can grip and hold onto branches and other objects for stability.",
        "opossums have a distinctive pattern of dark rings on their tails, with each ring becoming progressively lighter towards the tip.",
        "opossum eyes are small and black, with a somewhat vacant or glassy expression.",
        "opossums have long, pointed faces with prominent whiskers, aiding in their nocturnal foraging.",
        "opossums have a pouch in which they carry and nurse their young, as they are marsupials like kangaroos and wallabies."
    ],
    "rabbit": [
        "rabbits are small mammals with soft, dense fur covering their bodies, often in shades of brown, gray, or white.",
        "rabbits have long, upright ears that can rotate to detect sounds from various directions.",
        "rabbit eyes are large, round, and positioned on the sides of their head, providing them with a wide field of vision to detect predators.",
        "rabbits have powerful hind legs designed for hopping and jumping, with long feet and strong claws for traction.",
        "rabbit front legs are shorter and used for digging and manipulating food.",
        "rabbits possess a distinctive twitching nose, which is not only adorable but also serves as a highly sensitive organ to detect scents in their environment.",
        "rabbits have a small, fluffy tail that varies in color depending on the rabbit's breed.",
        "rabbit teeth are continuously growing, so they often chew on objects to keep them worn down and prevent overgrowth.",
        "rabbits have whiskers around their mouth, eyes, and legs, which aid in navigation and sensing their surroundings."
    ],
    "raccoon": [
        "raccoons are medium-sized mammals with a distinctive black 'mask' of fur around their eyes, which contrasts with their grayish-brown fur.",
        "raccoons have a stocky build with a bushy, ringed tail that often has alternating bands of dark and light fur.",
        "raccoon front paws resemble human hands with five dexterous digits, enabling them to manipulate objects and open containers.",
        "raccoons have sharp claws on their front and hind feet, aiding in climbing and grasping.",
        "raccoon ears are rounded and erect, providing keen auditory perception to detect sounds.",
        "raccoon eyes are dark and bright, adapted for night vision and foraging in low-light conditions.",
        "raccoons possess a pointed snout with a highly sensitive nose, allowing them to sniff out food and identify potential threats.",
        "raccoons have a robust body covered in dense fur, providing insulation against cold weather."
    ],
    "rodent": [
        "rodents are characterized by their continuously growing incisors that they must gnaw on to prevent overgrowth.",
        "rodents typically have small, rounded bodies with short legs and long tails, although body size can vary greatly among species.",
        "rodent fur can range in color from gray and brown to black and white, often providing camouflage in their respective habitats.",
        "rodents have relatively large, rounded ears that are sensitive to high-frequency sounds, aiding in detecting predators and communicating with conspecifics.",
        "rodent eyes are typically small and positioned on the sides of their heads, offering a wide field of view to watch for predators.",
        "rodents possess strong, nimble forelimbs equipped with sharp claws for digging, climbing, and manipulating objects.",
        "rodents have a keen sense of smell, using their sensitive noses to locate food, identify mates, and navigate their environment.",
        "Many rodents exhibit prolific breeding habits, with short gestation periods and large litter sizes, allowing them to adapt quickly to changing environmental conditions."
    ],
    "skunk": [
        "skunks are known for their distinctive black and white fur patterns, which serve as a warning to potential predators.",
        "skunks have robust bodies with relatively short legs and long, bushy tails, often held upright.",
        "skunk fur is typically black with one or more white stripes running down their backs, although variations in coloration can occur.",
        "skunks have small, rounded ears and a pointed snout, with prominent white markings on their faces.",
        "skunk eyes are small and dark, positioned on the sides of their heads to provide a wide field of vision.",
        "skunks possess powerful front claws used for digging and foraging for food.",
        "skunks are well-known for their ability to emit a strong-smelling spray from their anal glands as a defense mechanism against predators.",
        "skunks have a keen sense of smell and hearing, which they use to detect food and potential threats in their environment.",
        "skunks are primarily nocturnal animals, venturing out at night to search for food such as insects, small mammals, fruits, and plants."
    ],

    "squirrel": [
        "squirrels are small to medium-sized rodents known for their bushy tails, which they use for balance, communication, and as a form of protection against predators.",
        "squirrels typically have slender bodies with four legs, each ending in sharp claws that aid in climbing trees and grasping food.",
        "squirrels have large, round eyes positioned on the sides of their heads, providing them with excellent peripheral vision.",
        "squirrel fur can vary in color depending on the species, ranging from shades of brown and gray to reddish hues.",
        "Many species of squirrels have distinctive patterns or markings on their fur, such as stripes or patches.",
        "squirrels possess strong hind legs, allowing them to leap between branches and cover considerable distances quickly.",
        "squirrels have prominent front teeth that continuously grow throughout their lives, necessitating constant gnawing to keep them from becoming too long."
    ],
    "empty": [
        "These images typically depict natural landscapes devoid of any visible animals or human presence.",
        "empty Scenes may include forests, fields, mountains, deserts, or bodies of water, showcasing the beauty and serenity of the environment.",
        "empty images is often on capturing the atmosphere, lighting, and elements of the landscape, such as trees, rocks, clouds, and water reflections.",
        "empty images evoke a sense of tranquility and solitude, inviting viewers to immerse themselves in the beauty of nature.",
        "empty images can serve as backgrounds for various purposes, including presentations, websites, and digital artwork.",
        "empty images with only the environment can convey a sense of vastness and the untamed wilderness, reminding viewers of the importance of preserving and appreciating our natural world."
    ]

}


def zeroshot_classifier(classnames, templates1, templates2):
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


zeroshot_weights = zeroshot_classifier(list(class_indices.keys()), camera_trap_templates1, camera_trap_templates2)
torch.save(zeroshot_weights,'Text_features_16.pt')