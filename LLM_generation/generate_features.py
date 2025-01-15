import os
import torch
import clip
import numpy as np


def zeroshot_classifier(classnames, templates1, templates2,omg):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates1]
            texts = clip.tokenize(texts).to(device)
            template_embeddings = model_clip.encode_text(texts)  # embed with text encoder
            template_embeddings /= template_embeddings.norm(dim=-1, keepdim=True)
            template_embeddings = template_embeddings.mean(dim=0)
            template_embeddings /= template_embeddings.norm()

            texts2 = [template for template in templates2[classname]]  # format with class
            texts2 = clip.tokenize(texts2).to(device)
            description_embeddings = model_clip.encode_text(texts2)  # embed with text encoder
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = description_embeddings.mean(dim=0)
            description_embeddings /= description_embeddings.norm()

            class_embedding = template_embeddings *omg + description_embeddings * omg
            class_embedding /= class_embedding.norm()  # Normalize final embedding


            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights




mode_clip=['16']
class_indices_serengeti=['aardvark', 'aardwolf', 'baboon', 'batEaredFox', 'buffalo', 'bushbuck', 'caracal', 'cheetah', 'civet',
             'dikDik',
             'eland', 'elephant', 'gazelleGrants', 'gazelleThomsons', 'genet', 'giraffe', 'guineaFowl', 'hare','hartebeest', 'hippopotamus','honeyBadger', 'hyenaSpotted', 'hyenaStriped', 'impala', 'jackal', 'koriBustard', 'leopard', 'lionFemale',
             'lionMale', 'mongoose','ostrich', 'porcupine', 'reedbuck', 'reptiles', 'rhinoceros', 'rodents', 'secretaryBird', 'serval', 'topi', 'vervetMonkey','warthog', 'waterbuck', 'wildcat', 'wildebeest', 'zebra', 'zorilla']
class_indices_terra = {'badger': 0, 'bird': 1, 'bobcat': 2, 'car': 3, 'cat': 4, 'coyote': 5, 'deer': 6, 'dog': 7, \
                                 'empty': 8, 'fox': 9, 'opossum': 10, 'rabbit': 11, 'raccoon': 12, 'rodent': 13, 'skunk': 14,
                                 'squirrel': 15}
device = "cuda" if torch.cuda.is_available() else "cpu"



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

camera_trap_templates_terra_LLAMA = {
'badger':
             [
                'a badger is a mammal with a stout body and short sturdy legs.',
                'a badger\'s fur is coarse and typically grayish-black.',
                'badgers often feature a white stripe running from the nose to the back of the head dividing into two stripes along the sides of the body to the base of the tail.',
                'badgers have broad flat heads with small eyes and ears.',
                'badger noses are elongated and tapered ending in a black muzzle.',
                'badgers possess strong well-developed claws adapted for digging burrows.',
                'overall badgers have a rugged and muscular appearance suited for their burrowing lifestyle.'
            ]
        ,
        'bird':
            [
                'birds are warm-blooded vertebrates with feathers and wings.',
               'most birds have beaks and lay eggs.',
                'birds have lightweight yet strong skeletons and hollow bones.',
                'birds have a unique respiratory system that allows them to extract oxygen from the air more efficiently than mammals.',
                'birds have a highly efficient metabolism that allows them to survive on a diet of seeds, fruits, and insects.',
                'birds are known for their remarkable ability to migrate long distances each year.',
                'birds have a wide range of sizes, shapes, and colors, making them a diverse group of animals.'
            ]
        ,
        'bobcat':
            [
                'bobcats are small, adaptable predators with a distinctive "bobbed" tail.',
                'bobcats have a thick, sand-colored coat with black spots or stripes.',
                'bobcats have a broad, rounded head with a short, black-tipped tail.',
                'bobcats have large, round eyes and small ears.',
                'bobcats have strong, sharp claws and a powerful bite force.',
                'bobcats are solitary and territorial, with a home range that can vary from 1 to 20 square miles.',
                'bobcats are skilled hunters and can climb trees to stalk their prey.'
            ]
        ,
        'car':
            [
                'cars are vehicles with four wheels, powered by an engine or electric motor.',
                'cars have a body, wheels, and suspension system.',
                'cars have a steering system and brakes to control speed and direction.',
                'cars have a fuel tank and engine compartment.',
                'cars have a transmission system to change gears.',
                'cars have a braking system to slow or stop the vehicle.',
                'cars have a suspension system to absorb shock and maintain stability.'
            ]
        ,
        'cat':
            [
                'cats are small, carnivorous mammals with a flexible spine.',
                'cats have a soft, smooth coat that can be short or long.',
                'cats have a distinctive head shape with a short, rounded muzzle.',
                'cats have large, round eyes and small ears.',
                'cats have retractable claws and a powerful bite force.',
                'cats are known for their agility and ability to climb trees.',
                'cats are often kept as pets and are known for their affectionate nature.'
            ]
        ,
        'coyote':
            [
                'coyotes are medium-sized canines with a grayish-brown coat.',
                'coyotes have a pointed snout, erect ears, and a bushy tail.',
                'coyotes are omnivores and will eat a wide variety of plants and animals.',
                'coyotes are highly adaptable and can be found in a range of habitats.',
                'coyotes are social animals and often live in packs.',
                'coyotes are skilled hunters and can chase down prey over long distances.',
                'coyotes are known for their intelligence and ability to problem-solve.'
            ]
        ,
        'deer':
            [
                'deer are even-toed ungulates with a coat of fur and a set of antlers.',
                'deer have a slender body with a long neck and legs.',
                'deer have large, round eyes and a distinctive set of ears.',
                'deer have a unique digestive system that allows them to extract nutrients from plants.',
                'deer are herbivores and feed on a variety of plants, including leaves, twigs, and grasses.',
                'deer are known for their impressive jumping ability and can leap over 8 feet in a single bound.',
                'deer are often hunted for their meat and antlers, which are used in traditional medicine.'
            ]
        ,
        'dog':
             [
                'dogs are domesticated mammals that are closely related to wolves.',
                'dogs have a varied range of sizes, shapes, and coat types.',
                'dogs have a highly developed sense of smell and hearing.',
                'dogs have a flexible spine and are able to move in a variety of ways.',
                'dogs are highly social animals and thrive on interaction with their human family.',
                'dogs are known for their loyalty and ability to form strong bonds with their owners.',
                'dogs are often used for tasks such as herding, guarding, and assisting people with disabilities.'
            ]
        ,
        'empty':
            [
                'There is no information available about this species.',
                'This species does not exist or is unknown.',
                'No description is available for this species.'
            ]
        ,
        'fox':
            [
                'foxes are small to medium-sized canines with a bushy tail and pointed ears.',
                'foxes have a thick, insulating coat that helps them survive in cold climates.',
                'foxes are omnivores and will eat a wide variety of plants and animals.',
                'foxes are highly adaptable and can be found in a range of habitats.',
                'foxes are known for their cunning and ability to survive in a variety of environments.',
                'foxes are social animals and often live in pairs or small family groups.',
                'foxes are often hunted for their fur and are considered a pest species in some areas.'
            ]
        ,
        'opossum':
            [
                'opossums are small, nocturnal marsupials with a pointed snout and prehensile tail.',
                'opossums have a thick, woolly coat that helps them survive in cold climates.',
                'opossums are omnivores and will eat a wide variety of plants and animals.',
                'opossums are known for their ability to "play dead" when threatened, a behavior known as thanatosis.',
                'opossums are social animals and often live in small family groups.',
                'opossums are important in their ecosystems, helping to control pest populations.',
                'opossums are often considered a nuisance species and are hunted or trapped in some areas.'
            ]
        ,
        'rabbit':
            [
                'rabbits are small, herbivorous mammals with a long ears and a fluffy tail.',
                'rabbits have a distinctive set of teeth that are used for cutting and grinding plant material.',
                'rabbits are social animals and often live in large groups, called warrens.',
                'rabbits are known for their ability to reproduce quickly and can have up to 12 young per litter.',
                'rabbits are important in their ecosystems, helping to disperse seeds and maintain vegetation.',
                'rabbits are often hunted for their meat and are considered a pest species in some areas.',
                'rabbits are popular pets and are known for their gentle nature.'
            ]
        ,
        'raccoon':
            [
                'raccoons are medium-sized mammals with a distinctive black and white mask on their face.',
                'raccoons have a dexterous set of hands and are known for their ability to open complex food containers.',
                'raccoons are omnivores and will eat a wide variety of plants and animals.',
                'raccoons are social animals and often live in small family groups.',
                'raccoons are important in their ecosystems, helping to control pest populations.',
                'raccoons are often considered a nuisance species and are hunted or trapped in some areas.',
                'raccoons are known for their intelligence and ability to adapt to new environments.'
            ]
        ,
        'rodent':
            [
                'rodents are a group of mammals that includes rats, mice, and squirrels.',
                'rodents have a single pair of continuously growing incisors in each jaw.',
                'rodents are found on every continent and in almost every habitat.',
                'rodents are social animals and often live in large groups.',
                'rodents are important in their ecosystems, helping to disperse seeds and maintain vegetation.',
                'rodents are often hunted for their meat and are considered a pest species in some areas.',
                'rodents are known for their ability to adapt to new environments and are often considered pests.'
            ]
        ,
       'skunk':
           [
               'skunks are small to medium-sized mammals with a distinctive black and white striped body.',
               'skunks have a strong, foul-smelling secretion that they use for defense.',
               'skunks are omnivores and will eat a wide variety of plants and animals.',
               'skunks are social animals and often live in small family groups.',
               'skunks are important in their ecosystems, helping to control pest populations.',
               'skunks are often considered a nuisance species and are hunted or trapped in some areas.',
               'skunks are known for their distinctive odor and ability to spray it at predators.'
            ]
        ,
       'squirrel':
           [
               'squirrels are small to medium-sized rodents with a bushy tail and large eyes.',
               'squirrels have a highly developed sense of smell and are able to detect food from a distance.',
               'squirrels are omnivores and will eat a wide variety of plants and animals.',
               'squirrels are social animals and often live in small family groups.',
               'squirrels are important in their ecosystems, helping to disperse seeds and maintain vegetation.',
               'squirrels are often hunted for their meat and are considered a pest species in some areas.',
               'squirrels are known for their ability to climb and jump through trees.'
            ]

    }


camera_trap_templates_serengeti_LLAMA = {
    "aardvark": [
        "aardvarks are nocturnal mammals with a long snout, pointed ears, and a long, thin tail.",
        "their fur is thick and velvety, ranging in color from yellow to brown.",
        "aardvarks have powerful claws for digging and breaking open termite mounds.",
        "they are solitary animals and live in burrows.",
        "aardvarks are insectivores and feed on ants, termites, and other invertebrates.",
        "overall, aardvarks have a unique appearance and behavior adapted to their underground lifestyle."
    ],
    "aardwolf": [
        "aardwolves are nocturnal mammals with a long, pointed snout, and a long, thin tail.",
        "their fur is thick and velvety, ranging in color from yellow to brown.",
        "aardwolves have powerful claws for digging and breaking open termite mounds.",
        "they are solitary animals and live in burrows.",
        "aardwolves are insectivores and feed on ants, termites, and other invertebrates.",
        "overall, aardwolves have a unique appearance and behavior adapted to their underground lifestyle."
    ],
    "baboon": [
        "baboons are primates that live in large troops.",
        "they have thick, coarse fur that ranges in color from brown to gray.",
        "baboons have a distinctive muzzle and a prominent forehead.",
        "they are omnivores and feed on a variety of plants and animals.",
        "baboons are known for their intelligence and complex social behavior.",
        "overall, baboons have a unique appearance and behavior adapted to their social lifestyle."
    ],
    "batEaredFox": [
        "bat-eared foxes are nocturnal mammals with a distinctive pair of ear-like structures on their head.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "bat-eared foxes have powerful claws for digging and breaking open termite mounds.",
        "they are solitary animals and live in burrows.",
        "bat-eared foxes are insectivores and feed on ants, termites, and other invertebrates.",
        "overall, bat-eared foxes have a unique appearance and behavior adapted to their underground lifestyle."
    ],
    "buffalo": [
        "buffalo are large, hooved mammals that live in herds.",
        "they have thick, coarse fur that ranges in color from brown to gray.",
        "buffalo have a distinctive hump on their back and a prominent horn.",
        "they are herbivores and feed on a variety of plants.",
        "buffalo are known for their strength and agility.",
        "overall, buffalo have a unique appearance and behavior adapted to their social lifestyle."
    ],
    "bushbuck": [
        "bushbucks are antelopes that live in dense forests.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "bushbucks have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "bushbucks are known for their agility and ability to climb trees.",
        "overall, bushbucks have a unique appearance and behavior adapted to their forest lifestyle."
    ],
    "caracal": [
        "caracals are wild cats with a distinctive tuft of hair on their ears.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "caracals have powerful claws and sharp teeth.",
        "they are solitary animals and live in burrows.",
        "caracals are carnivores and feed on a variety of small animals.",
        "overall, caracals have a unique appearance and behavior adapted to their hunting lifestyle."
    ],
    "cheetah": [
        "cheetahs are wild cats with a distinctive coat pattern featuring black spots on a yellow background.",
        "their fur is thin and smooth, ranging in color from yellow to brown.",
        "cheetahs have powerful claws and sharp teeth.",
        "they are solitary animals and live in open grasslands.",
        "cheetahs are carnivores and feed on a variety of small animals.",
        "overall, cheetahs have a unique appearance and behavior adapted to their hunting lifestyle."
    ],
    "civet": [
        "civets are mammals with a distinctive coat pattern featuring black and white markings.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "civets have powerful claws and a long, thin tail.",
        "they are solitary animals and live in forests.",
        "civets are omnivores and feed on a variety of plants and animals.",
        "overall, civets have a unique appearance and behavior adapted to their forest lifestyle."
    ],
    "dikDik": [
        "dik-diks are small antelopes that live in dry, open grasslands.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "dik-diks have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "dik-diks are known for their agility and ability to run quickly.",
        "overall, dik-diks have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "eland": [
        "elands are large, hooved mammals that live in herds.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "elands have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "elands are known for their strength and agility.",
        "overall, elands have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "elephant": [
        "elephants are large, hooved mammals that live in herds.",
        "they have thick, coarse fur that ranges in color from gray to brown.",
        "elephants have large, curved tusks.",
        "they are herbivores and feed on a variety of plants.",
        "elephants are known for their intelligence and complex social behavior.",
        "overall, elephants have a unique appearance and behavior adapted to their social lifestyle."
    ],
    "gazelleGrants": [
        "Grant's gazelles are antelopes that live in open grasslands.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "Grant's gazelles have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "Grant's gazelles are known for their agility and ability to run quickly.",
        "overall, Grant's gazelles have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "gazelleThomsons": [
        "Thomson's gazelles are antelopes that live in open grasslands.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "Thomson's gazelles have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "Thomson's gazelles are known for their agility and ability to run quickly.",
        "overall, Thomson's gazelles have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "genet": [
        "genets are mammals with a distinctive coat pattern featuring black and white markings.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "genets have powerful claws and a long, thin tail.",
        "they are solitary animals and live in forests.",
        "genets are omnivores and feed on a variety of plants and animals.",
        "overall, genets have a unique appearance and behavior adapted to their forest lifestyle."
    ],
    "giraffe": [
        "giraffes are large, hooved mammals that live in herds.",
        "they have distinctive coats featuring brown or reddish-brown patches on a cream or white background.",
        "giraffes have long, thin necks and legs.",
        "they are herbivores and feed on a variety of plants.",
        "giraffes are known for their height and ability to reach high branches.",
        "overall, giraffes have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "guineaFowl": [
        "guinea fowl are birds that live in groups.",
        "they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
        "guinea fowl have long, pointed beaks.",
        "they are omnivores and feed on a variety of plants and insects.",
        "guinea fowl are known for their distinctive calls and ability to forage for food.",
        "overall, guinea fowl have a unique appearance and behavior adapted to their social lifestyle."
    ],
    "hare": [
        "hares are long-legged, furry mammals that live in open grasslands.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "hares have long, powerful hind legs and sharp teeth.",
        "they are herbivores and feed on a variety of plants.",
        "hares are known for their speed and ability to run quickly.",
        "overall, hares have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "hartebeest": [
        "hartebeests are antelopes that live in open grasslands.",
        "they have reddish-brown coats with white markings on their face and throat.",
        "hartebeests have large, curved horns.",
        "they are herbivores and feed on a variety of plants.",
        "hartebeests are known for their agility and ability to run quickly.",
        "overall, hartebeests have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "hippopotamus": [
        "hippos are large, semi-aquatic mammals that live in rivers and lakes.",
        "they have thick, coarse fur that ranges in color from gray to brown.",
        "hippos have large, rounded bodies and short legs.",
        "they are herbivores and feed on a variety of plants.",
        "hippos are known for their ability to hold their breath underwater.",
        "overall, hippos have a unique appearance and behavior adapted to their aquatic lifestyle."
    ],
    "honeyBadger": [
        "honey badgers are mammals with a distinctive coat pattern featuring black and white markings.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "honey badgers have powerful claws and a long, thin tail.",
        "they are solitary animals and live in forests.",
        "honey badgers are omnivores and feed on a variety of plants and animals.",
        "overall, honey badgers have a unique appearance and behavior adapted to their forest lifestyle."
    ],
    "hyenaSpotted": [
        "spotted hyenas are carnivorous mammals that live in groups.",
        "they have distinctive coats featuring yellowish-brown fur with black spots.",
        "spotted hyenas have powerful jaws and sharp teeth.",
        "they are scavengers and feed on a variety of carrion.",
        "spotted hyenas are known for their distinctive laugh-like calls.",
        "overall, spotted hyenas have a unique appearance and behavior adapted to their scavenging lifestyle."
    ],
    "hyenaStriped": [
        "striped hyenas are carnivorous mammals that live in groups.",
        "they have distinctive coats featuring yellowish-brown fur with black stripes.",
        "striped hyenas have powerful jaws and sharp teeth.",
        "they are scavengers and feed on a variety of carrion.",
        "striped hyenas are known for their distinctive calls.",
        "overall, striped hyenas have a unique appearance and behavior adapted to their scavenging lifestyle."
    ],
    "jackal": [
        "jackals are carnivorous mammals that live in groups.",
        "they have distinctive coats featuring yellowish-brown fur with black markings.",
        "jackals have powerful jaws and sharp teeth.",
        "they are scavengers and feed on a variety of carrion.",
        "jackals are known for their distinctive howls.",
        "overall, jackals have a unique appearance and behavior adapted to their scavenging lifestyle."
    ],
    "koriBustard": [
        "kori bustards are large, flightless birds that live in open grasslands.",
        "they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
        "kori bustards have long, powerful legs and sharp beaks.",
        "they are omnivores and feed on a variety of plants and insects.",
        "kori bustards are known for their distinctive calls and ability to run quickly.",
        "overall, kori bustards have a unique appearance and behavior adapted to their grassland lifestyle."
    ],
    "leopard": [
        "leopards are wild cats with distinctive coats featuring black spots on a yellow or golden background.",
        "their fur is thick and coarse, ranging in color from yellow to brown.",
        "leopards have powerful claws and sharp teeth.",
        "they are solitary animals and live in forests.",
        "leopards are carnivores and feed on a variety of small animals.",
        "overall, leopards have a unique appearance and behavior adapted to their hunting lifestyle."
    ]
}




omg = np.arange(0, 1.2, 0.2)
for mode_clip_i in mode_clip:
    model_clip, preprocess_clip = clip.load(f'ViT-B/{mode_clip_i}', device)
    model_clip.to(device)
    for omg_i in omg:
        zeroshot_weights = zeroshot_classifier(class_indices_serengeti, camera_trap_templates1, camera_trap_templates_serengeti,omg_i)
        torch.save(zeroshot_weights,f'../features/Features_serengeti/standard_features/Text_{mode_clip_i}_Ab_omg_{omg_i}.pt')

        zeroshot_weights = zeroshot_classifier(list(class_indices_terra.keys()), camera_trap_templates1, camera_trap_templates_terra,omg_i)
        torch.save(zeroshot_weights,f'../features/Features_terra/standard_features/Text_{mode_clip_i}_Ab_omg_{omg_i}.pt')
