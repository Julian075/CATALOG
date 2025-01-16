import os
import torch
import clip
import numpy as np


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
    
'aardvark':
            [
            " aardvarks are nocturnal mammals with a long snout, pointed ears, and a long, thin tail.",
            " their fur is thick and velvety, ranging in color from yellow to brown.",
            " aardvarks have powerful claws for digging and breaking open termite mounds.",
            " they are solitary animals and live in burrows.",
            " aardvarks are insectivores and feed on ants, termites, and other invertebrates.",
            " overall, aardvarks have a unique appearance and behavior adapted to their underground lifestyle.",
        ], 'aardwolf':
            [
            " aardwolves are nocturnal mammals with a long, pointed snout, and a long, thin tail.",
            " their fur is thick and velvety, ranging in color from yellow to brown.",
            " aardwolves have powerful claws for digging and breaking open termite mounds.",
            " they are solitary animals and live in burrows.",
            " aardwolves are insectivores and feed on ants, termites, and other invertebrates.",
            " overall, aardwolves have a unique appearance and behavior adapted to their underground lifestyle.",
        ], 'baboon':
            [
            " baboons are primates that live in large troops.",
            " they have thick, coarse fur that ranges in color from brown to gray.",
            " baboons have a distinctive muzzle and a prominent forehead.",
            " they are omnivores and feed on a variety of plants and animals.",
            " baboons are known for their intelligence and complex social behavior.",
            " overall, baboons have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'batEaredFox':
            [
            " bat-eared foxes are nocturnal mammals with a distinctive pair of ear-like structures on their head.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " bat-eared foxes have powerful claws for digging and breaking open termite mounds.",
            " they are solitary animals and live in burrows.",
            " bat-eared foxes are insectivores and feed on ants, termites, and other invertebrates.",
            " overall, bat-eared foxes have a unique appearance and behavior adapted to their underground lifestyle.",
        ], 'buffalo':
            [
            " buffalo are large, hooved mammals that live in herds.",
            " they have thick, coarse fur that ranges in color from brown to gray.",
            " buffalo have a distinctive hump on their back and a prominent horn.",
            " they are herbivores and feed on a variety of plants.",
            " buffalo are known for their strength and agility.",
            " overall, buffalo have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'bushbuck':
            [
            " bushbucks are antelopes that live in dense forests.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " bushbucks have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " bushbucks are known for their agility and ability to climb trees.",
            " overall, bushbucks have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'caracal':
            [
            " caracals are wild cats with a distinctive tuft of hair on their ears.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " caracals have powerful claws and sharp teeth.",
            " they are solitary animals and live in burrows.",
            " caracals are carnivores and feed on a variety of small animals.",
            " overall, caracals have a unique appearance and behavior adapted to their hunting lifestyle.",
        ], 'cheetah':
            [
            " cheetahs are wild cats with a distinctive coat pattern featuring black spots on a yellow background.",
            " their fur is thin and smooth, ranging in color from yellow to brown.",
            " cheetahs have powerful claws and sharp teeth.",
            " they are solitary animals and live in open grasslands.",
            " cheetahs are carnivores and feed on a variety of small animals.",
            " overall, cheetahs have a unique appearance and behavior adapted to their hunting lifestyle.",
        ], 'civet':
            [
            " civets are mammals with a distinctive coat pattern featuring black and white markings.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " civets have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " civets are omnivores and feed on a variety of plants and animals.",
            " overall, civets have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'dikDik':
            [
            " dik-diks are small antelopes that live in dry, open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " dik-diks have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " dik-diks are known for their agility and ability to run quickly.",
            " overall, dik-diks have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'eland':
            [
            " elands are large, hooved mammals that live in herds.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " elands have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " elands are known for their strength and agility.",
            " overall, elands have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'elephant':
            [
            " elephants are large, hooved mammals that live in herds.",
            " they have thick, coarse fur that ranges in color from gray to brown.",
            " elephants have large, curved tusks.",
            " they are herbivores and feed on a variety of plants.",
            " elephants are known for their intelligence and complex social behavior.",
            " overall, elephants have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'gazelleGrants':
            [
            " Grant's gazelles are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " Grant's gazelles have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " Grant's gazelles are known for their agility and ability to run quickly.",
            " overall, Grant's gazelles have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'gazelleThomsons':
            [
            " Thomson's gazelles are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " Thomson's gazelles have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " Thomson's gazelles are known for their agility and ability to run quickly.",
            " overall, Thomson's gazelles have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'genet':
            [
            " genets are mammals with a distinctive coat pattern featuring black and white markings.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " genets have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " genets are omnivores and feed on a variety of plants and animals.",
            " overall, genets have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'giraffe':
            [
            " giraffes are large, hooved mammals that live in herds.",
            " they have distinctive coats featuring brown or reddish-brown patches on a cream or white background.",
            " giraffes have long, thin necks and legs.",
            " they are herbivores and feed on a variety of plants.",
            " giraffes are known for their height and ability to reach high branches.",
            " overall, giraffes have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'guineaFowl':
            [
            " guinea fowl are birds that live in groups.",
            " they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
            " guinea fowl have long, pointed beaks.",
            " they are omnivores and feed on a variety of plants and insects.",
            " guinea fowl are known for their distinctive calls and ability to forage for food.",
            " overall, guinea fowl have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'hare':
            [
            " hares are long-legged, furry mammals that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " hares have long, powerful hind legs and sharp teeth.",
            " they are herbivores and feed on a variety of plants.",
            " hares are known for their speed and ability to run quickly.",
            " overall, hares have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'hartebeest':
            [
            " hartebeests are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " hartebeests have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " hartebeests are known for their agility and ability to run quickly.",
            " overall, hartebeests have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'hippopotamus':
            [
            " hippos are large, semi-aquatic mammals that live in rivers and lakes.",
            " they have thick, coarse fur that ranges in color from gray to brown.",
            " hippos have large, rounded bodies and short legs.",
            " they are herbivores and feed on a variety of plants.",
            " hippos are known for their ability to hold their breath underwater.",
            " overall, hippos have a unique appearance and behavior adapted to their aquatic lifestyle.",
        ], 'honeyBadger':
            [
            " honey badgers are mammals with a distinctive coat pattern featuring black and white markings.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " honey badgers have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " honey badgers are omnivores and feed on a variety of plants and animals.",
            " overall, honey badgers have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'hyenaSpotted':
            [
            " spotted hyenas are carnivorous mammals that live in groups.",
            " they have distinctive coats featuring yellowish-brown fur with black spots.",
            " spotted hyenas have powerful jaws and sharp teeth.",
            " they are scavengers and feed on a variety of carrion.",
            " spotted hyenas are known for their distinctive laugh-like calls.",
            " overall, spotted hyenas have a unique appearance and behavior adapted to their scavenging lifestyle.",
        ], 'hyenaStriped':
            [
            " striped hyenas are carnivorous mammals that live in groups.",
            " they have distinctive coats featuring yellowish-brown fur with black stripes.",
            " striped hyenas have powerful jaws and sharp teeth.",
            " they are scavengers and feed on a variety of carrion.",
            " striped hyenas are known for their distinctive calls.",
            " overall, striped hyenas have a unique appearance and behavior adapted to their scavenging lifestyle.",
        ], 'impala':
            [
            " impalas are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " impalas have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " impalas are known for their agility and ability to run quickly.",
            " overall, impalas have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'jackal':
            [
            " jackals are carnivorous mammals that live in groups.",
            " they have distinctive coats featuring yellowish-brown fur with black markings.",
            " jackals have powerful jaws and sharp teeth.",
            " they are scavengers and feed on a variety of carrion.",
            " jackals are known for their distinctive howls.",
            " overall, jackals have a unique appearance and behavior adapted to their scavenging lifestyle.",
        ], 'koriBustard':
            [
            " kori bustards are large, flightless birds that live in open grasslands.",
            " they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
            " kori bustards have long, powerful legs and sharp beaks.",
            " they are omnivores and feed on a variety of plants and insects.",
            " kori bustards are known for their distinctive calls and ability to run quickly.",
            " overall, kori bustards have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'leopard':
            [
            " leopards are wild cats with distinctive coats featuring black spots on a yellow or golden background.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " leopards have powerful claws and sharp teeth.",
            " they are solitary animals and live in forests.",
            " leopards are carnivores and feed on a variety of small animals.",
            " overall, leopards have a unique appearance and behavior adapted to their hunting lifestyle.",
        ], 'lionFemale':
            [
            " lionesses are female lions that live in prides.",
            " they have distinctive coats featuring golden or tawny fur with a long mane.",
            " lionesses have powerful jaws and sharp teeth.",
            " they are carnivores and feed on a variety of small animals.",
            " lionesses are known for their social behavior and cooperative hunting.",
            " overall, lionesses have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'lionMale':
            [
            " lions are male lions that live in prides.",
            " they have distinctive coats featuring golden or tawny fur with a long mane.",
            " lions have powerful jaws and sharp teeth.",
            " they are carnivores and feed on a variety of small animals.",
            " lions are known for their social behavior and dominant role in their prides.",
            " overall, lions have a unique appearance and behavior adapted to their social lifestyle.",
        ],'mongoose':
            [
            " mongooses are mammals with a distinctive coat pattern featuring black and white markings.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " mongooses have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " mongooses are omnivores and feed on a variety of plants and animals.",
            " overall, mongooses have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'ostrich':
            [
            " ostriches are large, flightless birds that live in open grasslands.",
            " they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
            " ostriches have long, powerful legs and sharp beaks.",
            " they are omnivores and feed on a variety of plants and insects.",
            " ostriches are known for their distinctive calls and ability to run quickly.",
            " overall, ostriches have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'porcupine':
            [
            " porcupines are mammals with a distinctive coat pattern featuring black and white quills.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " porcupines have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " porcupines are herbivores and feed on a variety of plants.",
            " overall, porcupines have a unique appearance and behavior adapted to their forest lifestyle.",
        ],'reedbuck':
            [
            " reedbuck are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " reedbuck have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " reedbuck are known for their agility and ability to run quickly.",
            " overall, reedbuck have a unique appearance and behavior adapted to their grassland lifestyle.",
        ],'reptiles':
            [
            " reptiles are a group of animals that include snakes, lizards, and turtles.",
            " they have scaly skin and lay eggs.",
            " reptiles have a wide range of sizes, shapes, and colors.",
            " they are found in a variety of habitats, including forests, grasslands, and deserts.",
            " reptiles are carnivores and feed on a variety of small animals.",
            " overall, reptiles have a unique appearance and behavior adapted to their environments.",
        ], 'rhinoceros':
            [
            " rhinoceroses are large, hooved mammals that live in herds.",
            " they have distinctive coats featuring gray or brown fur with white or black markings.",
            " rhinoceroses have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " rhinoceroses are known for their strength and ability to run quickly.",
            " overall, rhinoceroses have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'rodents':
            [
            " rodents are a group of mammals that include mice, rats, and squirrels.",
            " they have distinctive coats featuring brown or gray fur with white or black markings.",
            " rodents have large, prominent eyes and ears.",
            " they are herbivores and feed on a variety of plants.",
            " rodents are known for their ability to store food and adapt to their environments.",
            " overall, rodents have a unique appearance and behavior adapted to their environments.",
        ],'secretaryBird':
            [
            " secretary birds are large, predatory birds that live in open grasslands.",
            " they have distinctive coats featuring brown or reddish-brown feathers with white or black markings.",
            " secretary birds have long, powerful legs and sharp beaks.",
            " they are carnivores and feed on a variety of small animals.",
            " secretary birds are known for their distinctive calls and ability to run quickly.",
            " overall, secretary birds have a unique appearance and behavior adapted to their grassland lifestyle.",
        ],'serval':
            [
            " servals are wild cats with distinctive coats featuring black spots on a yellow or golden background.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " servals have powerful claws and sharp teeth.",
            " they are solitary animals and live in forests.",
            " servals are carnivores and feed on a variety of small animals.",
            " overall, servals have a unique appearance and behavior adapted to their hunting lifestyle.",
        ], 'topi':
            [
            " topi are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " topi have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " topi are known for their agility and ability to run quickly.",
            " overall, topi have a unique appearance and behavior adapted to their grassland lifestyle.",
        ],'vervetMonkey':
            [
            " vervet monkeys are primates that live in groups.",
            " they have distinctive coats featuring brown or gray fur with white or black markings.",
            " vervet monkeys have long, thin tails and large, prominent eyes.",
            " they are omnivores and feed on a variety of plants and animals.",
            " vervet monkeys are known for their intelligence and complex social behavior.",
            " overall, vervet monkeys have a unique appearance and behavior adapted to their social lifestyle.",
        ], 'warthog':
            [
            " warthogs are large, hooved mammals that live in herds.",
            " they have distinctive coats featuring gray or brown fur with white or black markings.",
            " warthogs have large, curved tusks.",
            " they are herbivores and feed on a variety of plants.",
            " warthogs are known for their strength and ability to run quickly.",
            " overall, warthogs have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'waterbuck':
            [
            " waterbucks are antelopes that live in open grasslands.",
            " they have reddish-brown coats with white markings on their face and throat.",
            " waterbucks have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " waterbucks are known for their agility and ability to run quickly.",
            " overall, waterbucks have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'wildcat':
            [
            " wildcats are small, carnivorous mammals that live in forests.",
            " they have distinctive coats featuring brown or gray fur with white or black markings.",
            " wildcats have powerful claws and sharp teeth.",
            " they are solitary animals and feed on a variety of small animals.",
            " wildcats are known for their agility and ability to climb trees.",
            " overall, wildcats have a unique appearance and behavior adapted to their forest lifestyle.",
        ], 'wildebeest':
            [
            " wildebeest are large, hooved mammals that live in herds.",
            " they have distinctive coats featuring gray or brown fur with white or black markings.",
            " wildebeest have large, curved horns.",
            " they are herbivores and feed on a variety of plants.",
            " wildebeest are known for their strength and ability to run quickly.",
            " overall, wildebeest have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'zebra':
            [
            " zebras are equines that live in open grasslands.",
            " they have distinctive coats featuring black and white stripes.",
            " zebras have large, prominent eyes and ears.",
            " they are herbivores and feed on a variety of plants.",
            " zebras are known for their agility and ability to run quickly.",
            " overall, zebras have a unique appearance and behavior adapted to their grassland lifestyle.",
        ], 'zorilla':
            [
            " zorillas are mammals with a distinctive coat pattern featuring black and white markings.",
            " their fur is thick and coarse, ranging in color from yellow to brown.",
            " zorillas have powerful claws and a long, thin tail.",
            " they are solitary animals and live in forests.",
            " zorillas are omnivores and feed on a variety of plants and animals.",
            " overall, zorillas have a unique appearance and behavior adapted to their forest lifestyle.",
    ]
}

camera_trap_templates_serengeti_Phi = {

"aardvark":
    ["an aardvark is a nocturnal mammal with a cylindrical body and a long sticky tongue.",
    "aardvarks have a sandy brown to grayish-black fur that helps them blend into their arid habitats.",
    "their ears are small and rounded, and they lack external ears.",
    "aardvarks possess powerful claws for digging and foraging for ants and termites.",
    "they have a distinctive snout and a long, flexible tail that aids in balance while burrowing.",
    "aardvarks have a stout, muscular build with a short, stout tail and a broad head.",
    "their eyes are small and set far back on the head, providing limited vision but excellent night vision.",
    "aardvarks are solitary animals, with a diet primarily consisting of ants and termites"],
"aardwolf":
    ["an aardwolf is a small carnivorous mammal related to the hyena family.",
    "aardwolves have a reddish-brown fur with white markings on their chest and belly.",
    "they have a slender build with a long, narrow snout and small, rounded ears.",
    "aardwolves possess long, powerful legs and sharp claws for digging and hunting.",
    "their eyes are small and set far back on the head, providing limited vision but excellent night vision.",
    "aardwolves are primarily insectivorous, feeding on termites and other insects.",
    "they are social animals, living in small family groups and communicating through vocalizations and scent marking."],
"baboon":
    ["baboons are large, terrestrial primates with a robust build and a long, dog-like muzzle.",
    "they have a distinctive coloration, with a mix of black, brown, and white fur.",
    "baboons have a long, muscular tail and a prominent rump, which they use for communication and display.",
    "their eyes are large and set on the sides of the head, providing a wide field of vision.",
    "baboons have a complex social structure, living in large troops with a dominant male and multiple females.",
    "they are omnivorous, feeding on a variety of plant and animal matter, including fruits, leaves, insects, and small vertebrates.",
    "baboons are highly intelligent and have been observed using tools, such as sticks to extract termites from mounds."],
"batEaredFox":
    [ "the bat-eared fox is a small, carnivorous mammal native to Africa and Asia.",
    "it has a distinctive appearance, with large, bat-like ears and a bushy tail.",
    "the bat-eared fox has a sandy-colored fur with black markings on its face and legs.",
    "it has a slender build with a long, narrow snout and small, rounded ears.",
    "the bat-eared fox is a nocturnal animal, hunting small mammals and birds.",
    "it is a solitary animal, with a diet primarily consisting of insects, particularly termites.",
    "the bat-eared fox is well adapted to arid environments, with its large ears helping to dissipate heat and its thick fur providing insulation."],

"buffalo":
    ["buffalo are large, herbivorous mammals with a distinctive hump on their shoulders and a long, curved horn.",
    "they have a thick, dark brown to reddish-brown fur, with a lighter color on their underbelly.",
    "buffalo have a broad, flat head with small, rounded ears and a short, muscular neck.",
    "they have a large, heavy body with strong legs and hooves adapted for grazing on grasses.",
    "buffalo are social animals, living in large herds led by a dominant bull.",
    "they are primarily grazers, feeding on grasses and other vegetation.",
    "buffalo are known for their aggressive behavior, particularly when defending their herd or territory."],

"bushbuck":[
    "bushbucks are medium-sized antelopes with a distinctive, spotted coat and a long, slender body.",
    "they have a reddish-brown to grayish-brown fur, with a white rump and a black-tipped tail.",
    "bushbucks have a slender build with a long, graceful neck and small, rounded ears.",
    "they have a distinctive, spiral-shaped horn on the forehead, which is shed and regrown annually.",
    "bushbucks are primarily browsers, feeding on leaves, fruits, and flowers.",
    "they are solitary animals, with a diet consisting of a variety of plant matter.",
    "bushbucks are well adapted to their arid habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"caracal":[
"caracals are medium-sized wild cats with a distinctive coat pattern and long, tufted ears.",
"they have a sandy-colored fur with black spots and stripes, and a white chest and belly.",
"caracals have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, tufted ear, with long, black tufts on the tips.",
"caracals are solitary animals, with a diet consisting of small mammals, birds, and reptiles.",
"they are skilled hunters, using their agility and speed to chase down prey.",
"caracals are well adapted to their arid habitats, with their long legs and powerful hind legs allowing them to move quickly and efficiently."],

"cheetah":[
"cheetahs are the fastest land animals, with a slender, muscular body and long, powerful legs.",
"they have a distinctive coat pattern, with black spots on a yellowish-tan background.",
"cheetahs have a long, slender body with a small head and a long, flexible spine.",
"they have a distinctive, black tear mark on their face, which helps to reduce glare when hunting in the sun.",
"cheetahs are primarily hunters, using their speed and agility to chase down prey.",
"they are solitary animals, with a diet consisting of small to medium-sized ungulates.",
"cheetahs are well adapted to their open habitats, with their long legs and flexible spine allowing them to run at high speeds."],

"civet":[
"civets are small, carnivorous mammals with a distinctive, musky odor and a long, bushy tail.",
"they have a reddish-brown to grayish-brown fur, with a white chest and belly.",
"civets have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, bushy tail, which they use for balance and communication.",
"civets are primarily nocturnal animals, hunting for small mammals, birds, and reptiles.",
"they are solitary animals, with a diet consisting of a variety of animal matter.",
"civets are well adapted to their arboreal habitats, with their long, flexible spine and powerful hind legs allowing them to move quickly and efficiently through the trees."],

"dikDik":[
"dik-diks are small, antelope-like mammals with a distinctive coat pattern and a long, slender body.",
"they have a reddish-brown to grayish-brown fur, with a white chest and belly.",
"dik-diks have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, slender tail, which they use for balance and communication.",
"dik-diks are primarily herbivores, feeding on leaves, fruits, and flowers.",
"they are social animals, living in small family groups.",
"dik-diks are well adapted to their arid habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"eland":[
"elands are large, antelope-like mammals with a distinctive coat pattern and a long, slender body.",
"they have a reddish-brown to grayish-brown fur, with a white chest and belly.",
"elands have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, slender tail, which they use for balance and communication.",
"elands are primarily herbivores, feeding on grasses and other vegetation.",
"they are social animals, living in large herds.",
"elands are well adapted to their arid habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"elephant":[
"elephants are the largest land animals, with a distinctive, long trunk and large, curved tusks.",
"they have a grayish-brown to black fur, with a white rump and a patch of lighter fur on their face.",
"elephants have a large, heavy body with a long, flexible trunk and powerful legs.",
"they have a distinctive, large, fan-shaped tail, which they use for balance and communication.",
"elephants are primarily herbivores, feeding on grasses, leaves, and bark.",
"they are social animals, living in large herds led by a dominant bull.",
"elephants are well adapted to their diverse habitats, with their large size and powerful limbs allowing them to move through a variety of environments."],

"gazelleGrants":[
"gazelleGrants are small, antelope-like mammals with a distinctive coat pattern and a long, slender body.",
"they have a reddish-brown to grayish-brown fur, with a white chest and belly.",
"gazelleGrants have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, slender tail, which they use for balance and communication.",
"gazelleGrants are primarily herbivores, feeding on grasses and other vegetation.",
"they are social animals, living in small family groups.",
"gazelleGrants are well adapted to their arid habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"gazelleThomsons":[
"gazelleThomsons are small, antelope-like mammals with a distinctive coat pattern and a long, slender body.",
"they have a reddish-brown to grayish-brown fur, with a white chest and belly.",
"gazelleThomsons have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, slender tail, which they use for balance and communication.",
"gazelleThomsons are primarily herbivores, feeding on grasses and other vegetation.",
"they are social animals, living in small family groups.",
"gazelleThomsons are well adapted to their arid habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"genet":[
"genets are small, carnivorous mammals with a distinctive coat pattern and a long, slender body.",
"they have a reddish-brown to grayish-brown fur, with black spots and stripes.",
"genets have a slender, muscular body with a long, flexible spine and powerful hind legs.",
"they have a distinctive, long, slender tail, which they use for balance and communication.",
"genets are primarily nocturnal animals, hunting for small mammals, birds, and reptiles.",
"they are solitary animals, with a diet consisting of a variety of animal matter.",
"genets are well adapted to their arboreal habitats, with their long, flexible spine and powerful hind legs allowing them to move quickly and efficiently through the trees."],

"giraffe":[
"giraffes are the tallest land animals, with a distinctive coat pattern and a long, slender neck.",
"they have a reddish-brown to grayish-brown fur, with black spots and stripes.",
"giraffes have a slender, muscular body with a long, flexible neck and powerful legs.",
"they have a distinctive, long, slender neck, which they use for reaching high branches and browsing on leaves.",
"giraffes are primarily herbivores, feeding on leaves, fruits, and flowers.",
"they are social animals, living in small family groups.",
"giraffes are well adapted to their savannah habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"guineaFowl":[
"guineaFowls are small, ground-dwelling birds with a distinctive plumage and a long, slender neck.",
"they have a black to grayish-brown plumage, with white spots and stripes.",
"guineaFowls have a slender, muscular body with a long, flexible neck and powerful legs.",
"they have a distinctive, long, slender neck, which they use for foraging for food on the ground.",
"guineaFowls are primarily omnivores, feeding on insects, seeds, and small vertebrates.",
"they are social animals, living in small family groups.",
"guineaFowls are well adapted to their savannah habitats, with their long legs and slender body allowing them to move quickly and efficiently."],

"hare":[
"a hare is a mammal with long powerful hind legs and a short tail.",
"hares have soft fur that is typically brown or gray, with white markings on the underparts.",
"hares have large, round ears that can move independently to detect predators.",
"hares have long, agile bodies that allow them to run at high speeds.",
"hares have a long, bushy tail that helps them balance while running.",
"hares have strong, sharp teeth for gnawing on tough plant material.",
"overall hares have a sleek and athletic appearance suited for their fast-paced lifestyle."],

"hartebeest":[
"a hartebeest is a large antelope with a distinctive curved horn in both sexes.",
"hartebeests have a light brown coat with white underparts and a black stripe running from the face to the rump.",
"hartebeests have long, slender legs and a long, graceful neck.",
"hartebeests have a large, flat head with a prominent forehead and small, pointed ears.",
"hartebeests have a long, slender muzzle and a long, thin tail.",
"hartebeests have a powerful, muscular build that allows them to run at high speeds.",
"overall hartebeests have a majestic and elegant appearance suited for their open grassland habitat."],

"hippopotamus":[
"a hippopotamus is a large, semi-aquatic mammal with a barrel-shaped body and short legs.",
"hippos have thick, grayish-brown skin that is smooth and wrinkled, with sparse hairs.",
"hippos have large, protruding eyes and nostrils that are located on the top of their heads.",
"hippos have a large, heavy head with a prominent snout and large, sharp tusks.",
"hippos have a short, muscular tail that is used for swatting away flies and other insects.",
"hippos have four sturdy, pillar-like legs that are adapted for walking and wading in water.",
"overall hippos have a bulky and muscular appearance suited for their aquatic lifestyle."],

"honeyBadger":[
"a honeybadger is a small, robust mammal with a stocky body and short legs.",
"honeybadgers have a thick, coarse fur that is typically grayish-brown with a black stripe running from the forehead to the base of the tail.",
"honeybadgers have a short, rounded snout and small, rounded ears.",
"honeybadgers have a long, muscular tail that is used for balance and defense.",
"honeybadgers have strong, well-developed claws that are adapted for digging and climbing.",
"honeybadgers have a compact and muscular build that allows them to withstand attacks from predators.",
"overall honeybadgers have a tough and resilient appearance suited for their harsh desert habitat."],

"hyenaSpotted":[
"a hyena spotted is a medium-sized carnivore with a long, slender body and a bushy tail.",
"hyenas spotted have a tawny coat with black spots and stripes.",
"hyenas spotted have a long, narrow head with a pointed snout and small, rounded ears.",
"hyenas spotted have a long, muscular neck and a powerful jaw with sharp teeth.",
"hyenas spotted have long, powerful legs that are adapted for running and jumping.",
"hyenas spotted have a long, bushy tail that is used for balance and communication.",
"overall hyenas spotted have a sleek and agile appearance suited for their fast-paced lifestyle."],

"hyenaStriped":[
"a hyena striped is a medium-sized carnivore with a long, slender body and a bushy tail.",
"hyenas striped have a tawny coat with black stripes.",
"hyenas striped have a long, narrow head with a pointed snout and small, rounded ears.",
"hyenas striped have a long, muscular neck and a powerful jaw with sharp teeth.",
"hyenas striped have long, powerful legs that are adapted for running and jumping.",
"hyenas striped have a long, bushy tail that is used for balance and communication.",
"overall hyenas striped have a sleek and agile appearance suited for their fast-paced lifestyle."],

"impala":[
"an impala is a medium-sized antelope with a slender body and long, graceful legs.",
"impalas have a reddish-brown coat with white underparts and a black stripe running from the face to the rump.",
"impalas have long, slender ears that can move independently to detect predators.",
"impalas have a long, slender neck and a long, thin tail.",
"impalas have a powerful, muscular build that allows them to run at high speeds.",
"impalas have a slender and elegant appearance suited for their open grassland habitat."],

"jackal":[
"a jackal is a medium-sized carnivore with a slender body and long legs.",
"jackals have a tawny coat with black markings on the back and a bushy tail.",
"jackals have a long, narrow head with a pointed snout and small, rounded ears.",
"jackals have a long, muscular neck and a powerful jaw with sharp teeth.",
"jackals have long, powerful legs that are adapted for running and jumping.",
"jackals have a bushy tail that is used for balance and communication.",
"overall jackals have a sleek and agile appearance suited for their fast-paced lifestyle."],

"koriBustard":[
"a kori bustard is a large, flightless bird with a long, slender body and long legs.",
"kori bustards have a mottled brown and gray plumage with a white underbelly.",
"kori bustards have a long, slender neck and a long, thin tail.",
"kori bustards have a large, round head with a prominent beak and small, rounded ears.",
"kori bustards have strong, powerful legs that are adapted for running and jumping.",
"kori bustards have a slender and elegant appearance suited for their open grassland habitat."],

"leopard":[
"a leopard is a large, solitary cat with a muscular body and long legs.",
"leopards have a tawny coat with black spots and rosettes.",
"leopards have a long, slender head with a prominent snout and small, rounded ears.",
"leopards have a long, muscular neck and a powerful jaw with sharp teeth.",
"leopards have long, powerful legs that are adapted for running and climbing.",
"leopards have a compact and muscular build that allows them to withstand attacks from predators.",
"overall leopards have a sleek and agile appearance suited for their stealthy hunting lifestyle."],

"lionFemale":[
"a lion female is a large, powerful cat with a tawny coat and a distinctive mane.",
"lion females have a shorter, less dense mane than male lions.",
"lion females have a long, slender head with a prominent snout and small, rounded ears.",
"lion females have a long, muscular neck and a powerful jaw with sharp teeth.",
"lion females have long, powerful legs that are adapted for running and climbing.",
"lion females have a compact and muscular build that allows them to withstand attacks from predators.",
"overall lion females have a sleek and agile appearance suited for their role as hunters and protectors of their pride."],

"lionMale":[
"a lion male is a large, powerful cat with a tawny coat and a distinctive mane.",
"lion males have a longer, more dense mane than female lions.",
"lion males have a long, slender head with a prominent snout and small, rounded ears.",
"lion males have a long, muscular neck and a powerful jaw with sharp teeth.",
"lion males have long, powerful legs that are adapted for running and climbing.",
"lion males have a compact and muscular build that allows them to withstand attacks from predators.",
"overall lion males have a sleek and agile appearance suited for their role as leaders of their pride and protectors of their territory."],

"mongoose":[
"a mongoose is a small, agile mammal with a slender body and long legs.",
"mongooses have a sandy or grayish coat with black markings on the back and a bushy tail.",
"mongooses have a long, slender head with a pointed snout and small, rounded ears.",
"mongooses have a long, muscular neck and a powerful jaw with sharp teeth.",
"mongooses have long, powerful legs that are adapted for running and jumping.",
"mongooses have a slender and agile appearance suited for their fast-paced lifestyle."],

"ostrich":[
"an ostrich is a large, flightless bird with a long, slender body and long legs.",
"ostriches have a mottled brown and gray plumage with a white underbelly.",
"ostriches have a long, slender neck and a long, thin tail.",
"ostriches have a large, round head with a prominent beak and small, rounded ears.",
"ostriches have long, powerful legs that are adapted for running and jumping.",
"ostriches have a slender and elegant appearance suited for their open grassland habitat."],

"porcupine":[
    " a porcupine is a rodent with a stocky build and a prehensile tail.",
    " porcupines have a coat of coarse, bristly quills that cover their back and tail.",
    " the quills are hollow and barbed, providing protection against predators.",
    " porcupines have small, round ears and a short, flat snout.",
    " their eyes are large and set far apart, providing a wide field of vision.",
    " porcupines have a relatively small head compared to their body size.",
    " their tails are long and covered in soft fur, used for balance and communication.",
    " overall, porcupines have a distinctive appearance with their quills and prehensile tail."],

"reedbuck":[
"a reedbuck is a medium-sized antelope with a slender build and long legs.",
"reedbucks have a reddish-brown coat with white underparts and a distinctive black rump.",
"their antlers are small and shed annually.",
"reedbucks have long, slender ears and a long, thin muzzle.",
"their eyes are set wide apart, providing a wide field of vision.",
"reedbucks have a long, slender tail with a tuft of hair at the tip.",
"overall, reedbucks have a graceful appearance with their slender build and long legs."],

"reptiles":[
"reptiles are a diverse group of cold-blooded vertebrates with scaly skin.",
"reptiles include snakes, lizards, turtles, and crocodiles.",
"reptiles have a wide range of body shapes and sizes, from tiny geckos to massive crocodiles.",
"reptiles have a variety of adaptations for survival, including camouflage, venom, and protective shells.",
"reptiles are ectothermic, meaning they rely on external sources of heat to regulate their body temperature.",
"reptiles reproduce in a variety of ways, including laying eggs or giving birth to live young.",
"overall, reptiles are a fascinating and diverse group of animals with unique adaptations for survival."],

"rhinoceros":[
"a rhinoceros is a large, herbivorous mammal with two horns on its snout.",
"rhinoceroses have thick, gray skin that is covered in dull, reddish-brown hair.",
"the skin of a rhinoceros is tough and protective, with a thick layer of keratin.",
"rhinoceroses have a large, prehensile upper lip that is used for grasping food.",
"rhinoceroses have small, rounded ears and a short, thick neck.",
"rhinoceroses have a large, heavy body with a short, stout tail.",
"overall, rhinoceroses have a distinctive appearance with their two horns and thick skin."],

"rodents":[
"rodents are a diverse group of mammals with continuously growing incisors.",
"rodents include mice, rats, squirrels, and beavers.",
"rodents have a wide range of body sizes and shapes, from tiny mice to large beavers.",
"rodents have a variety of adaptations for survival, including sharp teeth, keen senses, and burrowing abilities.",
"rodents are found in a wide range of habitats, from forests and grasslands to deserts and urban areas.",
"rodents reproduce rapidly, with some species having up to 10 litters per year.",
"overall, rodents are a diverse and adaptable group of animals with unique adaptations for survival."],

"secretaryBird":[
"the secretary bird is a large, flightless bird with long legs and a distinctive crest of feathers on its head.",
"the secretary bird has a black body with white stripes on its wings and tail.",
"the secretary bird has a long, sharp beak that is used for hunting and tearing apart prey.",
"the secretary bird has large, powerful legs that are used for running and kicking.",
"the secretary bird has a distinctive call that sounds like a series of grunts and whistles.",
"the secretary bird is found in the grasslands and savannas of Africa.",
"overall, the secretary bird has a striking appearance with its long legs, crest of feathers, and black and white plumage."],

"serval":[
"the serval is a medium-sized wild cat with a slender body and long legs.",
"the serval has a tawny coat with black spots and a distinctive black stripe running from the nose to the back of the head.",
"the serval has large, tufted ears that are used for detecting prey.",
"the serval has a long, slender tail that is used for balance and communication.",
"the serval has sharp, retractable claws that are used for hunting and climbing.",
"the serval is found in the grasslands and savannas of Africa.",
"overall, the serval has a graceful appearance with its slender body, long legs, and distinctive coat pattern."],

"topi":[
"the topi is a medium-sized antelope with a slender build and long legs.",
"the topi has a reddish-brown coat with white underparts and a distinctive black rump.",
"the topi has a long, slender tail with a tuft of hair at the tip.",
"the topi has large, pointed ears that are used for detecting predators.",
"the topi has a long, slender muzzle and small, round eyes.",
"the topi is found in the grasslands and savannas of Africa.",
"overall, the topi has a graceful appearance with its slender build, long legs, and distinctive coat pattern."],

"vervetMonkey":[
"the vervet monkey is a small, agile primate with a long, slender body and a prehensile tail.",
"the vervet monkey has a reddish-brown coat with a white belly and a distinctive black face.",
"the vervet monkey has large, round ears that are used for detecting predators.",
"the vervet monkey has a long, thin muzzle and small, round eyes.",
"the vervet monkey has a long, slender tail that is used for balance and communication.",
"the vervet monkey is found in the forests and savannas of Africa.",
"overall, the vervet monkey has a nimble appearance with its long, slender body, prehensile tail, and distinctive coat pattern."],

"warthog":[
"the warthog is a large, stout pig with a distinctive hump on its shoulders and a long, curved tusk on each side of its mouth.",
"the warthog has a reddish-brown coat with a white belly and a distinctive black face.",
"the warthog has small, rounded ears and a short, thick neck.",
"the warthog has a long, muscular body with short legs and a short, stout tail.",
"the warthog has a distinctive snorting sound that it uses to communicate with other warthogs.",
"the warthog is found in the grasslands and savannas of Africa.",
"overall, the warthog has a distinctive appearance with its hump, tusks, and reddish-brown coat."],

"waterbuck":[
"the waterbuck is a large, antelope-like animal with a distinctive hump on its shoulders and a long, spiral horn on each side of its head.",
"the waterbuck has a reddish-brown coat with white underparts and a distinctive black rump.",
"the waterbuck has large, rounded ears and a short, thick neck.",
"the waterbuck has a long, muscular body with short legs and a short, stout tail.",
"the waterbuck has a distinctive snorting sound that it uses to communicate with other waterbucks.",
"the waterbuck is found in the grasslands and savannas of Africa.",
"overall, the waterbuck has a distinctive appearance with its hump, spiral horns, and reddish-brown coat."],

"wildcat":[
"the wildcat is a small, agile cat with a slender body and long legs.",
"the wildcat has a tawny coat with black spots and a distinctive black stripe running from the nose to the back of the head.",
"the wildcat has large, tufted ears that are used for detecting prey.",
"the wildcat has a long, slender tail that is used for balance and communication.",
"the wildcat has sharp, retractable claws that are used for hunting and climbing.",
"the wildcat is found in a wide range of habitats, from forests and grasslands to deserts and urban areas.",
"overall, the wildcat has a sleek appearance with its slender body, long legs, and distinctive coat pattern."],

"wildebeest":[
"the wildebeest is a large, hoofed mammal with a distinctive curved horn on each side of its head.",
"the wildebeest has a reddish-brown coat with white underparts and a distinctive black rump.",
"the wildebeest has large, rounded ears and a short, thick neck.",
"the wildebeest has a long, muscular body with short legs and a short, stout tail.",
"the wildebeest has a distinctive bellowing sound that it uses to communicate with other wildebeests.",
"the wildebeest is found in the grasslands and savannas of Africa.",
"overall, the wildebeest has a distinctive appearance with its curved horns, reddish-brown coat, and white underparts."],

"zebra":[
"the zebra is a large, hoofed mammal with black and white stripes.",
"the zebra has a reddish-brown coat with white underparts and a distinctive black rump.",
"the zebra has large, rounded ears and a short, thick neck.",
"the zebra has a long, muscular body with short legs and a short, stout tail.",
"the zebra has a distinctive braying sound that it uses to communicate with other zebras.",
"the zebra is found in the grasslands and savannas of Africa.",
"overall, the zebra has a striking appearance with its black and white stripes, reddish-brown coat, and white underparts."],

"zorilla":[
"the zorilla is a small, burrowing mammal with a long, slender body and a short, stout tail.",
"the zorilla has a reddish-brown coat with white underparts and a distinctive black rump.",
"the zorilla has large, rounded ears and a short, thick neck.",
"the zorilla has a long, muscular body with short legs and a short, stout tail.",
"the zorilla has a distinctive barking sound that it uses to communicate with other zorillas.",
"the zorilla is found in the grasslands and savannas of Africa.",
"overall, the zorilla has a distinctive appearance with its long, slender body, short legs, and reddish-brown coat."],

}

camera_trap_templates_terra_Phi = {
"badger":[
" a badger is a mammal with a stout body and short sturdy legs.",
" a badger’s fur is coarse and typically grayish-black.",
" badgers often feature a white stripe running from the nose to the back of the head dividing into two stripes along the sides of the body to the base of the tail.",
" badgers have broad flat heads with small eyes and ears.",
" badger noses are elongated and tapered ending in a black muzzle.",
" badgers possess strong well-developed claws adapted for digging burrows.",
" overall badgers have a rugged and muscular appearance suited for their burrowing lifestyle."],

"bird":[
" a bird is a warm-blooded vertebrate characterized by feathers, beak, and the ability to fly (although some species are flightless).",
" birds have a lightweight skeleton with hollow bones, reducing their overall body weight.",
" feathers cover the bird’s body, providing insulation, protection, and, in many cases, aiding in flight.",
" the beak is a hard, keratinous structure that varies in shape and size depending on the species’ dietary needs.",
" birds have a unique respiratory system that includes air sacs, allowing for efficient oxygen exchange during flight."],

"bobcat":[
" a bobcat is a medium-sized wild cat native to North America, with a distinctive short, “bobbed” tail.",
" bobcats have a robust, muscular body, with a short, thick tail and a broad head.",
" their fur is short and usually sandy or reddish-brown, with distinctive dark stripes on the legs and tail.",
" bobcats possess keen senses, including excellent night vision and a highly developed sense of hearing.",
" they are solitary and largely nocturnal, preferring to hunt small mammals and birds."],

"car":[
" a car is a wheeled motor vehicle that is used for transportation.",
" cars have a metal frame, four wheels, and an enclosed cabin for passengers.",
" they are powered by an internal combustion engine or, more recently, electric motors.",
" cars are designed with a variety of safety features, including seat belts and airbags.",
" modern cars often feature advanced technology, such as GPS navigation and in-car entertainment systems."],

"cat":[
" a cat is a small carnivorous mammal with soft fur, retractable claws, and a keen sense of hearing and sight.",
" cats have a flexible body, quick reflexes, and sharp teeth designed for hunting.",
" domestic cats come in various colors and patterns, with long, slender bodies and a distinctive 'M' shape on their forehead.",
" wild cats, such as lions and tigers, have larger bodies, shorter legs, and a more muscular build.",
" cats are known for their agility and ability to climb and jump with ease."],

"coyote":[
" a coyote is a canid native to North America, known for its sharp, pointed ears and bushy tail.",
" coyotes have a lean, athletic build with long legs for endurance running.",
" their fur is typically grayish-brown, with a white underbelly and a bushy, black-tipped tail.",
" coyotes are highly adaptable, able to thrive in various habitats, from forests and grasslands to urban areas.",
" they are primarily nocturnal, hunting small mammals and birds."],

"deer":[
" a deer is a hoofed mammal belonging to the family Cervidae, characterized by its slender body and antlers (in males).",
" deer vary in size and color, with species such as the white-tailed deer, red deer, and caribou.",
" deer are herbivores, feeding on a diet of leaves, grass, and fruits.",
" males, or bucks, grow antlers annually, which are shed and regrown each year.",
" deer are known for their agility, able to run quickly and leap over obstacles."],

"dog":[
" a dog is a domesticated carnivorous mammal belonging to the Canidae family, known for its loyalty and diverse breeds.",
" dogs have a wide range of sizes, colors, and shapes, with various breeds serving different purposes, such as herding, hunting, and companionship.",
" dogs have a keen sense of smell and hearing, making them excellent trackers and protectors.",
" they communicate through vocalizations, body language, and facial expressions.",
" dogs are social animals, forming strong bonds with their owners and other dogs."],

"empty":[
" empty is the lack of content or occupancy.",
" it can refer to a container, space, or concept that is unfilled or devoid of substance.",
" empty can also symbolize solitude, freedom, or potential, depending on the context.",
" in a literal sense, an empty room lacks furniture and possessions.",
" metaphorically, an empty vessel is open to new ideas and experiences."],

"fox":[
" a fox is a small, opportunistic carnivore native to Europe, Asia, and North America, known for its bushy tail and cunning nature.",
" foxes have a slender body, long legs, and a pointed snout, with a coat varying in color from red to gray to brown.",
" foxes are highly adaptable, able to thrive in various habitats, including forests, grasslands, and urban areas.",
" they are primarily nocturnal, hunting rodents, birds, and fruits.",
" foxes are known for their intelligence and cunning, often using elaborate strategies to catch their prey."],

"opossum":[
" an opossum is a marsupial native to North and South America, known for its long, hairless tail and prehensile snout.",
" opossums have a small, pointed snout with a long, sensitive prehensile snout, used for grooming and finding food.",
" their fur is typically grayish-brown, with a lighter belly and a white face.",
" opossums are omnivorous, feeding on fruits, insects, small mammals, and carrion.",
" they are marsupials, giving birth to underdeveloped young that crawl into a pouch on their mother’s belly for further development."],

"rabbit":[
" a rabbit is a small mammal belonging to the family Leporidae, characterized by its long ears, hind legs, and fluffy tail.",
" rabbits have a soft, dense fur that varies in color from white to brown to gray, with some species having distinctive markings.",
" their fur provides insulation and camouflage in their natural habitats, which range from forests and grasslands to deserts.",
" rabbits are herbivores, feeding on a diet of grass, leaves, and other plant material.",
" they are known for their reproductive capacity, with females breeding multiple times per year and producing several litters."],

"raccoon":[
" a raccoon is a medium-sized mammal native to North America, known for its distinctive facial mask and ringed tail.",
" raccoons have a stocky body, with a bushy, ringed tail, and a mask of dark fur around their eyes.",
" their fur is typically grayish-brown, with a lighter underside and a dense, coarse coat.",
" raccoons are omnivorous, feeding on a variety of foods, including fruits, nuts, insects, and small mammals.",
" they are highly adaptable, able to thrive in diverse environments, from forests and wetlands to urban areas."],

"rodent":[
" a rodent is a mammal belonging to the order Rodentia, characterized by its continuously growing incisors in the upper and lower jaws.",
" rodents include a diverse group of species, such as mice, rats, squirrels, and beavers.",
" their body size varies, with some species as small as 2 inches long, while others, like beavers, can grow to over 3 feet long.",
" rodents have a wide range of diets, feeding on seeds, fruits, nuts, insects, and even carrion in some species.",
" they are known for their ability to reproduce rapidly, with some species capable of producing multiple litters per year."],

"skunk":[
" a skunk is a small mammal native to North and South America, known for its distinctive black and white coloration and its ability to spray a foul-smelling liquid as a defense mechanism.",
" skunks have a stocky body, with a short, furry tail and a striped pattern of black and white.",
" their fur is typically thick and dense, with a soft undercoat and a coarse outer coat.",
" skunks are omnivores, feeding on a diet of insects, small mammals, fruits, and vegetables.",
" they are solitary animals, living in underground burrows or dens."],

"squirrel":[
" a squirrel is a small to medium-sized rodent belonging to the family Sciuridae, characterized by its long, bushy tail and sharp claws for climbing.",
" squirrels have a small, slender body, with a long, fluffy tail, and a bushy, round head.",
" their fur varies in color, with some species having a reddish-brown coat, while others are gray, black, or white.",
" squirrels are primarily herbivores, feeding on nuts, seeds, fruits, and flowers.",
" they are highly adapted for climbing, with strong hind limbs and sharp claws that allow them to scale trees and jump between branches with ease."],

}

camera_trap_templates_serengeti_Qwen ={
'aardvark':[
" an aardvark is a nocturnal mammal with a long, snouted face and a tubular body.",
" aardvarks have sparse, coarse hair that ranges in color from gray to brown.",
" they possess large, rabbit-like ears and a long, prehensile tongue used for catching ants and termites.",
" aardvarks have powerful claws on their front feet, which they use for digging burrows and breaking into insect nests.",
" their skin is tough and thick, providing protection from bites.",
" aardvarks have a unique zigzag pattern of muscles in their tongue, which helps them to flick it rapidly in and out of ant and termite nests.",
" overall, aardvarks have a somewhat awkward appearance but are highly adapted for their underground lifestyle."],

'ardwolf':[
" an aardwolf is a small, nocturnal mammal with a slender body and long legs.",
" aardwolves have short, dense fur that is typically a yellowish-brown or tawny color with black vertical stripes.",
" they have a bushy tail that is often held upright when standing.",
" aardwolves have a long snout and sharp, pointed teeth, which they use to crush insects.",
" their ears are large and rounded, helping them to hear prey moving underground.",
" aardwolves feed primarily on harvester termites, making them important predators in their ecosystem.",
" they are known for their distinctive striped pattern and are often mistaken for aardvarks due to their similar habitat and diet."],

'baboon':[
" a baboon is a robust primate with a stocky build and a prominent face.",
" baboons have a varied coat color depending on the species, ranging from olive-gray to brown or even red.",
" they possess a distinctive patch of bare skin on their rump, which can be brightly colored in some species.",
" baboons have a pronounced sagittal crest on their skull, where muscles attach to aid in chewing.",
" their faces are marked with a series of ridges and folds, giving them a somewhat fierce appearance.",
" baboons have opposable thumbs, which assist in grasping objects and climbing trees.",
" they are highly social animals, living in large groups called troops, and are known for their complex social structures."],

'batEaredFox':[
" a bat-eared fox is a small canid with exceptionally large, bat-like ears.",
" its fur is generally sandy brown or tan with black markings around the eyes and on the legs.",
" bat-eared foxes have a short, bushy tail that is usually tipped with black.",
" their ears are not only large but also highly mobile, allowing them to pinpoint the location of prey.",
" bat-eared foxes have a slender build, with long legs and a lean body, which makes them fast runners.",
" they primarily feed on insects, small mammals, and birds, making them important predators in their environment.",
" their large ears also help regulate their body temperature, especially in hot climates."],

'buffalo':[
" a buffalo, specifically the African buffalo, is a large bovine with a heavy, muscular build.",
" their coat is typically dark gray or black, although it can vary in color based on age and sex.",
" buffaloes have a distinctive hump above the shoulders and a large, muscular head.",
" they have large horns that curve upwards and outwards, forming a crescent shape.",
" buffaloes have a tough hide that is covered with sparse hair, providing protection from predators.",
" they live in large herds and are known for their aggressive behavior when threatened.",
" their powerful build and curved horns make them formidable animals in their natural habitat."],

'bushbuck':[
" a bushbuck is a medium-sized antelope with a reddish-brown coat that is often mottled with white spots.",
" they have a slender body with long legs and a short tail.",
" bushbucks have a distinctive white ring around their eyes and a white chevron between their horns.",
" their horns are spiral and occur only in males, while females are hornless.",
" bushbucks are shy and elusive, spending most of their time hidden in dense vegetation.",
" they have a strong sense of smell and are primarily browsers, feeding on leaves, shoots, and fruits.",
" their coloring provides excellent camouflage in their forest habitats."],

'caracal':[
" a caracal is a medium-sized wild cat with a slender body and long legs.",
" their fur is generally tawny or reddish-brown with black spots and streaks.",
" caracals have a distinctive black tuft at the tip of their ears, which gives them a unique appearance.",
" their eyes are large and round, typically amber or golden in color.",
" caracals have a short tail that is usually black-tipped.",
" they are excellent climbers and are known for their ability to take down prey much larger than themselves.",
" caracals are adaptable and can be found in a variety of habitats, including deserts, savannas, and forests."],

'cheetah':[
" a cheetah is a slender, fast-running feline with a spotted coat.",
" their fur is typically tawny or yellowish with black spots arranged in a unique pattern.",
" cheetahs have a small head with high-set eyes and a long, slender tail.",
" they have distinctive black 'tear marks' running from the corners of their eyes down the sides of their nose, which may help reduce glare while hunting.",
" cheetahs have non-retractable claws that provide traction during sprints.",
" they are the fastest land animals, capable of reaching speeds over 100 km/h.",
" their spotted coat provides excellent camouflage in grasslands and open savannas."],

'civet':[
" a civet is a small, carnivorous mammal with a slender body and a long, bushy tail.",
" their fur is typically gray or brown with black spots or stripes.",
" civets have a distinctive mask-like marking around their eyes, which can be black or dark brown.",
" they have a long snout and small, round ears.",
" civets are nocturnal and have excellent night vision and a keen sense of smell.",
" they are omnivorous, feeding on a variety of insects, small mammals, and fruits.",
" civets are known for their musky scent glands, which they use for marking territory and attracting mates."],

'dikDik':[
" a dik-dik is a small antelope with a distinctive prehensile lip and a compact body.",
" their fur is typically reddish-brown with white underparts and black markings around the eyes and mouth.",
" dik-diks have long, thin legs and a short tail.",
" they have small, rounded ears and a small head with a prominent nose.",
" dik-diks are known for their high-pitched, barking calls, which they use for communication and warning of danger.",
" they are shy and elusive, living in dense vegetation and avoiding open areas.",
" their small size and agility allow them to move quickly through underbrush, evading predators."],

'eland':[
" an eland is a large, spiral-horned antelope with a light tan or reddish-brown coat.",
" they have a distinctive dewlap, a fold of loose skin hanging from the throat.",
" elands have large, rounded ears and a short tail.",
" their horns are long and spiral, occurring in both males and females.",
" elands are the largest antelope species and are known for their gentle nature.",
" they are browsers, feeding on leaves, flowers, and fruits.",
" their size and spiral horns make them easily recognizable in their savanna and woodland habitats."],

'elephant':[
" an elephant is a large, gray mammal with a long trunk and large ears.",
" their skin is thick and gray, with wrinkles and folds that give them a distinctive appearance.",
" elephants have a long, flexible trunk that serves as a nose and arm.",
" they have large, fan-shaped ears that are covered in blood vessels and help regulate body temperature.",
" elephants have four tusks, two in the upper jaw and two in the lower jaw, although tusks in females are often smaller or absent.",
" they are highly social animals, living in family groups led by a matriarch.",
" their intelligence and memory are renowned, and they play a crucial role in their ecosystem through seed dispersal and habitat modification."],

'gazelleGrants':[
" a Grant's gazelle is a medium-sized antelope with a slender body and long legs.",
" their coat is typically tawny or reddish-brown with white underparts and a distinctive black stripe on the rump.",
" Grant's gazelles have a long neck and a small head with large, forward-facing eyes.",
" they have a short tail that is usually black-tipped.",
" Grant's gazelles are known for their graceful leaps and bounds, which they use to evade predators.",
" they are primarily grazers, feeding on grasses and other low-lying vegetation.",
" their coloring and agility make them well-suited to their open grassland habitats."],

'gazelleThomsons':[
" a Thomson's gazelle is a medium-sized antelope with a slender build and long legs.",
" their coat is typically reddish-brown with white underparts and a distinctive black stripe on the rump.",
" Thomson's gazelles have a long neck and a small head with large, forward-facing eyes.",
" they have a short tail that is usually black-tipped.",
" Thomson's gazelles are known for their high-speed dashes and sharp turns, which they use to escape predators.",
" they are primarily grazers, feeding on grasses and other low-lying vegetation.",
" their coloring and agility make them well-adapted to their open grassland habitats."],

'genet':[
" a genet is a small, carnivorous mammal with a slender body and a bushy tail.",
" their fur is typically gray or brown with black spots or stripes.",
" genets have a distinctive mask-like marking around their eyes, which can be black or dark brown.",
" they have a long snout and small, rounded ears.",
" genets are nocturnal and have excellent night vision and a keen sense of smell.",
" they are omnivorous, feeding on a variety of insects, small mammals, and fruits.",
" genets are known for their agility and are skilled climbers, using their long tails for balance."],

'giraffe':[
" a giraffe is a tall mammal with a long neck and legs, standing up to 18 feet tall.",
" a giraffe's coat is covered in distinctive irregular patches of brown or orange separated by lighter lines, creating a unique pattern.",
" giraffes have small heads with large eyes and short horns called ossicones covered in skin and hair.",
" their tongues can be up to 20 inches long, prehensile and covered in small, claw-like projections called papillae, aiding in grasping leaves.",
" giraffes have a spotted pattern that varies among individuals, serving as camouflage in the savanna.",
" their legs are long and slender, with hooves that are wide and splayed to support their weight.",
" overall, giraffes have a graceful yet powerful build, designed for reaching high foliage."],

'guineaFowl':[
" a guinea fowl is a bird with a compact body and a fan-shaped tail.",
" its plumage is predominantly gray with white spots, giving it a speckled appearance.",
" guinea fowls have a distinctive crest on their heads, which is usually black and horn-colored.",
" they have a white face with a red wattle hanging down from the throat.",
" their eyes are surrounded by a ring of bright blue skin.",
" guinea fowls have strong, yellow legs and feet equipped with sharp claws.",
" these birds are known for their loud, resonant call and their ability to run swiftly on the ground."],

'hare':[
" a hare is a medium-sized mammal with a long, powerful hind leg and a short, bushy tail.",
" its fur is soft and dense, typically ranging in color from gray to brown, with white underparts.",
" hares have large, round ears that are usually tipped with black.",
" their eyes are large and positioned on the sides of the head, providing a wide field of vision.",
" hares have a lean and muscular body, adapted for quick bursts of speed.",
" they have long, agile front legs and large, robust hind legs with long, straight claws.",
" overall, hares have a sleek, athletic appearance suited for their life as prey animals."],

'hartebeest':[
" a hartebeest is a large antelope with a sturdy build and a distinctively curved horn.",
" its coat is reddish-brown to tan, with a lighter underbelly and a dark patch around the eyes and mouth.",
" hartebeests have a pair of long, S-shaped horns that curve backward and inward.",
" their heads are broad with a concave profile, and they have a noticeable ridge above the eyes.",
" hartebeests have a stocky body with a deep chest and a muscular neck.",
" their legs are long and slender, with hooves that are spread to provide stability.",
" overall, hartebeests have a robust and imposing presence, well-suited for their grazing habits."],

'hippopotamus':[
" a hippopotamus is a large, semi-aquatic mammal with a barrel-shaped body and short legs.",
" its skin is thick and gray, often appearing pink due to blood vessels near the surface.",
" hippos have a wide mouth with large, tusk-like canine teeth that can reach up to 2 feet in length.",
" their eyes and ears are located high on their heads, allowing them to see and hear while mostly submerged.",
" hippos have a short, stubby tail and a thick layer of fat beneath their skin.",
" their legs are short and wide, adapted for walking on the riverbed.",
" overall, hippos have a massive and powerful build, reflecting their dominance in aquatic environments."],

'honeyBadger':[
" a honey badger is a small, stocky mammal with a thick, loose-fitting black and white striped coat.",
" its fur is dense and wiry, providing protection against bites and scratches.",
" honey badgers have a distinctive white stripe running from the top of the head to the tip of the tail.",
" their heads are small and rounded, with a short, pointed snout.",
" honey badgers have small, round ears and small, dark eyes.",
" they possess strong jaws and sharp, non-retractable claws used for digging and fighting.",
" overall, honey badgers have a tough and resilient appearance, well-suited for their aggressive nature."],

'hyenaSpotted':[
" a spotted hyena is a carnivorous mammal with a robust build and a distinctive laugh-like vocalization.",
" its coat is a mixture of brown and black spots on a tan background, with a pale belly.",
" spotted hyenas have a broad head with a powerful jaw and large, sharp teeth.",
" their ears are round and upright, and their eyes are small but keen.",
" they have a long, bushy tail that is often held erect when running.",
" spotted hyenas have strong legs and a flexible spine, enabling them to run at high speeds.",
" overall, spotted hyenas have a powerful and agile appearance, reflecting their role as apex predators."],

'hyenaStriped':[
" a striped hyena is a carnivorous mammal with a slender build and a pale, striped coat.",
" its coat has dark brown or black vertical stripes, similar to a tiger, but less distinct.",
" striped hyenas have a broad head with a pointed snout and powerful jaws.",
" their eyes are small and set far apart, with a keen sense of smell.",
" they have a long, bushy tail that is often carried horizontally.",
" striped hyenas have long, slender legs and a flexible spine, adapted for both running and climbing.",
" overall, striped hyenas have a lean and agile appearance, reflecting their scavenging and hunting habits."],

'impala':[
" an impala is a medium-sized antelope with a slender build and striking spiral horns in males.",
" its coat is a reddish-brown with a white underbelly, and a black stripe runs along the lower part of its body.",
" male impalas have long, spiral horns that curve backward and outward, up to 3 feet in length.",
" their heads are relatively small with a narrow snout and large ears.",
" impalas have a long, slender neck and a deep chest.",
" their legs are long and muscular, with hooves that are spread to provide stability.",
" overall, impalas have a graceful and athletic appearance, well-suited for their life as browsers."],

'jackal':[
" a jackal is a small, slender carnivorous mammal with a bushy tail and a distinctive howl.",
" its coat is typically sandy or gray, with a darker stripe along the back and tail.",
" jackals have a pointed snout and large, pointed ears that can swivel independently.",
" their eyes are large and expressive, with a keen sense of sight and hearing.",
" jackals have long, slender legs and a lithe body, adapted for speed and agility.",
" their paws are small and padded, with non-retractable claws.",
" overall, jackals have a lean and agile appearance, reflecting their role as opportunistic hunters and scavengers."],

'koriBustard':[
" a kori bustard is a large, ground-dwelling bird with a distinctive appearance and flight capabilities.",
" its plumage is predominantly light brown with a white underbelly and a black stripe across the breast.",
" kori bustards have a large, broad bill with a hooked tip and a yellow cere.",
" their eyes are large and set far apart, with a keen sense of vision.",
" kori bustards have long, powerful legs and a short, broad tail.",
" their wings are long and pointed, capable of supporting their heavy bodies in flight.",
" overall, kori bustards have a majestic and powerful appearance, well-suited for their habitat."],

'leopard':[
" a leopard is a large, carnivorous cat with a spotted coat and a powerful build.",
" its coat is yellowish-brown with black rosettes, each containing black spots.",
" leopards have a broad head with a short, broad snout and large, piercing eyes.",
" their ears are small and rounded, with tufts of black hair at the tips.",
" leopards have a long, muscular body with a thick, spotted tail.",
" their legs are long and powerful, with large, retractable claws.",
" overall, leopards have a sleek and agile appearance, reflecting their stealth and strength."],

'lionFemale':[
" a female lion is a large, powerful carnivorous cat with a tawny coat and a slender build.",
" her coat is a uniform tawny color with faint black spots, less prominent than those of a leopard.",
" female lions have a broad head with a short, broad snout and large, amber eyes.",
" their ears are small and rounded, with tufts of black hair at the tips.",
" female lions have a long, muscular body with a thick, spotted tail.",
" their legs are long and powerful, with large, retractable claws.",
" overall, female lions have a majestic and powerful appearance, reflecting their role as dominant members of the pride."],

'lionMale':[
" a male lion is a large, powerful carnivorous cat with a tawny coat and a distinctive mane.",
" his coat is a uniform tawny color with faint black spots, less prominent than those of a leopard.",
" male lions have a broad head with a short, broad snout and large, amber eyes.",
" their ears are small and rounded, with tufts of black hair at the tips.",
" male lions have a long, muscular body with a thick, spotted tail.",
" their legs are long and powerful, with large, retractable claws.",
" male lions are distinguished by their thick mane, which ranges in color from light gold to dark brown and can extend to the shoulders and back.",
" overall, male lions have an imposing and majestic appearance, symbolizing power and leadership."],

'mongoose':[
" a mongoose is a small, agile mammal with a slender body and a bushy tail.",
" its coat is typically brown or gray with dark markings, though colors vary among species.",
" mongooses have a small, elongated head with a pointed snout and large, round eyes.",
" their ears are small and rounded, and they have whiskers on their faces and legs.",
" mongooses have long, slender legs and a lithe body, adapted for speed and agility.",
" their paws are small and padded, with non-retractable claws.",
" overall, mongooses have a lean and agile appearance, reflecting their role as opportunistic hunters and scavengers."],

'ostrich':[
" an ostrich is a large, flightless bird with a long neck and legs.",
" its plumage is primarily white with black and gray feathers on the wings and tail.",
" ostriches have a long, thin neck with a small head and large, round eyes.",
" their bills are long and slender, ending in a hooked tip.",
" ostriches have long, powerful legs with two toes, each equipped with a large, clawed foot.",
" their bodies are long and streamlined, with a small, featherless head.",
" ostriches have a long, broad tail with long, wispy feathers.",
" overall, ostriches have a striking and imposing appearance, well-suited for their life on the savanna."],

'porcupine':[
" a porcupine is a large rodent characterized by its thick, spiny coat of quills.",
" a porcupine’s quills are typically brown or black, with white tips, and are barbed at the tip.",
" porcupines have a rounded body shape with short legs and a bushy tail.",
" their faces are small and round, with small eyes and a blunt snout.",
" porcupines have large, flat teeth adapted for gnawing on bark and wood.",
" they are mostly nocturnal and use their sharp claws for climbing trees and digging dens.",
" porcupines have a defensive mechanism where they can release their quills when threatened."],

'reedbuck':[
" a reedbuck is a medium-sized antelope with slender legs and a long neck.",
" reedbucks have a reddish-brown coat with white underbellies and a white rump patch.",
" they have a distinctive white ring around their eyes and a dark stripe along their back.",
" reedbucks have small, pointed ears and a small, straight horn (only in males).",
" their hooves are broad and slightly webbed, ideal for walking on soft ground.",
" reedbucks are generally shy and prefer to hide in dense vegetation.",
" they are excellent swimmers and can escape predators by entering water."],

'reptiles':[
" reptiles are cold-blooded vertebrates with scaly skin, usually laying eggs on land.",
" reptiles vary widely in size and shape, from small lizards to large crocodiles.",
" their skin is covered in scales which can be smooth or bumpy, and is often colored for camouflage.",
" most reptiles have four limbs, though some species like snakes have evolved to lose them.",
" reptiles have a three-chambered heart (except for crocodilians, which have a four-chambered heart).",
" they are found in diverse habitats including deserts, forests, and aquatic environments.",
" reptiles breathe through lungs and have a backbone made of vertebrae."],

'rhinoceros':[
" a rhinoceros is a large herbivorous mammal with thick, armored skin.",
" rhinos have a distinctive horn on their nose, which varies in number and shape between species.",
" their skin is tough and gray, covered in plates called epidermal scales.",
" rhinos have a small brain relative to their body size and a short tail.",
" they have a wide mouth with lips adapted for grazing and browsing.",
" rhinos have poor eyesight but excellent hearing and smell.",
" they are known for their solitary nature and territorial behavior."],

'rodents':[
" rodents are small mammals characterized by their large, chisel-like incisors that grow continuously.",
" their bodies are generally small and compact, with fur that can vary in color from brown to gray.",
" rodents have long, sensitive whiskers and small, rounded ears.",
" they have a variety of body shapes depending on the species, from the sleek body of a mouse to the stocky build of a beaver.",
" rodents are highly adaptable and can be found in almost every habitat on Earth.",
" they are primarily nocturnal and feed on a variety of foods, including seeds, nuts, and plants.",
" rodents are known for their rapid reproduction rates and ability to burrow."],

'secretaryBird':[
" a secretary bird is a tall, terrestrial bird of prey with a distinctive crest of feathers on its head.",
" secretary birds have long legs and toes, adapted for running and hunting on the ground.",
" their plumage is primarily gray with black and white markings, giving them a striking appearance.",
" they have a long, snake-like neck and a hooked bill, both adaptations for capturing prey.",
" secretary birds have keen eyesight and can spot prey from a distance.",
" they hunt by stomping on insects and small animals with their powerful feet before swallowing them whole.",
" secretary birds are known for their unique method of flight, with their long legs trailing behind them."],

'serval':[
" a serval is a medium-sized wild cat with a slender body and long legs.",
" servals have a tawny coat covered in black spots and stripes, providing excellent camouflage.",
" they have a small head with large, rounded ears and large, expressive eyes.",
" servals have a long neck and a bushy tail with a black tip.",
" their paws are large and have non-retractable claws, aiding in their ability to climb trees.",
" servals are excellent climbers and jumpers, able to leap up to 12 feet vertically.",
" they are nocturnal hunters, feeding on small mammals, birds, and insects."],

'topi':[
" a topi is an antelope with a distinctive white face and a dark brown body.",
" topis have a high, lyre-shaped horns that curve backward and outward.",
" their coat is short and glossy, with a black stripe running down the spine.",
" topis have a slender body with long legs, allowing them to run swiftly.",
" they have a small head with large ears and a pointed snout.",
" topis are highly social and live in large herds.",
" they are found in grasslands and open savannas, where they graze on grasses."],

'vervetMonkey':[
" a vervet monkey is a small, arboreal primate with a greenish-gray back and yellow underbelly.",
" vervet monkeys have a distinctively blue face framed by white hair, with black hands and feet.",
" they have a small head with large, expressive eyes and a pink nose.",
" vervet monkeys have a long, prehensile tail that they use for balance while climbing.",
" their hands and feet are dexterous, with opposable thumbs and fingers.",
" vervet monkeys are highly social and live in complex multi-male, multi-female groups.",
" they are omnivorous, eating fruits, leaves, insects, and small animals."],

'warthog':[
" a warthog is a wild pig with a distinctive appearance, featuring two pairs of large, fleshy warts on its face.",
" warthogs have a brownish-gray coat with sparse hair, making them appear almost bald.",
" they have a long, muscular body with short legs and a broad head.",
" warthogs have a pair of large, curved tusks that protrude from their lower jaw.",
" their eyes are small and set far back on their heads, providing good peripheral vision.",
" warthogs are known for their unique behavior of wallowing in mud to protect their skin from sunburn and insect bites.",
" they are primarily grazers and live in open grasslands and savannas."],

'waterbuck':[
" a waterbuck is a large antelope with a robust body and a distinctive white ring around its rump.",
" waterbucks have a reddish-brown coat that becomes more gray with age.",
" they have a long, narrow face with a white blaze on their forehead.",
" waterbucks have spiral horns that curve backwards and are present only in males.",
" their ears are large and mobile, helping them detect predators.",
" waterbucks are excellent swimmers and often rest in water to cool off and avoid biting flies.",
" they are found in wetland areas and near water sources, where they graze on grasses."],

'wildcat':[
" a wildcat is a small, wild felid with a tawny coat covered in black spots and stripes.",
" wildcats have a slender body with long legs and a bushy tail.",
" they have a small head with large, round eyes and pointed ears.",
" wildcats have retractable claws and sharp, curved teeth adapted for hunting.",
" their coat is dense and provides excellent camouflage in their forest habitats.",
" wildcats are solitary and primarily nocturnal, feeding on small mammals and birds.",
" they are skilled climbers and can take refuge in trees to avoid predators."],

'wildebeest':[
" a wildebeest is a large, bovine antelope with a distinctive face and horns.",
" wildebeests have a grayish-brown coat with a darker mane and beard.",
" they have a square, flat face with a pair of horns that curve forward and then downward.",
" wildebeests have a long, muscular neck and a broad, deep chest.",
" their eyes are set far apart, providing good binocular vision.",
" wildebeests are famous for their annual migration across the Serengeti, seeking fresh grazing grounds.",
" they are herd animals and rely on each other for protection against predators."],

'zebra':[
" a zebra is a large, striped equine with a black and white coat.",
" zebras have a stocky build with long legs and a long neck.",
" they have a distinctive pattern of black and white stripes that varies between individuals.",
" zebras have a small head with large, round ears and a black muzzle.",
" their hooves are hard and adapted for running on rough terrain.",
" zebras are social animals and live in herds, using their stripes for camouflage and recognition.",
" they are herbivores, primarily feeding on grasses and leaves."],

'zorilla':[
" a zorilla is a small, carnivorous mammal closely related to the skunk, known for its potent spray.",
" zorillas have a black and white striped coat that resembles a skunk, providing a warning signal to predators.",
" they have a small head with a long snout and small, rounded ears.",
" zorillas have short legs and a bushy tail.",
" their anal glands produce a strong-smelling secretion as a defense mechanism.",
" zorillas are nocturnal and feed on insects, small mammals, and birds.",
" they are found in grasslands and savannas, where they burrow for shelter."],
    }

camera_trap_templates_terra_Qwen = {
'badger':[
"a badger is a mammal with a stout body and short sturdy legs",
"a badger’s fur is coarse and typically grayish-black",
"badgers often feature a white stripe running from the nose to the back of the head dividing into two stripes along the sides of the body to the base of the tail",
"badgers have broad flat heads with small eyes and ears",
"badger noses are elongated and tapered ending in a black muzzle",
"badgers possess strong well-developed claws adapted for digging burrows",
"overall badgers have a rugged and muscular appearance suited for their burrowing lifestyle"],

'bird':[
"birds are warm-blooded vertebrates with feathers covering their bodies",
"they have a beak or bill instead of teeth, which varies in shape depending on diet",
"most birds have two wings and two legs, though some species have adaptations like flightless wings",
"their eyes are large and prominent, aiding in vision and often featuring colorful irises",
"birds have lightweight skeletons made primarily of hollow bones, which help them fly",
"they lay eggs, usually in nests, which can vary greatly in construction and location",
"birds exhibit a wide range of behaviors including migration, territoriality, and social interaction"],

'bobcat':[
"a bobcat is a medium-sized wild cat native to North America",
"it has a tawny coat with dark spots and streaks, providing camouflage in its woodland habitat",
"bobcats have a short, bobbed tail, which is how they got their name",
"their ears are tufted at the tips, giving them excellent hearing",
"bobcats have sharp, retractable claws and powerful jaws adapted for hunting",
"they are solitary animals and are known for their agility and stealth",
"bobcats can climb trees and swim, but they prefer to hunt on the ground"],

'car':[
"a car is a four-wheeled motor vehicle designed for transporting people and goods",
"cars come in various sizes and styles, including sedans, SUVs, and convertibles",
"they are powered by internal combustion engines or electric motors",
"cars have a closed cabin with windows and doors",
"they are equipped with a steering wheel, pedals for acceleration and braking, and a gear shift",
"cars are typically painted in a variety of colors and often have distinctive designs",
"cars are a common mode of transportation in urban and rural areas"],

'cat':[
"a cat is a small, typically furry, carnivorous mammal",
"cats have soft, dense fur that comes in a variety of colors and patterns",
"they have sharp, retractable claws and keen senses, particularly sight and hearing",
"cats have a flexible body and long, slender legs that allow them to jump great heights",
"their faces are characterized by whiskers, large eyes, and a small nose",
"cats are known for being independent and often playful",
"they communicate through vocalizations such as meowing and purring"],

'coyote':[
"a coyote is a medium-sized wild canid found throughout North America",
"its fur is typically grayish-brown, with lighter underparts and darker patches on the face and back",
"coyotes have a bushy tail that is often carried low when walking",
"they have large, pointed ears and yellow eyes",
"coyotes have strong jaws and sharp teeth, ideal for tearing meat",
"they are highly adaptable and can thrive in a variety of habitats, including forests, deserts, and urban areas",
"coyotes are known for their howling calls, which can be heard over long distances"],

'deer':[
"deer are herbivorous mammals known for their graceful movements and antlers in males",
"their coats are usually reddish-brown in summer and grayish-brown in winter, providing camouflage",
"deer have slender legs, allowing them to run quickly to escape predators",
"they have large, expressive eyes and a small nose with a prehensile upper lip",
"males grow antlers annually, which they shed and regrow each year",
"deer are generally shy and live in herds for protection",
"they feed on vegetation such as grasses, leaves, and twigs"],

'dog':[
"a dog is a domesticated carnivorous mammal known for its loyalty and companionship",
"dogs have a wide variety of coat types and colors, ranging from short and smooth to long and fluffy",
"they have sharp senses, including smell, hearing, and vision",
"dogs have a strong sense of pack hierarchy and communicate through body language and vocalizations",
"they are known for their ability to perform tasks such as guarding, herding, and assisting in search and rescue operations",
"dogs require regular exercise and socialization to maintain good health and behavior",
"they form strong bonds with humans and are often referred to as 'man's best friend'"],

'empty':[
"an empty space is devoid of physical objects or contents",
"it represents a lack of matter within a given area",
"empty spaces can vary in size, from microscopic to vast cosmic voids",
"they are crucial in physics for understanding concepts like vacuum and pressure",
"empty spaces allow for movement and are essential for the functioning of many systems",
"they can symbolize absence or emptiness in philosophical and artistic contexts",
"empty spaces are also important in design and architecture for functionality and aesthetics"],

'fox':[
"a fox is a small to medium-sized carnivorous mammal known for its cunning and agility",
"its fur is typically red or orange, though some species have gray or silver coats",
"foxes have bushy tails, which they use for balance and communication",
"they have large, expressive eyes and a narrow snout",
"foxes have sharp teeth and claws, which they use for hunting small animals and insects",
"they are highly adaptable and can be found in various habitats, including forests, grasslands, and urban areas",
"foxes are known for their intelligence and are often depicted in folklore and literature"],

'opossum':[
"an opossum is a small, nocturnal marsupial found in North and South America",
"its fur is grayish-white with black guard hairs, giving it a grizzled appearance",
"opossums have a long, pointed snout and large, round ears",
"they have a prehensile tail that they use for climbing and grasping objects",
"opossums are known for playing dead (playing possum) as a defense mechanism",
"they have opposable thumbs on their hind feet, which helps them grasp branches and food",
"opossums are omnivorous and eat a variety of foods, including fruits, insects, and small animals"],

'rabbit':[
"a rabbit is a small, herbivorous mammal known for its long ears and hopping gait",
"its fur is soft and can be various shades of brown, gray, or white",
"rabbits have large, sensitive ears that can rotate independently to detect sounds",
"they have long, powerful hind legs that enable them to leap great distances",
"rabbits have small front paws and sharp claws used for digging burrows and grooming",
"they are primarily nocturnal or crepuscular, feeding on grasses, leaves, and bark",
"rabbits are known for their rapid reproduction and are a favorite prey for many predators"],

'raccoon':[
"a raccoon is a medium-sized nocturnal mammal with a distinctive black mask around its eyes",
"its fur is thick and usually grayish-brown with black rings around its tail",
"raccoons have dexterous front paws with five fingers, which they use for manipulating objects",
"they have large, expressive eyes and a pointed snout",
"raccoons are omnivorous and eat a variety of foods, including fruits, insects, and small animals",
"they are known for their intelligence and problem-solving skills",
"raccoons are adaptable and can be found in both urban and rural environments"],

'rodent':[
"a rodent is a small mammal characterized by continuously growing incisors",
"they have a wide range of appearances, from tiny mice to larger species like beavers",
"rodents have long, gnawing incisors that they use to cut through hard materials",
"their fur can vary in color and texture, providing camouflage in their respective habitats",
"rodents are known for their prolific breeding and are found in almost every part of the world",
"they are omnivorous, eating plants, insects, and other small animals",
"rodents play important roles in ecosystems, serving as both predator and prey"],

'skunk':[
"a skunk is a small, nocturnal mammal known for its potent odor as a defense mechanism",
"its fur is typically black with white stripes running down its body",
"skunks have short legs and a squat posture",
"they have large, dark eyes and a small, pointed snout",
"skunks can spray a foul-smelling liquid from their anal glands to deter predators",
"they are omnivorous and eat a variety of foods, including insects, small animals, and plants",
"skunks are generally solitary animals and are found in various habitats, including forests, grasslands, and suburban areas"],

'squirrel':[
"a squirrel is a small, agile mammal known for its bushy tail and ability to climb trees",
"its fur can be various colors, including gray, brown, and red",
"squirrels have large, bushy tails that they use for balance and communication",
"they have sharp claws and teeth, which they use for climbing and gnawing",
"squirrels are diurnal and are active during the day, especially in the morning and evening",
"they are omnivorous and eat a variety of foods, including nuts, seeds, and fruits",
"squirrels are known for their gathering behavior, storing food for the winter",]
}

model_clip, preprocess_clip = clip.load(f'ViT-B/16', device)
model_clip.to(device)

LLMs ={'LLAMA':[camera_trap_templates_serengeti_LLAMA,camera_trap_templates_terra_LLAMA],'Phi':[camera_trap_templates_serengeti_Phi,camera_trap_templates_terra_Phi],'Qwen':[camera_trap_templates_serengeti_Qwen,camera_trap_templates_terra_Qwen]}
for LLM_i in LLMs.keys():
    serengeti_templates=LLMs[LLM_i][0]
    terra_templates= LLMs[LLM_i][1]

    zeroshot_weights = zeroshot_classifier(class_indices_serengeti, camera_trap_templates1, serengeti_templates)
    torch.save(zeroshot_weights,f'features/Features_serengeti/standard_features/Text_features_16_{LLM_i}.pt')

    zeroshot_weights = zeroshot_classifier(list(class_indices_terra.keys()), camera_trap_templates1, terra_templates)
    torch.save(zeroshot_weights,f'features/Features_terra/standard_features/Text_features_16_{LLM_i}.pt')
