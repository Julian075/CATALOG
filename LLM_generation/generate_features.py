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

"hmpala":[
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



omg = np.arange(0, 1.2, 0.2)
for mode_clip_i in mode_clip:
    model_clip, preprocess_clip = clip.load(f'ViT-B/{mode_clip_i}', device)
    model_clip.to(device)
    for omg_i in omg:
        zeroshot_weights = zeroshot_classifier(class_indices_serengeti, camera_trap_templates1, camera_trap_templates_serengeti,omg_i)
        torch.save(zeroshot_weights,f'../features/Features_serengeti/standard_features/Text_{mode_clip_i}_Ab_omg_{omg_i}.pt')

        zeroshot_weights = zeroshot_classifier(list(class_indices_terra.keys()), camera_trap_templates1, camera_trap_templates_terra,omg_i)
        torch.save(zeroshot_weights,f'../features/Features_terra/standard_features/Text_{mode_clip_i}_Ab_omg_{omg_i}.pt')
