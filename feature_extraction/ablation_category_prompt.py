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
            template_embeddings /= template_embeddings.norm(dim=-1, keepdim=True)
            template_embeddings = template_embeddings.unsqueeze(0)

            texts2 = [template for template in templates2[classname]]  # format with class
            texts2 = clip.tokenize(texts2).to(device)
            description_embeddings = model_clip.encode_text(texts2)  # embed with text encoder
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = description_embeddings.mean(dim=0)
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = description_embeddings.unsqueeze(0)

            class_embedding = torch.cat ((template_embeddings *(1-omg),description_embeddings * omg), dim = 0)
            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm()  # Normalize final embedding


            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def zeroshot_classifier_op2(classnames, templates1, templates2,omg):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates1]
            texts = clip.tokenize(texts).to(device)
            template_embeddings = model_clip.encode_text(texts)  # embed with text encoder
            template_embeddings /= template_embeddings.norm(dim=-1, keepdim=True)

            texts2 = [template for template in templates2[classname]]  # format with class
            texts2 = clip.tokenize(texts2).to(device)
            description_embeddings = model_clip.encode_text(texts2)  # embed with text encoder
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = torch.cat ((template_embeddings *omg,description_embeddings * omg), dim = 0)
            class_embedding = class_embedding.mean(dim=0)
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

camera_trap_templates_terra = {
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

camera_trap_templates_serengeti = {
    "aardvark": [
        "aardvarks are nocturnal mammals with a distinctive elongated snout.",
        "aardvarks have a robust body covered in coarse, grayish-brown fur.",
        "aardvarks possess long, tubular ears that can be oriented independently.",
        "aardvarks have powerful limbs equipped with strong claws for digging.",
        "aardvarks use their long, sticky tongues to capture ants and termites.",
        "aardvarks have a large, muscular tail that aids in balance.",
        "aardvarks have small eyes adapted for night vision."
    ],

    "aardwolf": [
        "aardwolfs are small insectivorous mammals related to hyenas.",
        "aardwolfs have a slender body covered in coarse, yellowish-gray fur with dark stripes.",
        "aardwolfs possess large, pointed ears adapted for hearing insects underground.",
        "aardwolfs have a long, bushy tail with dark tips.",
        "aardwolfs use their sticky tongues to feed primarily on termites.",
        "aardwolfs have sharp claws for digging and foraging.",
        "aardwolfs have a mane of longer hair running along their back."
    ],

    "baboon": [
        "baboons are large primates with powerful builds.",
        "baboons have a distinctive dog-like snout and large canine teeth.",
        "baboons possess thick fur that can range from olive to yellowish-brown.",
        "baboons have a short tail and prominent, bare, colored skin patches on their buttocks.",
        "baboons live in complex social structures called troops.",
        "baboons have strong limbs adapted for both quadrupedal walking and climbing.",
        "baboons exhibit a wide range of vocalizations and facial expressions."
    ],

    "batEaredFox": [
        "batEaredFoxes are small canids with large, distinctive ears.",
        "batEaredFoxes have a slender body covered in yellowish-gray fur with black patches on their limbs and face.",
        "batEaredFoxes use their large ears to detect insects underground.",
        "batEaredFoxes possess a bushy tail with a black tip.",
        "batEaredFoxes primarily feed on insects, especially termites and ants.",
        "batEaredFoxes have sharp, curved claws for digging.",
        "batEaredFoxes have keen night vision and are primarily nocturnal."
    ],

    "buffalo": [
        "buffalo are large, robust bovines native to Africa.",
        "buffalo have a heavy build with a large head and short, thick neck.",
        "buffalo possess a dark brown to black coat, often with a tuft of hair on their forehead.",
        "buffalo have curved horns that extend outward and upward.",
        "buffalo live in large herds and exhibit strong social bonds.",
        "buffalo have powerful legs adapted for long-distance walking and running.",
        "buffalo are known for their aggressive nature and formidable strength."
    ],

    "bushbuck": [
        "bushbucks are medium-sized antelopes with a shy and solitary nature.",
        "bushbucks have a reddish-brown coat with white spots and stripes.",
        "bushbucks possess slender bodies and long legs adapted for agile movement.",
        "bushbucks have short, spiraled horns, typically found in males.",
        "bushbucks have large, expressive eyes and rounded ears.",
        "bushbucks are known for their excellent jumping abilities.",
        "bushbucks prefer dense underbrush and forested areas for cover."
    ],

    "caracal": [
        "caracals are medium-sized wild cats with distinctive tufted ears.",
        "caracals have a sleek, muscular body covered in reddish-brown fur.",
        "caracals possess long legs and a short tail.",
        "caracals have large, expressive eyes with a keen sense of sight.",
        "caracals use their strong hind legs to leap high and catch birds in mid-air.",
        "caracals have sharp retractable claws and strong jaws for hunting.",
        "caracals are solitary and primarily nocturnal hunters."
    ],

    "cheetah": [
        "cheetahs are large cats known for their incredible speed.",
        "cheetahs have a slender, streamlined body covered in tan fur with black spots.",
        "cheetahs possess a small, rounded head with distinctive black 'tear marks' running from the eyes to the mouth.",
        "cheetahs have long legs and a deep chest adapted for fast running.",
        "cheetahs possess a flexible spine and non-retractable claws for better traction.",
        "cheetahs have a long tail that helps with balance during high-speed chases.",
        "cheetahs are primarily diurnal hunters, relying on sight to locate prey."
    ],

    "civet": [
        "civets are small, nocturnal mammals with a cat-like appearance.",
        "civets have a slender body covered in coarse, grayish-brown fur with dark spots or stripes.",
        "civets possess a long, bushy tail and pointed ears.",
        "civets have sharp claws and strong limbs for climbing.",
        "civets produce a musky scent from their perineal glands.",
        "civets have large eyes adapted for night vision.",
        "civets are omnivorous, feeding on fruits, insects, and small vertebrates."
    ],

    "dikDik": [
        "dikDiks are small antelopes known for their diminutive size.",
        "dikDiks have a slender body covered in short, reddish-brown fur.",
        "dikDiks possess large, dark eyes and a pointed snout.",
        "dikDiks have small, delicate legs adapted for agile movement.",
        "dikDiks have short, pointed horns, typically found in males.",
        "dikDiks are known for their distinctive alarm calls that sound like 'dik-dik'.",
        "dikDiks prefer dense bush and thicket habitats for cover."
    ],

    "eland": [
        "elands are one of the largest antelope species.",
        "elands have a heavy, muscular body covered in short, tan or grayish-brown fur.",
        "elands possess long, spiraled horns found in both males and females.",
        "elands have a large dewlap and a prominent hump on their shoulders.",
        "elands are known for their ability to jump high despite their size.",
        "elands have large, expressive eyes and long ears.",
        "elands prefer open grasslands and savannas for grazing."
    ],

    "elephant": [
        "elephants are the largest land mammals.",
        "elephants have a large, robust body covered in thick, gray skin.",
        "elephants possess long trunks used for grasping objects, drinking, and communication.",
        "elephants have large ears that help regulate body temperature.",
        "elephants possess long, curved tusks made of ivory, typically found in males.",
        "elephants have strong, pillar-like legs to support their massive weight.",
        "elephants are known for their intelligence, social behavior, and long memory."
    ],

    "gazelleGrants": [
        "gazelleGrants are medium-sized antelopes with slender bodies.",
        "gazelleGrants have a tan coat with white underparts and a distinctive white rump patch.",
        "gazelleGrants possess long, lyre-shaped horns, found in both males and females.",
        "gazelleGrants have large, dark eyes and long ears.",
        "gazelleGrants are known for their graceful, bounding leaps.",
        "gazelleGrants prefer open savannas and grasslands for grazing.",
        "gazelleGrants are social animals, often forming small herds."
    ],

    "gazelleThomsons": [
        "gazelleThomsons are small, agile antelopes.",
        "gazelleThomsons have a tan coat with white underparts and a black stripe running along their sides.",
        "gazelleThomsons possess short, slender horns found in both males and females.",
        "gazelleThomsons have large, dark eyes and pointed ears.",
        "gazelleThomsons are known for their speed and agility.",
        "gazelleThomsons prefer open savannas and grasslands for grazing.",
        "gazelleThomsons are social animals, often forming large herds."
    ],

    "genet": [
        "genets are small, slender carnivores with a cat-like appearance.",
        "genets have a long, slender body covered in spotted or striped fur.",
        "genets possess a long, bushy tail with black rings.",
        "genets have sharp claws and strong limbs for climbing.",
        "genets have large eyes adapted for night vision.",
        "genets produce a musky scent from their perineal glands.",
        "genets are omnivorous, feeding on small vertebrates, insects, and fruits."
    ],

    "giraffe": [
        "giraffes are the tallest land animals.",
        "giraffes have a long neck and legs, covered in a distinctive patterned coat of brown patches on a tan background.",
        "giraffes possess ossicones, which are short, horn-like structures on their heads.",
        "giraffes have a long, prehensile tongue used for grasping leaves.",
        "giraffes have large eyes and excellent vision.",
        "giraffes are known for their graceful, slow movements.",
        "giraffes prefer open woodlands and savannas where they can browse on trees."
    ],

    "guineaFowl": [
        "guineaFowls are ground-dwelling birds with a distinctive appearance.",
        "guineaFowls have a round body covered in dark plumage with white spots.",
        "guineaFowls possess a bare, bluish head with a casque on top.",
        "guineaFowls have strong legs adapted for running.",
        "guineaFowls produce loud, distinctive calls.",
        "guineaFowls are social birds, often found in large flocks.",
        "guineaFowls feed on insects, seeds, and small vertebrates."
    ],

    "hare": [
        "hares are fast-running mammals with long ears and powerful hind legs.",
        "hares have a slender body covered in soft, brownish-gray fur.",
        "hares possess large, dark eyes and long whiskers.",
        "hares have long, powerful hind legs adapted for running and jumping.",
        "hares are known for their speed and agility.",
        "hares are solitary animals, often found in open fields and grasslands.",
        "hares feed on a variety of vegetation, including grasses, leaves, and bark."
    ],

    "hartebeest": [
        "hartebeests are large antelopes with a distinctive elongated face.",
        "hartebeests have a reddish-brown coat with lighter underparts.",
        "hartebeests possess long, lyre-shaped horns found in both males and females.",
        "hartebeests have a robust body and long legs adapted for running.",
        "hartebeests are known for their speed and endurance.",
        "hartebeests prefer open savannas and grasslands for grazing.",
        "hartebeests are social animals, often forming large herds."
    ],

    "hippopotamus": [
        "hippopotamuses are large, semi-aquatic mammals.",
        "hippopotamuses have a massive body covered in thick, hairless skin.",
        "hippopotamuses possess a large head with a wide mouth and tusk-like teeth.",
        "hippopotamuses have small eyes and ears located on the top of their heads.",
        "hippopotamuses spend much of their time in water to keep cool.",
        "hippopotamuses have powerful legs adapted for walking on land and in water.",
        "hippopotamuses are known for their aggressive behavior and formidable strength."
    ],

    "honeyBadger": [
        "honeyBadgers are small carnivores known for their strength and ferocity.",
        "honeyBadgers have a stocky build with coarse, black and white fur.",
        "honeyBadgers possess sharp claws and powerful limbs for digging.",
        "honeyBadgers have a broad head with small ears and a strong jaw.",
        "honeyBadgers are known for their fearless nature and ability to fend off larger predators.",
        "honeyBadgers feed on a variety of prey, including insects, small mammals, and honey.",
        "honeyBadgers are primarily solitary and nocturnal hunters."
    ],
    "hyenaSpotted": [
        "hyenaSpotted are large carnivores known for their powerful build and social structure.",
        "hyenaSpotted have a tan coat with dark spots and a shaggy mane.",
        "hyenaSpotted possess strong jaws and teeth for crushing bones.",
        "hyenaSpotted have large, rounded ears and a robust head.",
        "hyenaSpotted live in complex social groups called clans.",
        "hyenaSpotted are known for their distinctive 'laughing' vocalizations.",
        "hyenaSpotted are both hunters and scavengers, feeding on a variety of prey."
    ],

    "hyenaStriped": [
        "hyenaStriped are medium-sized carnivores with a distinctive striped coat.",
        "hyenaStriped have a grayish-brown coat with black stripes and a shaggy mane.",
        "hyenaStriped possess strong jaws and teeth for crushing bones.",
        "hyenaStriped have large, rounded ears and a robust head.",
        "hyenaStriped are primarily solitary and nocturnal.",
        "hyenaStriped are known for their ability to scavenge and hunt small prey.",
        "hyenaStriped produce a range of vocalizations, including growls and howls."
    ],

    "impala": [
        "impalas are medium-sized antelopes known for their agility and grace.",
        "impalas have a reddish-brown coat with lighter underparts and a distinctive black 'M' marking on their rump.",
        "impalas possess long, lyre-shaped horns found in males.",
        "impalas have slender bodies and long legs adapted for jumping and running.",
        "impalas are known for their remarkable leaping abilities.",
        "impalas prefer open woodlands and savannas for grazing.",
        "impalas are social animals, often forming large herds."
    ],

    "jackal": [
        "jackals are medium-sized canids with a sleek, slender build.",
        "jackals have a coat that can range from golden to grayish-brown, often with a darker saddle on their back.",
        "jackals possess large ears and a pointed snout.",
        "jackals have long legs adapted for running and hunting.",
        "jackals are known for their distinctive howling vocalizations.",
        "jackals are opportunistic feeders, hunting small mammals and scavenging.",
        "jackals are social animals, often living in pairs or small family groups."
    ],

    "koriBustard": [
        "koriBustard are large, ground-dwelling birds native to Africa.",
        "koriBustard have a grayish-brown plumage with white underparts and black markings.",
        "koriBustard possess a long neck and strong legs adapted for walking.",
        "koriBustard have a broad head with a distinctive crest.",
        "koriBustard are known for their slow, deliberate movements and ground-feeding habits.",
        "koriBustard feed on insects, small vertebrates, and plant matter.",
        "koriBustard are primarily solitary and can perform impressive displays during courtship."
    ],

    "leopard": [
        "leopards are large, solitary cats known for their agility and stealth.",
        "leopards have a sleek, muscular body covered in a golden coat with black rosettes.",
        "leopards possess a large head and powerful jaws.",
        "leopards have long legs and a long tail for balance.",
        "leopards are excellent climbers and often rest in trees.",
        "leopards are primarily nocturnal hunters, relying on stealth and strength.",
        "leopards are adaptable and can live in a variety of habitats, from forests to savannas."
    ],

    "lionFemale": [
        "lionFemale are large, social cats known for their role in hunting and nurturing the pride.",
        "lionFemale have a tawny coat without the mane found in males.",
        "lionFemale possess strong, muscular bodies adapted for hunting large prey.",
        "lionFemale have keen senses of sight, smell, and hearing.",
        "lionFemale are the primary hunters in the pride, working cooperatively to take down prey.",
        "lionFemale are known for their strong social bonds and care for their cubs.",
        "lionFemale live in prides that include related females and their offspring."
    ],

    "lionMale": [
        "lionMale are large, powerful cats known for their impressive manes and dominance.",
        "lionMale have a tawny coat with a mane that varies in color from blond to black.",
        "lionMale possess strong, muscular bodies and powerful jaws.",
        "lionMale have keen senses of sight, smell, and hearing.",
        "lionMale are known for their role in protecting the pride and marking territory.",
        "lionMale may participate in hunting, especially for larger prey.",
        "lionMale live in prides that include related females, cubs, and occasionally other males."
    ],

    "mongoose": [
        "mongooses are small, agile carnivores with a sleek, slender body.",
        "mongooses have a coat that can range from gray to brownish, often with a speckled pattern.",
        "mongooses possess sharp claws and strong limbs for digging and climbing.",
        "mongooses have a long, bushy tail and a pointed snout.",
        "mongooses are known for their quick reflexes and ability to hunt venomous snakes.",
        "mongooses are social animals, often living in large groups called mobs.",
        "mongooses are omnivorous, feeding on insects, small vertebrates, and fruits."
    ],

    "ostrich": [
        "ostriches are the largest and heaviest birds.",
        "ostriches have a large, flightless body covered in soft, downy feathers.",
        "ostriches possess long, powerful legs adapted for running at high speeds.",
        "ostriches have a long neck and small head with large eyes.",
        "ostriches are known for their strong, swift kicks.",
        "ostriches are social birds, often found in flocks.",
        "ostriches feed on a variety of vegetation, including grasses, leaves, and seeds."
    ],

    "porcupine": [
        "porcupines are large rodents known for their protective quills.",
        "porcupines have a robust body covered in sharp, barbed quills.",
        "porcupines possess strong, curved claws adapted for climbing and digging.",
        "porcupines have a rounded head with small ears and large, dark eyes.",
        "porcupines are primarily nocturnal, feeding on bark, leaves, and fruits.",
        "porcupines are solitary animals, often found in forests and wooded areas.",
        "porcupines use their quills as a defense mechanism against predators."
    ],

    "reedbuck": [
        "reedbucks are medium-sized antelopes with a shy and elusive nature.",
        "reedbucks have a reddish-brown coat with lighter underparts.",
        "reedbucks possess short, curved horns found in males.",
        "reedbucks have a slender body and long legs adapted for running.",
        "reedbucks are known for their distinctive whistling alarm calls.",
        "reedbucks prefer wetlands and grasslands near water sources.",
        "reedbucks are primarily grazers, feeding on grasses and sedges."
    ],

    "reptiles": [
        "reptiles are a diverse group of cold-blooded vertebrates.",
        "reptiles have scaly skin that helps prevent water loss.",
        "reptiles possess a wide range of body shapes and sizes, from small lizards to large crocodiles.",
        "reptiles lay eggs or give birth to live young, depending on the species.",
        "reptiles are ectothermic, relying on external sources of heat to regulate their body temperature.",
        "reptiles include snakes, lizards, turtles, and crocodiles.",
        "reptiles are found in a variety of habitats, from deserts to rainforests."
    ],

    "rhinoceros": [
        "rhinoceroses are large, heavy-bodied mammals known for their thick skin and horns.",
        "rhinoceroses have a massive body covered in gray or brown skin.",
        "rhinoceroses possess one or two horns on their snout, depending on the species.",
        "rhinoceroses have small eyes and ears with a keen sense of smell.",
        "rhinoceroses are known for their aggressive behavior and formidable strength.",
        "rhinoceroses are primarily solitary animals, often found in grasslands and savannas.",
        "rhinoceroses are herbivores, feeding on grasses, leaves, and branches."
    ],

    "rodents": [
        "rodents are a diverse group of small mammals known for their continuously growing incisors.",
        "rodents have a wide range of body shapes and sizes, from tiny mice to large capybaras.",
        "rodents possess sharp incisors that are used for gnawing.",
        "rodents have a high reproductive rate and can produce multiple litters per year.",
        "rodents are found in a variety of habitats, from forests to grasslands.",
        "rodents are herbivorous, omnivorous, or sometimes carnivorous, depending on the species.",
        "rodents play important ecological roles, including seed dispersal and serving as prey for many predators."
    ],

    "secretaryBird": [
        "secretary birds are large, terrestrial birds of prey.",
        "secretary birds have a slender body covered in light gray feathers with black flight feathers.",
        "secretary birds possess long legs adapted for walking and hunting on the ground.",
        "secretary birds have a distinctive crest of long, dark feathers on their head.",
        "secretary birds are known for their unique hunting technique of stomping on prey.",
        "secretary birds are solitary or found in pairs, often in open grasslands and savannas.",
        "secretary birds feed on insects, small mammals, and reptiles."
    ],

    "serval": [
        "servals are medium-sized wild cats with a slender, agile body.",
        "servals have a coat covered in golden-yellow fur with black spots and stripes.",
        "servals possess long legs and a short tail.",
        "servals have large ears and a small head with a pointed snout.",
        "servals are known for their exceptional hunting skills, particularly for catching birds.",
        "servals are solitary animals, often found in savannas and wetlands.",
        "servals feed on a variety of prey, including rodents, birds, and insects."
    ],

    "topi": [
        "topis are large antelopes with a striking appearance.",
        "topis have a reddish-brown coat with darker patches on their face, shoulders, and legs.",
        "topis possess long, ringed horns found in both males and females.",
        "topis have a robust body and long legs adapted for running.",
        "topis are known for their remarkable speed and endurance.",
        "topis prefer open grasslands and savannas for grazing.",
        "topis are social animals, often forming large herds."
    ],

    "vervetMonkey": [
        "vervetMonkeys are small primates known for their distinctive black faces and grayish fur.",
        "vervetMonkeys have a slender body with long limbs adapted for climbing.",
        "vervetMonkeys possess a long, prehensile tail used for balance.",
        "vervetMonkeys live in complex social groups called troops.",
        "vervetMonkeys are known for their vocalizations and alarm calls that warn of predators.",
        "vervetMonkeys are omnivorous, feeding on fruits, leaves, insects, and small vertebrates.",
        "vervetMonkeys are found in a variety of habitats, including savannas, forests, and riverine areas."
    ],

    "warthog": [
        "warthogs are large, wild pigs known for their distinctive tusks and facial warts.",
        "warthogs have a robust body covered in sparse, bristly hair.",
        "warthogs possess long, curved tusks and prominent facial warts.",
        "warthogs have a large head with a broad snout adapted for digging.",
        "warthogs are known for their ability to run at high speeds.",
        "warthogs prefer open savannas and grasslands for grazing and digging.",
        "warthogs are social animals, often found in small family groups."
    ],

    "waterbuck": [
        "waterbucks are large antelopes known for their robust build and shaggy coat.",
        "waterbucks have a reddish-brown to grayish-brown coat with a white ring on their rump.",
        "waterbucks possess long, spiral horns found in males.",
        "waterbucks have a large, muscular body and long legs adapted for running.",
        "waterbucks prefer habitats near water sources such as rivers and lakes.",
        "waterbucks are known for their strong swimming abilities.",
        "waterbucks feed on grasses and aquatic plants."
    ],

    "wildcat": [
        "wildcats are small to medium-sized cats known for their solitary and elusive nature.",
        "wildcats have a coat that varies from gray to brownish with dark stripes or spots.",
        "wildcats possess sharp claws and strong limbs for climbing and hunting.",
        "wildcats have a rounded head with large ears and keen senses.",
        "wildcats are primarily nocturnal hunters, feeding on small mammals, birds, and insects.",
        "wildcats are found in a variety of habitats, from forests to grasslands.",
        "wildcats are known for their territorial behavior and solitary lifestyle."
    ],

    "wildebeest": [
        "wildebeests are large, migratory antelopes known for their massive herds.",
        "wildebeests have a sturdy body covered in grayish-brown fur with a darker mane.",
        "wildebeests possess long, curved horns found in both males and females.",
        "wildebeests have a large head and a broad, muscular neck.",
        "wildebeests are known for their spectacular annual migrations.",
        "wildebeests prefer open grasslands and savannas for grazing.",
        "wildebeests are social animals, often forming large herds during migration."
    ],

    "zebra": [
        "zebras are large equids known for their distinctive black and white stripes.",
        "zebras have a robust body covered in short, bristly hair.",
        "zebras possess long, slender legs adapted for running.",
        "zebras have a large head with a prominent mane running down their neck.",
        "zebras are known for their social behavior and complex vocalizations.",
        "zebras prefer open grasslands and savannas for grazing.",
        "zebras are social animals, often forming large herds for protection."
    ],

    "zorilla": [
        "zorillas are small carnivorous mammals known for their distinctive fur and their ability to release a strong odor as a defense mechanism.",
        "zorillas have black fur with white stripes running along their bodies.",
        "zorillas possess a small head with rounded ears and a pointed snout.",
        "zorillas have strong claws adapted for digging.",
        "zorillas are nocturnal and solitary, preferring to come out at night to search for food.",
        "zorillas feed on insects, small vertebrates, and occasionally fruits.",
        "zorillas are known for their defensive behavior, emitting an unpleasant odor when they feel threatened."
    ]
}


#omg = np.round(np.arange(0, 1.1, 0.1),2)
omg=[0.9,1.0]
for mode_clip_i in mode_clip:
    model_clip, preprocess_clip = clip.load(f'ViT-B/{mode_clip_i}', device)
    model_clip.to(device)
    for omg_i in omg:
        zeroshot_weights = zeroshot_classifier(class_indices_serengeti, camera_trap_templates1, camera_trap_templates_serengeti,omg_i)
        torch.save(zeroshot_weights,f'features/Features_serengeti/standard_features/Text_{mode_clip_i}_Ab3_{omg_i}_2.pt')

        zeroshot_weights = zeroshot_classifier(list(class_indices_terra.keys()), camera_trap_templates1, camera_trap_templates_terra,omg_i)
        torch.save(zeroshot_weights,f'features/Features_terra/standard_features/Text_{mode_clip_i}_Ab3_{omg_i}_2.pt')





