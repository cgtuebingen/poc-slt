import torch


def map_shapeNetCorev2(label):

    synsteIds_categories_mapping = {
        "airplane": 0,
        "trash can": 1,
        "bag": 2,
        "basket": 3,
        "bathtub": 4,
        "bed": 5,
        "bench": 6,
        "birdhouse": 7,
        "bookshelf": 8,
        "bottle": 9,
        "bowl": 10,
        "bus": 11,
        "cabinet": 12,
        "camera": 13,
        "can": 14,
        "cap": 15,
        "car": 16,
        "cell phone": 17,
        "chair": 18,
        "clock": 19,
        "keyboard": 20,
        "dishwasher": 21,
        "display": 22,
        "headphone": 23,
        "faucet": 24,
        "file": 25,
        "guitar": 26,
        "helmet": 27,
        "jar": 28,
        "knife": 29,
        "lamp": 30,
        "laptop": 31,
        "speaker": 32,
        "mailbox": 33,
        "microphone": 34,
        "microwave": 35,
        "bike": 36,
        "mug": 37,
        "piano": 38,
        "pillow": 39,
        "pistol": 40,
        "pot": 41,
        "printer": 42,
        "remote": 43,
        "rifle": 44,
        "rocket": 45,
        "skateboard": 46,
        "sofa": 47,
        "stove": 48,
        "table": 49,
        "telephone": 50,
        "tower": 51,
        "train": 52,
        "watercraft": 53,
        "washing machine": 54,
    }

    class_id = synsteIds_categories_mapping.get(label)
    return class_id


def make_target(batch_size: int, num_classes: int):
    # Example of target with class probabilities
    N = batch_size
    C = num_classes
    input = torch.rand(batch_size, num_classes, requires_grad=True)
    print("Input:", input, ", shape: ", input.shape)

    target = torch.empty(N, dtype=torch.long).random_(C)
    print("Target:", target, ", shape: ", target.shape)

    loss = torch.nn.CrossEntropyLoss()
    output = loss(input, target)
    # output.backward()
    print("Cross Entropy Loss:", output, ", shape: ", output.shape)
    print("Input grads: ", input.grad, ", shape: ", input.grad.shape)

    return output


def map_folder_to_label(folder_name: str):

    synsteIds_categories = {
        "02691156": "airplane",
        "02747177": "trash can",
        "02773838": "bag",
        "02801938": "basket",
        "02808440": "bathtub",
        "02818832": "bed",
        "02828884": "bench",
        "02843684": "birdhouse",
        "02871439": "bookshelf",
        "02876657": "bottle",
        "02880940": "bowl",
        "02924116": "bus",
        "02933112": "cabinet",
        "02942699": "camera",
        "02946921": "can",
        "02954340": "cap",
        "02958343": "car",
        "02992529": "cell phone",
        "03001627": "chair",
        "03046257": "clock",
        "03085013": "keyboard",
        "03207941": "dishwasher",
        "03211117": "display",
        "03261776": "headphone",
        "03325088": "faucet",
        "03337140": "file",
        "03467517": "guitar",
        "03513137": "helmet",
        "03593526": "jar",
        "03624134": "knife",
        "03636649": "lamp",
        "03642806": "laptop",
        "03691459": "speaker",
        "03710193": "mailbox",
        "03759954": "microphone",
        "03761084": "microwave",
        "03790512": "bike",
        "03797390": "mug",
        "03928116": "piano",
        "03938244": "pillow",
        "03948459": "pistol",
        "03991062": "pot",
        "04004475": "printer",
        "04074963": "remote",
        "04090263": "file",
        "04099429": "rocket",
        "04225987": "skateboard",
        "04256520": "sofa",
        "04330267": "stove",
        "04379243": "table",
        "04401088": "telephone",
        "04460130": "tower",
        "04468005": "train",
        "04530566": "watercraft",
        "04554684": "washing machine",
    }
    label = synsteIds_categories.get(folder_name)
    return label
