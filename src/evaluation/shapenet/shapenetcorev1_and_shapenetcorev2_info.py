import os
import pickle
shapenetcorev1_subset_cats = {
    "04530566": "watercraft",
    "04090263": "file",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "speaker",
    "02933112": "cabinet",
    "03001627": "chair",
    "02828884": "bench",
    "02958343": "car",
    "02691156": "airplane",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "phone",
}
# These categories are [airplane, bench, cabinet, car, chair, display, lamp, speaker, rifle, sofa, table, phone, watercraf]
synsteIds_categories_shapentcorev2 = {
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
    "rifle": 25,
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
    "file": 44,
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
# Mine
synsteIds_categories_extracted_shapent13catgories = {
    "airplane": 0,
    "bench": 6,
    "cabinet": 12,
    "car": 16,
    "chair": 18,
    "display": 22,
    "lamp": 30,
    "speaker": 32,
    "rifle": 44,
    "sofa": 47,
    "table": 49,
    "phone": 17,
    "watercraft": 53,
}

def extract_eval_split_list(filelist_dir: str, val_split_list_dir: str, ext: str = "_test.lst"):
    all_files = os.listdir(filelist_dir)
    print(all_files)
    ext = "_test.lst"
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(file)

    for j in range(len(file_path_list)):
        file_name = file_path_list[j]
        file_name_path = filelist_dir + file_name
        # string = read_text_file(file_name_path)
        folder_name_dict = {}
        file_name_trim = file_name.replace(ext, "")
        folder_name_dict["folder_name"] = file_name_trim
        counter = 0
        with open(file_name_path, "r") as file:
            for line in file:
                # line = next(file).rstrip("\n")
                line = line.rstrip("\n")
                # folder_name_dict.update({line : counter})
                folder_name_dict[line] = counter
                counter += 1
                print(line)
        # write dict per folder name
        current_dict_name = val_split_list_dir + file_name_trim + ".pkl"
        with open(current_dict_name, "wb") as fp:
            pickle.dump(folder_name_dict, fp)
            print("dictionary saved successfully to file")


def extract_train_split_list(filelist_dir: str, train_split_list_dir: str, ext: str = "_train.lst"):
    all_files = os.listdir(filelist_dir)
    print(all_files)
    ext = "_train.lst"
    file_path_list = []
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith(ext):
            file_path_list.append(file)

    for j in range(len(file_path_list)):
        file_name = file_path_list[j]
        file_name_path = filelist_dir + file_name
        # string = read_text_file(file_name_path)
        folder_name_dict = {}
        file_name_trim = file_name.replace(ext, "")
        folder_name_dict["folder_name"] = file_name_trim
        counter = 0
        with open(file_name_path, "r") as file:
            for line in file:
                # line = next(file).rstrip("\n")
                line = line.rstrip("\n")
                # folder_name_dict.update({line : counter})
                folder_name_dict[line] = counter
                counter += 1
                print(line)
        # write dict per folder name
        current_dict_name = train_split_list_dir + file_name_trim + ".pkl"
        with open(current_dict_name, "wb") as fp:
            pickle.dump(folder_name_dict, fp)
            print("dictionary saved successfully to file")

# if __name__ == "__main__":
#     filelist_dir = "/graphics/scratch2/staff/zakeri/filelist_val_and_train_shapenetcorev1_offical_splits/filelists/"
#     train_split_list_dir = "/graphics/scratch2/staff/zakeri/train_split_shapenetcorev1_list_dir/"
#     extract_train_split_list(filelist_dir, train_split_list_dir)