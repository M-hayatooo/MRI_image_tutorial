import json
import pickle
import socket
from ast import Not
from pathlib import Path
import glob
import numpy as np
import tqdm


def load_adni(
    datasets={"ADNI2", "ADNI2-2"},
    classes={
        "CN",
        "AD",
        "MCI",
        "EMCI",
        "LMCI",
        "SMC",
    },
    size="full",
    unique=True,
    blacklist=True,
    not_csv = False,
    dryrun=False,
):
    root_dir = "/data2" if socket.gethostname().startswith("plant-ai") else "/home"
    folder = glob.glob(root_dir + "/radiology_datas/clean/meta/json/*")
    all_subjects = []
    for f in folder:
        all_subjects += json.loads(Path(f).read_text())
                       
    matching_images = []
    for subject in all_subjects:
        if subject["dataset"] not in datasets:
            continue
        if subject["class"] not in classes:
            continue
        if not_csv and subject["not_csv"]:
            continue
        
        for image in subject["images"]:
            if image["blacklisted"] and blacklist:
                continue
            image["pid"] = subject["id"]
            image["class"] = subject["class"]
            image["dataset"] = subject["dataset"]
            matching_images.append(image)
            if unique:
                break
                
    if not dryrun:
        image_loaders = {
            ".pkl": lambda pkl_path: pickle.loads(pkl_path.read_bytes()),
            ".npy": lambda npy_path: np.load(npy_path),
        }

        for image in tqdm.tqdm(matching_images):
            if size == "half":
                img_path = root_dir / Path(image["halfsize_img_path"])
            if size == "full":
                img_path = root_dir / Path(image["fullsize_img_path"])
            image["voxel"] = image_loaders[img_path.suffix](img_path)
                    
    return matching_images

    