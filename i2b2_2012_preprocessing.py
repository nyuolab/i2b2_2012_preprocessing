#!/usr/bin/env python
# coding: utf-8

import shutil
from pathlib import Path
import os
import sys
from collections import defaultdict
import numpy as np
import pathlib
import json
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from shutil import copyfile
import csv
from spacy.lang.en import English
from utils import check_exists, get_paths, make_dir, copy_text, dataset_xml2brat, get_ann_files,\
                    brat2bio_dict, make_if_nonexist, unzip_if_not_exists
import random 

# change setting here
cwd = os.getcwd() # assume the tar.gz files are in the current directory
gzs = {"train": "2012-07-15.original-annotation.release.tar.gz",
       "test": "2012-08-08.test-data.event-timex-groundtruth.tar.gz"}
BRAT_TEMP = "T{}\t{} {} {}\t{}"
EVENTS = {'PROBLEM', 'TEST', 'TREATMENT', 'CLINICAL_DEPT', 'EVIDENTIAL', 'OCCURRENCE'}
SPEC = {'&': 'AAMMPP'}
random_state = 13
train_val_split_ratio = 0.7 # 70 percent train, 30 percent val for train_gz
data_dir = "dataset"
verbose = False
folders = {"train": ".".join(gzs["train"].split(".")[:-2]),
           "test":  ".".join(gzs["test"].split(".")[:-2])}
folders["test"] = os.path.join(folders["test"], "xml") # test.gz has slightly different folder structure
# folder for raw files
infiles = {"train": os.path.join(cwd, folders["train"]),
           "test": os.path.join(cwd, folders["test"])}
# folder for brat files
brat_out = {"train": "brat-train",
            "test": "brat-test"}
# folder for bio files
bio_out = {"train": "bio-train",
           "test": "bio-test"}

print("========preprocessing starts!===========")
# ## Step 1: convert xml to BRAT
# Code from Xi Yang (University of Florida)
# 
# Modified by Lavender Jiang

print("========xml to BRAT===========")

for key in gzs:
    if not check_exists(cwd, gzs[key]):
        raise RuntimeError(f"Please make sure you have downloaded {gzs[key]} from n2c2 portal!")

for key in folders:
    folder = folders[key]
    if key == "test": 
        # for test folder, combine txt files and xml files to one folder
        # this ensures test/xml has similar structure as train folder
        folder_up = Path(folder).parent
        unzip_if_not_exists(cwd, folder_up, gzs[key])
        print(f"folder_up is {folder_up}")
        print(f"copying txt file from {folder_up}/i2b2/*.txt to {folder}")
        os.system(f"cp {folder_up}/i2b2/*.txt {folder}")
    else:
        unzip_if_not_exists(cwd, folder, gzs[key])
        
splits = gzs.keys()
in_paths = get_paths(infiles, splits)
out_paths = get_paths(brat_out, splits)
make_dir(out_paths)
copy_text(in_paths, out_paths)

dataset_xml2brat(in_paths, out_paths, BRAT_TEMP, EVENTS, verbose=verbose)


# ## Step 2: convert BRAT to BIO
# Code from Xi Yang (University of Florida)
# 
# Modified by Lavender Jiang

print("========BRAT to BIO===========")
nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe("sentencizer")
ann_files_d = get_ann_files(brat_out, ["train", "test"])
# print(f"ann_files_d is {ann_files_d}")
make_dir(bio_out)
brat2bio_dict(ann_files_d, brat_out, bio_out, nlp, EVENTS)


# ## Step 3: Combine files to dataset
print("========Building BIO dataset===========")

train_ids, dev_ids = train_test_split(list(ann_files_d['train']), train_size=train_val_split_ratio, random_state=random_state, shuffle=True)
test_ids = list(ann_files_d['test'])
random.Random(random_state).shuffle(test_ids)
print(f"train size {len(train_ids)}, val size {len(dev_ids)}, test size {len(test_ids)}")
i2b2_datasets = {"train":train_ids, "dev":dev_ids, "test":test_ids}
json.dump(i2b2_datasets, open("i2b2_2012_datasets.json", "w", encoding="utf-8"))
make_if_nonexist(data_dir)

# Merge BIO format train, validation and test datasets
for split in ["train", "dev", "test"]:
    if split in ["train", "dev"]:
        outputpath = bio_out["train"]
    else:
        outputpath = bio_out["test"]
    split_dir = os.path.join(data_dir, split)
    make_if_nonexist(split_dir)
    with open(os.path.join(data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
        for fid in i2b2_datasets[split]:
            copyfile(f"{outputpath}/{fid}.bio.txt", os.path.join(split_dir,f"{fid}.bio.txt"))
            with open(f"{outputpath}/{fid}.bio.txt", "r", encoding="utf-8") as fr:
                txt = fr.read().strip()
                if txt != '':
                    f.write(txt)
                    f.write("\n\n")

# ## Step 4: Bio 2 Nemo
# Code from nVidia NeMo
print("========BIO to NeMo===========")
for split in splits:
    os.system(f'python bio2nemo.py --data_file {data_dir}/{split}.txt')

print("========preprocessing finished!===========")
