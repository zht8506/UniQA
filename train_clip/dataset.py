import clip
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random

class image_caption_dataset(Dataset):
    def __init__(self, data_root, img_cap_path, preprocess):
        img_caps=[]
        for i in img_cap_path:
            img_cap=json.load(open(i))
            img_caps.extend(img_cap)
        image_path = []
        captions = []
        print(len(img_caps))

        for i in img_caps:
            image_name=i["image_path"]
            image_path.append(os.path.join(data_root,image_name))
            captions.append(i["prompt"])

        self.image_path = image_path
        self.captions = captions
        self.preprocess=preprocess

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]))
        captions = self.captions[idx]
        if isinstance(captions,str):
            captions=[captions]
        print(len(captions))
        captions_token=clip.tokenize(captions,truncate=True)
        index=random.randint(0, len(captions)-1) # random sample a caption
        caption=captions_token[index]

        return image, caption