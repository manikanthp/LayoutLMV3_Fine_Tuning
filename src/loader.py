# Loader

from utils import *
import torch
from tqdm import tqdm
from PIL import Image


class dataSet:
    def __init__(self,json_path,processor=None) -> None:
        self.json_data = train_data_format(read_json(json_path))

        self.processor = processor


    def __len__(self)->int:
        # print(self.json_data)
        return len(self.json_data)

    def __getitem__(self,index)->dict:
        imgs = []
        words = []
        label = []
        bboxes = []
        data = self.json_data[index]
        
        imgs.append(Image.open(data['img_path']).convert('RGB'))
        words.append(data['tokens'])
        label.append(data['ner_tag'])
        bboxes.append(data['bboxes'])

        encoding = self.processor(
            imgs,
            words,
            boxes = bboxes,
            word_labels = label,
            max_length=512,padding="max_length",truncation="longest_first",return_tensors='pt'
        )

        return {
            "input_ids" : torch.tensor(encoding["input_ids"],dtype=torch.int64).flatten(),
            "attention_mask" : torch.tensor(encoding["attention_mask"],dtype=torch.int64).flatten(),
            "bbox" : torch.tensor(encoding["bbox"],dtype=torch.int64).flatten(end_dim=1),
            "pixel_values" : torch.tensor(encoding["pixel_values"],dtype=torch.float32).flatten(end_dim=1),
            "lables" : torch.tensor(encoding["labels"],dtype=torch.int64)
        }

