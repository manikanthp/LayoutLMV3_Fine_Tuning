# utils
from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read_json(json_path:str)->dict:
    with open(json_path,'r') as fp:
        data = json.loads(fp.read())
    return data


def train_data_format(json_to_dict:list):

    final_list = []
    count=0
    for item in json_to_dict:
        count = count+1
        # print(item['annotations'])
        test_dict = {"id":int,"tokens":[],"bboxes":[],"ner_tag":[]}
        test_dict["id"] = count
        test_dict["img_path"] = item['file_name']
        for cont in item['annotations']:
            test_dict['tokens'].append(cont['text'])
            test_dict['bboxes'].append(cont['box'])
            test_dict['ner_tag'].append(cont['label'])


        final_list.append(test_dict)
    #print(final_list)
    return final_list


# making changes for inferencing

ocr = PaddleOCR(use_angle_cls=False, 
                lang='en',
                  rec=False,
                ) # need to run only once to download and load model into memory 



def scale_bounding_box(box:list[int], width:float, height:float) -> list[int]:

    return [
                100*box[0]/width,
                100*box[1]/height,
                (100*box[0]/width)+box[2],
                (100*box[1]/height)+box[3]
    ]


def process_bbox(box:list):
    return [box[0][0], box[1][1], box[2][0]-box[0][0], box[2][1]-box[1][1]]

def dataSetFormat(img_file):

    width, height = img_file.size

    ress = ocr.ocr(np.asarray(img_file))

    test_dict = {'tokens':[], "bboxes":[]}
    test_dict['img_path'] = img_file

    for item in ress[0]:
        
        test_dict['tokens'].append(item[1][0])
        test_dict['bboxes'].append(scale_bounding_box(process_bbox(item[0]), width, height))

    return test_dict, width, height




def plot_img(im, bbox_list, label_list, prob_list, width, height):

    plt.imshow(im)
    ax = plt.gca()
    for i, (item) in enumerate(zip(bbox_list)):
        #prob = str(round(prob, 2))
        item = item[0]
        print("Items :", item)
        rect = Rectangle((item[0]*width/100, item[1]*height/100), item[2]-item[0], item[3]-item[1], linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        # set the size
        #plot_width, plot_height = ax.get_xlim()[1], ax.get_ylim()[1]
        #text_size = min(plot_width, plot_height) * 0.05
        ax.text(
                    item[0]*width/100,
                    item[1]*height/100,
                    f"{label_list[i]}",
                    bbox={'facecolor': [1, 1, 1], 'alpha': 0.5},
                    clip_box = ax.clipbox,
                    clip_on = True
        )

    plt.show()
    plt.savefig("test_image.jpg")
    print("Done")
    plt.clf()