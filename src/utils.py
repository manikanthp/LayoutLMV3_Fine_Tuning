# utils

import json

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

