import os
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import json
from uuid import uuid4
import numpy as np


# loading the engine
# OCR enginer
ocr = PaddleOCR(use_angle_cls=False, 
                lang='en',
                  rec=False,
                ) # need to run only once to download and load model into memory 


images_folder_path  = "D:/Projects/AI_Projects/NLP/Document_AI/LayoutLM_Models/image" 


def create_image_url(filename):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    return f'http://localhost:8080/{filename}'



def convert_bounding_box(bounding_box):
    """Converts a bounding box of [x1, y1, x2, y2] into [x, y, height, width].

    Args:
    bounding_box: A list of four numbers, representing the x1, y1, x2, and y2
        coordinates of the bounding box.

    Returns:
    A list of four numbers, representing the x, y, height, and width of the
        bounding box.
    """

    x1, y1, x2, y2 = bounding_box
    x = min(x1, x2)
    y = min(y1, y2)
    width = x2 - x1
    height = y2 - y1

    return [x, y, width, height]

def create_image_url(filename):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    return f'http://localhost:8080/{filename}'



def extracted_tables_to_label_studio_json_file_with_paddleOCR(images_folder_path):
    label_studio_task_list = []
    for images in os.listdir(images_folder_path):
        if images.endswith('.png'):
            output_json = {}
            annotation_result = []

            print(images)

            output_json['data'] =  {"ocr":create_image_url(images)}

                    
            img = Image.open(f'D:/Projects/AI_Projects/NLP/Document_AI/LayoutLM_Models/image/{images}')

            img = np.asarray(img)
            image_height, image_width = img.shape[:2]


            result = ocr.ocr(img,cls=False)

            #print(result)

            for output in result:

                for item in output: 
                    co_ord = item[0]
                    text = item[1][0]

                    four_co_ord = [co_ord[0][0],co_ord[1][1],co_ord[2][0]-co_ord[0][0],co_ord[2][1]-co_ord[1][1]]

                    #print(four_co_ord)
                    #print(text)

                    bbox = {
                    'x': 100 * four_co_ord[0] / image_width,
                    'y': 100 * four_co_ord[1] / image_height,
                    'width': 100 * four_co_ord[2] / image_width,
                    'height': 100 * four_co_ord[3] / image_height,
                    'rotation': 0
                            }
                    

                    if not text:
                        continue
                    region_id = str(uuid4())[:10]
                    score = 0.5
                    bbox_result = {
                        'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
                        'value': bbox}
                    transcription_result = {
                        'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
                        'value': dict(text=[text], **bbox), 'score': score}
                    annotation_result.extend([bbox_result, transcription_result])
                    #print('annotation_result :\n',annotation_result)
            output_json['predictions'] = [ {"result": annotation_result,  "score":0.97}]

            label_studio_task_list.append(output_json)
        

    # saving label_stdui_task_list as json file to import in label_studio
    with open('TC_label-studio_input_file.json', 'w') as f:
        json.dump(label_studio_task_list, f, indent=4)


extracted_tables_to_label_studio_json_file_with_paddleOCR(images_folder_path)
#   