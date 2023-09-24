import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import numpy as np
from engine import *
from trainer import *
from loader import *
from utils import *
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForTokenClassification



featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokeniser = LayoutLMv3TokenizerFast.from_pretrained("D:/Projects/AI_Projects/NLP/Document_AI/LayoutLM_Models/inputs/layoutlmv3Microsoft",ignore_mismatched_sizes=True)
processor = LayoutLMv3Processor(tokenizer=tokeniser,feature_extractor=featur_extractor)


# Load the image
image = Image.open("D:/Projects/AI_Projects/NLP/Document_AI/LayoutLM_Models/image/Training_Images/92094751.png")
image.show()
test_dict, width_scale, height_scale = dataSetFormat(image)

print("test_dict['bboxes'] :",test_dict['bboxes'])


model = ModelModule(4)
encoding = processor(
                        test_dict['img_path'].convert('RGB'),
                        test_dict['tokens'],
                        boxes= test_dict['bboxes'],

                        max_length=256,
                        padding="max_length", truncation=True, return_tensors='pt',
                        return_offsets_mapping=True
                    )

print(" encoding ",encoding['bbox'])

model.load_state_dict(torch.load("D:/Projects/AI_Projects/NLP/Document_AI/LayoutLM_Models/src/model_20.bin"))


inputs_ids = torch.tensor(encoding['input_ids'], dtype=torch.int64).flatten()
attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.int64).flatten()
bbox = torch.tensor(encoding['bbox'], dtype=torch.int64).flatten(end_dim=1)
pixel_values = torch.tensor(encoding['pixel_values'], dtype=torch.float32).flatten(end_dim=1)

print("bbox :",bbox)


with torch.no_grad():
    op = model(input_ids = inputs_ids.unsqueeze(0), 
               attention_mask=attention_mask.unsqueeze(0),
               bbox=bbox.unsqueeze(0),
               pixel_values = pixel_values.unsqueeze(0)
               )
    predictions = op.argmax(-1).squeeze().tolist()

    prob = nnf.softmax(op, dim=1)
    txt = prob.squeeze().numpy()/np.sum(prob.squeeze().numpy(), axis=1).reshape(-1,1)
    output_prob = np.max(txt, axis=1)

     
pred = torch.tensor(predictions)
offset_mapping = encoding['offset_mapping']
is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
true_predictions = torch.tensor(np.array([pred.item() for idx, pred in enumerate(pred) if not is_subword[idx]]))

true_prob = torch.tensor(np.array([output_prob.item() for idx, output_prob in enumerate(output_prob) if not is_subword[idx]]))

true_boxes = torch.tensor([box.tolist() for idx, box in enumerate(bbox) if not is_subword[idx]])

concat_torch = torch.column_stack((true_boxes, true_predictions, true_prob))


one_class = concat_torch[torch.where((concat_torch[:,4]==1) & (concat_torch[:,3]==0)  & (concat_torch[:,2]==0))]
two_class = concat_torch[torch.where((concat_torch[:,4]==2) & (concat_torch[:,3]==0)  & (concat_torch[:,2]==0))]
three_class = concat_torch[torch.where((concat_torch[:,4]==3) & (concat_torch[:,3]==0)  & (concat_torch[:,2]==0))]
four_class = concat_torch[torch.where((concat_torch[:,4]==4) & (concat_torch[:,3]==0)  & (concat_torch[:,2]==0))]


finl = torch.row_stack((one_class, two_class, three_class, four_class))
unique_ = torch.unique(finl, dim=0)

plot_img(test_dict['img_path'], unique_[:, :4] ,unique_[:, 4].tolist(), unique_[:, 5].tolist(), width_scale, height_scale)


print(unique_)