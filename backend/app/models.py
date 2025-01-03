import torch
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

class ClothingClassifier:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
        self.model.eval()

        self.detector = YOLO("yolov8.pt")

        self.categories = {
            'tops': {
                'types': [
                    "t-shirt", "blouse", "sweater", "tank top", "hoodie", 
                    "button-up shirt", "crop top", "turtleneck"
                ],
                'styles': [
                    "casual", "formal", "streetwear", "vintage", "basic",
                    "y2k", "minimalist", "designer"
                ]
            },
            'bottoms': {
                'types': [
                    "jeans", "trousers", "skirt", "shorts", "leggings",
                    "cargo pants", "mini skirt", "maxi skirt"
                ],
                'styles': [
                    "casual", "formal", "streetwear", "vintage", "basic",
                    "y2k", "minimalist", "designer"
                ]
            },
            'dresses': {
                'types': [
                    "mini dress", "maxi dress", "midi dress", "sundress",
                    "slip dress", "shirt dress"
                ],
                'styles': [
                    "casual", "formal", "party", "vintage", "elegant",
                    "minimalist", "designer", "basic"
                ]
            },
            'outerwear': {
                'types': [
                    "jacket", "coat", "blazer", "cardigan", "denim jacket",
                    "leather jacket", "puffer jacket"
                ],
                'styles': [
                    "casual", "formal", "streetwear", "vintage", "basic",
                    "y2k", "minimalist", "designer"
                ]
            }
        }

    def process_image(self, image): # image is coming from routes
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 

        results = self.detector(cv2_image)
        detected_items = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cropped_img = cv2_image[y1:y2, x1:x2]
                
                pil_crop = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                
                classification = self._classify_item(pil_crop)
                
                if classification:
                    save_path = os.path.join('uploads/crops', f"{len(detected_items)}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, cropped_img)
                    
                    detected_items.append({
                        'classification': classification,
                        'crop_path': save_path
                    })

        return detected_items


    def _classify_item(self, image: Image.Image):
            best_category = None
            best_confidence = 0
            final_classification = {}

            for category, labels in self.categories.items():
                inputs = self.processor(
                    images=image,
                    text=labels['types'],
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]
                    confidence = torch.max(probs).item()
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_category = category
                        type_idx = torch.argmax(probs).item()
                        final_classification = {
                            'category': category,
                            'type': {
                                'name': labels['types'][type_idx],
                                'confidence': float(confidence)
                            }
                        }

            if best_category:
                style_inputs = self.processor(
                    images=image,
                    text=self.categories[best_category]['styles'],
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    style_outputs = self.model(style_inputs)
                    style_probs = style_outputs.logits_per_image.softmax(dim=1)[0]
                    style_idx = torch.argmax(style_probs).item()
                    
                    final_classification['style'] = {
                        'name': self.categories[best_category]['styles'][style_idx],
                        'confidence': float(style_probs[style_idx])
                    }

            return final_classification if best_confidence > 0.3 else None
    


        
        

    


        

    

    








