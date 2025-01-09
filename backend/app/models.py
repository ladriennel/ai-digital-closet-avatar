import torch
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import uuid

class ClothingClassifier:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
        self.model.eval()

        self.detector = YOLO("yolov8n.pt")

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
                
                # cropped_img = cv2_image[y1:y2, x1:x2]
                
                cropped_img = self.add_Buffer(cv2_image, cv2_image[y1:y2, x1:x2], (x1,y1,x2,y2))

                rgba = self.remove_background_grabcut(cropped_img)

                pil_crop = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

                classification = self._classify_item(pil_crop)
                
                if classification:
                    item_uuid = str(uuid.uuid4())
                    filename = f"{item_uuid}.png"
                    save_path = os.path.join('uploads/crops', filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, rgba)
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
                    outputs = self.model(**inputs)
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
                    style_outputs = self.model(**style_inputs)
                    style_probs = style_outputs.logits_per_image.softmax(dim=1)[0]
                    style_idx = torch.argmax(style_probs).item()
                    
                    final_classification['style'] = {
                        'name': self.categories[best_category]['styles'][style_idx],
                        'confidence': float(style_probs[style_idx])
                    }
            return final_classification if best_confidence > 0.25 else None
    
    def add_Buffer(self, orginal_image, crop_image, crop_box):
       
        original_height, original_width, _ = orginal_image.shape
        crop_height, crop_width, _  = crop_image.shape
        left, upper, right, lower = crop_box

        buffer_width = int(crop_width * .2)
        buffer_height = int(crop_height * .2)

        left = max(0, left - buffer_width)
        upper = max(0, upper - buffer_height)
        right = min(original_width, right + buffer_width)
        lower = min(original_height, lower + buffer_height)

        return orginal_image[upper:lower, left:right]
    
    def remove_background_grabcut(self, image):
        # Create a mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create rectangle for initial guess
        # Make it slightly smaller than the image to avoid edge artifacts
        rect = (5, 5, width-5, height-5)
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        
        # Run grabcut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where sure and likely foreground are 1, rest is 0
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        
        # Convert to RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        rgba[:, :, 3] = mask2 * 255
        
        return rgba




        
classifier = ClothingClassifier()

        
        

    


        

    

    
