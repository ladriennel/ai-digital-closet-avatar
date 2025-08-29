from flask import Blueprint, jsonify, request
import os
from PIL import Image
from .models import classifier
from .database import add_clothing_items, get_all_clothes, get_clothes_by_category


bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return jsonify({'message': 'Welcome to the digi closet api'})


ALLOWED_EXTENSIONS  = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        return True
    return False

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        # check file res
        file.seek(0)  # Reset file pointer to beginning
        img = Image.open(file)
        width, height = img.size
        if width < 512 and height < 512:
            return jsonify({'error': 'Bad Resolution'})
        try:
            detected_items = classifier.process_image(img)
            saved_items = add_clothing_items(detected_items)

            return jsonify({
            'success': True,
            'message': 'Items saved successfully',
            'items': saved_items
            })
        
        except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'})            

    return jsonify({'error': 'Invalid file type'})








