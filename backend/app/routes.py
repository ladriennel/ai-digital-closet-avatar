from flask import Blueprint, jsonify, request
import os
import PIL

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
        img = PIL.Image.open(file)
        width, height = img.size
        if width >= 512 and height >= 512:
            file.save(os.path.join('uploads', file.filename))
            return jsonify({'message': 'File uploaded successfully'})
        else:
            return jsonify({'error': ' image resolution has to be greater than 512x512'})
    return jsonify({'error': 'Invalid file type'})




