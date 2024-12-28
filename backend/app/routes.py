from flask import Blueprint, jsonify, request
import os

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
        file.save(os.path.join('uploads', file.filename))
        return jsonify({'message': 'File uploaded successfully'})
    return jsonify({'error': 'Invalid file type'})



