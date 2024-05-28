from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import shutil
import uuid
import cv2

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('No selected files')
            return redirect(request.url)
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_files.append(filename)
            else:
                flash('Invalid file format for one or more files. Allowed formats: png, jpg, jpeg')
                return redirect(request.url)
        
        flash('Files uploaded successfully')
        return redirect(url_for('run_yolo', filenames=",".join(uploaded_files)))
    return redirect(request.url)

@app.route('/run_yolo/<filenames>')
def run_yolo(filenames):
    try:
        model = YOLO('model.pt')
        files = filenames.split(',')

        results = []
        for filename in files:
            source_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            unique_id = str(uuid.uuid4())
            result_dir = os.path.join('runs', 'detect', unique_id)
            
            # Create the directory if it does not exist
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # Load and resize the original image
            img = cv2.imread(source_file_path)
            img = cv2.resize(img, (640, 480))  # Resize to desired dimensions

            predictions = model.predict(source=img, conf=0.5)

            # Draw bounding boxes manually with class labels
            for pred in predictions:
                for detection in pred.boxes:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    cls = int(detection.cls[0])
                    label = f"{model.names[cls]}"
                    
                    # Calculate the center of the bounding box
                    bb_center_x = (x1 + x2) // 2
                    bb_center_y = (y1 + y2) // 2
                    
                    # Calculate the text size based on the bounding box size
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    font_scale = min((x2 - x1) / text_size[0], (y2 - y1) / text_size[1]) * 0.8
                    
                    # Adjust font size if it's too large
                    if font_scale > 1:
                        font_scale = 1
                    
                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)  # Black bounding box
                    
                    # Calculate text position
                    text_x = bb_center_x - text_size[0] // 2
                    text_y = bb_center_y + text_size[1] // 2
                    
                    # Draw the label
                    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)  # Black text with anti-aliasing



            # Save the modified image
            unique_filename = f"{unique_id}_{filename}"
            destination_path = os.path.join('static', unique_filename)
            cv2.imwrite(destination_path, img)
            results.append(unique_filename)
        
        return render_template('result.html', filenames=results)

    except Exception as e:
        flash(f'Error processing files: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
