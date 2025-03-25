import os
import io
from flask import Flask, request, render_template, send_file, redirect, flash
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image

# Import your Generator model from model.py
from model import Generator

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Create a folder for uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Set device and load the model checkpoint (update the checkpoint path as needed)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Generator()
# checkpoint_path = os.path.join('path_to_checkpoint', 'generator.pth')  # Replace with your checkpoint path
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.to(device)
# model.eval()


# Set device and load the model checkpoint with the actual path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator()

# Replace with your actual checkpoint file path (using a raw string)
checkpoint_path = r"C:\Users\shuti\OneDrive\Documents\Term 8 Modules\40.319 Statistical and Machine Learning\sml_tas\animegan2-pytorch\weights\paprika\generator.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()



def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    tensor = (tensor.squeeze(0).detach().cpu() + 1) / 2  # Convert from [-1,1] to [0,1]
    transform = transforms.ToPILImage()
    return transform(tensor)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Open and preprocess the uploaded image
            input_image = Image.open(input_path).convert('RGB')
            processed_input = preprocess_image(input_image)

            # Generate the output image using the model
            with torch.no_grad():
                output_tensor = model(processed_input)
            output_image = postprocess_image(output_tensor)

            # Save output image to a bytes buffer for download
            img_io = io.BytesIO()
            output_image.save(img_io, 'JPEG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='output.jpg')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
