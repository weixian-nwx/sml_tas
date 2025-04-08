<<<<<<< HEAD
# app.py
import os
import time
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tools.utils import load_test_data, save_images, check_folder
from net import generator

# Ensure the templates folder is set correctly.
app = Flask(__name__, template_folder="templates")
app.secret_key = "secret_key_for_session"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
check_folder(UPLOAD_FOLDER)
check_folder(RESULT_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

STYLE_CHECKPOINTS = {
    "Hayao": "checkpoint/generator_Hayao_weight",
    "Paprika": "checkpoint/generator_Paprika_weight",
    "Shinkai": "checkpoint/generator_Shinkai_weight",
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_image(input_image_path, style):
    checkpoint_dir = STYLE_CHECKPOINTS.get(style)
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory for style {style} not found.")
    img_size = [256, 256]
    sample_image = load_test_data(input_image_path, img_size)
    if len(sample_image.shape) == 3:
        sample_image = np.expand_dims(sample_image, axis=0)
    tf.reset_default_graph()
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name="test_real")
    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, ckpt_name)
            print(" [*] Model restored from {}".format(ckpt_name))
        else:
            raise ValueError("Checkpoint not found for style: " + style)
        output = sess.run(test_generated, feed_dict={test_real: sample_image})
    timestamp = int(time.time())
    output_filename = f"result_{style}_{timestamp}.jpg"
    output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
    save_images(output, output_path, None)
    return output_filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_route():
    if "file" not in request.files:
        flash("No file part in the request")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)
    else:
        flash("Allowed file types are png, jpg, jpeg, gif")
        return redirect(request.url)
    style = request.form.get("style")
    if style not in STYLE_CHECKPOINTS:
        flash("Invalid style selected.")
        return redirect(request.url)
    try:
        result_filename = generate_image(upload_path, style)
    except Exception as e:
        flash(str(e))
        return redirect(url_for("index"))
    return render_template(
        "results.html",
        original_image=url_for("static", filename=f"uploads/{filename}"),
        generated_image=url_for("static", filename=f"results/{result_filename}"),
        result_filename=result_filename,
    )

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    # Debug: print the absolute path of the templates folder
    print("Templates Folder:", os.path.join(os.getcwd(), "templates"))
=======
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
>>>>>>> 068bf002f72ce88e46c2ac5f948ef6aaaa5de2f4
    app.run(debug=True)
