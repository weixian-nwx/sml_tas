# Refactored Flask Application for Cartoon Style Transfer

import os
import io
import time
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, send_file
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image

# Import custom utilities and models
from tools.utils import load_test_data, save_images, check_folder
from net import generator
from model import Generator

# Disable eager execution for TensorFlow
tf.compat.v1.disable_eager_execution()

# Flask app initialization
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

# PyTorch model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator()
checkpoint_path = r"C:\Users\shuti\OneDrive\Documents\Term 8 Modules\40.319 Statistical and Machine Learning\sml_tas\weights\paprika\generator.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()


# Helper functions

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)


def postprocess_image(tensor):
    tensor = (tensor.squeeze(0).detach().cpu() + 1) / 2
    transform = transforms.ToPILImage()
    return transform(tensor)


def generate_image(input_image_path, style):
    checkpoint_dir = STYLE_CHECKPOINTS.get(style)
    img_size = [256, 256]
    sample_image = load_test_data(input_image_path, img_size)
    if len(sample_image.shape) == 3:
        sample_image = np.expand_dims(sample_image, axis=0)
    tf.reset_default_graph()
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name="test_real")
    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
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
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)
        style = request.form.get("style")
        try:
            result_filename = generate_image(upload_path, style)
            return render_template("results.html", original_image=url_for("static", filename=f"uploads/{filename}"), generated_image=url_for("static", filename=f"results/{result_filename}"))
        except Exception as e:
            flash(str(e))
            return redirect(url_for("index"))
    flash("Allowed file types are png, jpg, jpeg, gif")
    return redirect(request.url)


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
