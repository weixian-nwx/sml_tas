import os
import io
import time
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from flask_socketio import SocketIO, emit  # <-- Import SocketIO

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

# Initialize SocketIO (using threading mode for compatibility on Windows)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# PyTorch model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator()
checkpoint_path = r"C:\Users\shuti\OneDrive\Documents\Term 8 Modules\40.319 Statistical and Machine Learning\sml_tas\weights\paprika\generator.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Helper function to check allowed file extensions
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
    # On page load, you could either load the gallery or let the JS do it via an API call.
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_route():
    # 1) Make sure there's a file in the POST
    if "file" not in request.files:
        flash("No file part in the request")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    # 2) Check extension
    if not allowed_file(file.filename):
        flash("Allowed file types are png, jpg, jpeg, gif")
        return redirect(request.url)

    # 3) Save the upload
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    # 4) Run your generator
    style = request.form.get("style")
    try:
        result_filename = generate_image(upload_path, style)
    except Exception as e:
        flash(f"Error during generation: {e}")
        return redirect(url_for("index"))

    # 5) Build the URLs
    orig_url = url_for("static", filename=f"uploads/{filename}")
    gen_url  = url_for("static", filename=f"results/{result_filename}")

    # 6) Render index.html with both images
    return render_template(
        "index.html",
        original_image=orig_url,
        generated_image=gen_url,
        style=style,
        # optionally: gallery_images=[]
    )



# @app.route("/generate", methods=["POST"])
# def generate_route():
#     if "file" not in request.files:
#         flash("No file part in the request")
#         return redirect(request.url)
#     file = request.files["file"]
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(upload_path)
#         style = request.form.get("style")
#         try:
#             result_filename = generate_image(upload_path, style)
#             # Emit a SocketIO event to notify connected clients of the new image.
#             socketio.emit("new_image", {"filename": result_filename})
#             return render_template("results.html", 
#                                    original_image=url_for("static", filename=f"uploads/{filename}"),
#                                    generated_image=url_for("static", filename=f"results/{result_filename}"))
#         except Exception as e:
#             flash(str(e))
#             return redirect(url_for("index"))
#     flash("Allowed file types are png, jpg, jpeg, gif")
#     return redirect(request.url)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

# Run the app with SocketIO
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)


















