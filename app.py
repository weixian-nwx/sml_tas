# app.py
import traceback
import os
import io
import time
import numpy as np
import torch
import tensorflow as tf
from flask import (
    Flask, request, render_template,
    redirect, url_for, send_from_directory,
    flash, jsonify
)
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from flask_socketio import SocketIO, emit

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
    "Hayao":     "checkpoint/generator_Hayao_weight",
    "Paprika":   "checkpoint/generator_Paprika_weight",
    "Shinkai":   "checkpoint/generator_Shinkai_weight",
}

# Initialize SocketIO
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# PyTorch model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator()
# checkpoint_path = (
#     r"C:\Users\shuti\OneDrive\Documents\Term 8 Modules\40.319 Statistical and Machine Learning\sml_tas\weights\paprika\generator.pth"
# )
# checkpoint_path = os.path.join("weights", "generator.pth")
checkpoint_path = "weights/paprika/generator.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# def generate_image(input_image_path, style):
#     checkpoint_dir = STYLE_CHECKPOINTS.get(style)
#     img_size = [256, 256]
#     sample_image = load_test_data(input_image_path, img_size)
#     if sample_image.ndim == 3:
#         sample_image = np.expand_dims(sample_image, axis=0)

#     tf.reset_default_graph()
#     test_real = tf.placeholder(
#         tf.float32, [1, None, None, 3], name="test_real"
#     )
#     with tf.variable_scope("generator", reuse=False):
#         test_generated = generator.G_net(test_real).fake

#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
#         output = sess.run(test_generated, feed_dict={test_real: sample_image})

#     timestamp = int(time.time())
#     output_filename = f"result_{style}_{timestamp}.jpg"
#     output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
#     save_images(output, output_path, None)
#     return output_filename



def generate_image(input_image_path, style):
    try:
        checkpoint_dir = STYLE_CHECKPOINTS.get(style)
        if not checkpoint_dir:
            raise ValueError(f"Invalid style selected: {style}")

        img_size = [256, 256]
        sample_image = load_test_data(input_image_path, img_size)
        if sample_image.ndim == 3:
            sample_image = np.expand_dims(sample_image, axis=0)

        tf.compat.v1.reset_default_graph()
        test_real = tf.compat.v1.placeholder(
            tf.float32, [1, None, None, 3], name="test_real"
        )

        with tf.compat.v1.variable_scope("generator", reuse=False):
            test_generated = generator.G_net(test_real).fake

        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            output = sess.run(test_generated, feed_dict={test_real: sample_image})

        timestamp = int(time.time())
        output_filename = f"result_{style}_{timestamp}.jpg"
        output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
        save_images(output, output_path, None)
        return output_filename

    except Exception as e:
        print("[ERROR] Failed in generate_image:", e)
        raise



@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/generate", methods=["POST"])
# def generate_route():
#     # 1) Check file
#     if "file" not in request.files:
#         err = "No file part in the request"
#         if "application/json" in request.headers.get("Accept", ""):
#             return jsonify(error=err), 400
#         flash(err)
#         return redirect(request.url)

#     file = request.files["file"]
#     if file.filename == "":
#         err = "No file selected"
#         if "application/json" in request.headers.get("Accept", ""):
#             return jsonify(error=err), 400
#         flash(err)
#         return redirect(request.url)

#     if not allowed_file(file.filename):
#         err = "Allowed file types are png, jpg, jpeg, gif"
#         if "application/json" in request.headers.get("Accept", ""):
#             return jsonify(error=err), 400
#         flash(err)
#         return redirect(request.url)

#     # 2) Save file
#     filename = secure_filename(file.filename)
#     upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     file.save(upload_path)

#     # 3) Generate
#     style = request.form.get("style")
#     try:
#         result_filename = generate_image(upload_path, style)
#     except Exception as e:
#         err = f"Error during generation: {e}"
#         if "application/json" in request.headers.get("Accept", ""):
#             return jsonify(error=err), 500
#         flash(err)
#         return redirect(url_for("index"))

#     orig_url = url_for("static", filename=f"uploads/{filename}")
#     gen_url  = url_for("static", filename=f"results/{result_filename}")

#     # 4) JSON response for AJAX
#     if "application/json" in request.headers.get("Accept", ""):
#         return jsonify(orig_url=orig_url, gen_url=gen_url)

#     # 5) Fallback full render
#     return render_template(
#         "index.html",
#         original_image=orig_url,
#         generated_image=gen_url,
#         style=style
#     )


@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        # 1) File checks
        if "file" not in request.files:
            return jsonify(error="No file part in the request"), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify(error="No file selected"), 400

        if not allowed_file(file.filename):
            return jsonify(error="Allowed file types are png, jpg, jpeg, gif"), 400

        # 2) Save upload
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        # 3) Style selection
        style = request.form.get("style")
        if not style:
            return jsonify(error="Style not selected"), 400

        # 4) Try generating image
        result_filename = generate_image(upload_path, style)

        # 5) Return result
        orig_url = url_for("static", filename=f"uploads/{filename}")
        gen_url = url_for("static", filename=f"results/{result_filename}")
        return jsonify(orig_url=orig_url, gen_url=gen_url, style=style)

    except Exception as e:
        # Show full error logs in Render
        traceback.print_exc()
        return jsonify(error=str(e)), 500





@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        app.config["RESULT_FOLDER"],
        filename,
        as_attachment=True
    )

# if __name__ == "__main__":
#     socketio.run(app, host="0.0.0.0", port=5000)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
#     socketio.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    import eventlet
    import eventlet.wsgi
    port = int(os.environ.get("PORT", 5000))
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", port)), app)
