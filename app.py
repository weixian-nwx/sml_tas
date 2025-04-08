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
    app.run(debug=True)
