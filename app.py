import io
from flask import Flask, request, send_file, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load pretrained model once
MODEL = MobileNetV2(weights="imagenet")


# Utilities

def load_image_bytes_to_tensor(file_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32")
    tensor = preprocess_input(arr)          # scales to [-1, 1]
    tensor = tf.expand_dims(tensor, 0)      # shape (1, H, W, C)
    return tensor, img

def deprocess_and_bytes(adv_tensor):
    # adv_tensor expected in model input range [-1, 1]
    arr = adv_tensor[0].numpy()
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype("uint8")
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio

def predict_label_onehot(model, image_tensor):
    logits = model(image_tensor, training=False)
    probs = tf.nn.softmax(logits)
    class_idx = tf.argmax(probs[0])
    one_hot = tf.one_hot(class_idx, logits.shape[-1])
    one_hot = tf.reshape(one_hot, (1, -1))
    return one_hot, int(class_idx.numpy())

def fgsm_attack(model, original_tensor, epsilon):
    orig = tf.identity(original_tensor)
    one_hot, _ = predict_label_onehot(model, orig)
    orig_var = tf.Variable(orig)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(orig_var)
        logits = model(orig_var, training=False)
        loss = loss_fn(one_hot, tf.nn.softmax(logits))
    gradient = tape.gradient(loss, orig_var)
    adv = orig + epsilon * tf.sign(gradient)
    adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


# Web interface

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    uploaded = request.files.get("image")
    if not uploaded:
        return "No file uploaded", 400

    # parse epsilon; fallback to 0.05
    try:
        epsilon = float(request.form.get("epsilon", "0.05"))
    except:
        epsilon = 0.05

    image_bytes = uploaded.read()
    tensor, _ = load_image_bytes_to_tensor(image_bytes)
    adv_tensor = fgsm_attack(MODEL, tensor, epsilon)
    out_bio = deprocess_and_bytes(adv_tensor)
    filename = f"adv_eps{epsilon}.png"
    return send_file(out_bio, mimetype="image/png", as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
