from flask import Flask, render_template, jsonify, request
import model_manager
import io
import base64
from PIL import Image

app = Flask(__name__)


def read_base64_image(base64_string: str) -> Image.Image:
    """Decodes a base64 encoded image string and returns a PIL Image object."""

    # Remove the data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object from the decoded data
    image_bytes = io.BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_bytes)

    return image


@app.route("/")
def index():
    return render_template("index.html")


def extract_answer_from_output(question: str, output: str) -> str:
    output = output.replace(" ' ", "'")  # Remove extra spaces around apostrophe
    if output.lower().startswith(question.lower()):
        output = output[len(question):].strip()
    else:
        output = output.strip()

    return output


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    mm = model_manager.ModelManager()
    if data["question"] is not None:
        mm.ModelName = model_manager.ITTModelName.GITVQA
    generated_caption = mm.infer(img=read_base64_image(data["img"]), question=data["question"])
    if data["question"] is not None:
        generated_caption = extract_answer_from_output(data["question"], generated_caption)
    return jsonify(result=generated_caption)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
