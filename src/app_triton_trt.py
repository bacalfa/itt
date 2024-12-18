from flask import Flask, render_template, jsonify, request
import model_manager_triton
import io
import base64
from PIL import Image
import numpy as np
import logging
# import tritonclient.http as tritonclient
import tritonclient.grpc as tritonclient

# from functools import partial

app = Flask(__name__)
logger = app.logger
logger.setLevel(logging.DEBUG)


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
    model_manager_triton.ModelManager.create_model_repository()
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
    logger.debug("Inside infer")
    data = request.get_json()
    image = np.array(read_base64_image(data["img"])) / 255.0
    logger.debug(f"Image shape {image.shape}")

    # Prepare input data for Triton
    inputs = [
        tritonclient.InferInput("image", image.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(image.astype(np.float32))
    if data["question"] is not None:
        question_array = np.array([data["question"].encode("UTF-8")], dtype=np.bytes_)
        logger.debug(f"Question shape {question_array.shape}")
        inputs.append(tritonclient.InferInput("question", question_array.shape, "BYTES"))
        inputs[-1].set_data_from_numpy(question_array)

    # # Define callback for asynchronous inference
    # def callback(user_data, result, error):
    #     if error:
    #         user_data.append(error)
    #     else:
    #         user_data.append(result)

    # Send request to Triton server
    model_name = model_manager_triton.ITTModelName.GIT.name.lower()
    if data["question"] is not None:
        model_name = model_manager_triton.ITTModelName.GITVQA.name.lower()
    # url = "triton:8000"
    url = "triton:8001"
    model_version = "1"
    output_name = "generated_caption"
    output = tritonclient.InferRequestedOutput(output_name)
    # response = []
    with tritonclient.InferenceServerClient(url, verbose=False) as client:
        logger.debug("Initialized Triton client")
        # client.async_infer(model_name, model_version=model_version, inputs=inputs, outputs=[output],
        #                    callback=partial(callback, response))
        response = client.infer(model_name, model_version=model_version, inputs=inputs, outputs=[output])
        logger.debug("Executed inference")
        logger.debug(f"Results {response.get_response()}")

    # Process response from Triton
    generated_caption = response.as_numpy(output_name)[0].decode("UTF-8")
    if data["question"] is not None:
        generated_caption = extract_answer_from_output(data["question"], generated_caption)
    logger.debug(f"Extracted output {generated_caption}")
    return jsonify(result=generated_caption)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
