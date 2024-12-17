import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline
import PIL.Image
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.DEBUG)
h2 = logging.StreamHandler(sys.stderr)
h2.setLevel(logging.DEBUG)
logger.addHandler(h1)
logger.addHandler(h2)


class TritonPythonModel:
    def initialize(self, args):
        self.generator = pipeline("visual-question-answering", model="microsoft/git-base-textvqa")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Convert Tensor to PIL Image
            image = pb_utils.get_input_tensor_by_name(request, "image")
            logger.debug(f"1 image type {type(image)}")
            image = image.as_numpy() * 255.0
            logger.debug(f"2 image type {type(image)}")
            logger.debug(f"2 image shape {image.shape}")
            image = PIL.Image.fromarray(image.astype(np.uint8))
            logger.debug(f"3 image type {type(image)}")
            logger.debug(f"3 image size {image.size}")

            # Decode the Byte Tensor into Text
            question = pb_utils.get_input_tensor_by_name(request, "question")
            logger.debug(f"4 question type {type(question)}")
            question = question.as_numpy()
            logger.debug(f"5 question type {type(question)}")
            logger.debug(f"5 question shape {question.shape}")
            question = question[0].decode("UTF-8")
            logger.debug(f"6 question type {type(question)}")
            logger.debug(f"6 question length {len(question)}")

            # Call the Model pipeline
            pipeline_output = self.generator(image, question)
            generated_txt = pipeline_output[0]["answer"]
            output = generated_txt
            logger.debug(f"7 output {output}")
            logger.debug(f"7 output.encode() {output.encode()}")

            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_caption",
                        np.array([output.encode("UTF-8")], dtype=np.bytes_),
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self, args):
        self.generator = None
