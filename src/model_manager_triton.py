import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from typing import Optional, Callable, Union
import numpy as np
import PIL
import enum
import triton_config

device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ITTModelName(enum.StrEnum):
    GIT = "microsoft/git-base-coco"
    GITVQA = "microsoft/git-base-textvqa"
    ViTGPT2 = "nlpconnect/vit-gpt2-image-captioning"


class ModelManager:
    def __init__(self, model_name: str = ITTModelName.GIT):
        self._model_name: str = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        if self._model_name == ITTModelName.GIT:
            self._model = GIT()
        elif self._model_name == ITTModelName.GITVQA:
            self._model = GIT(with_question=True)
        elif self._model_name == ITTModelName.ViTGPT2:
            self._model = ViTGPT2()

    @property
    def ModelName(self) -> str:
        return self._model_name

    @ModelName.setter
    def ModelName(self, model_name: str):
        self._model_name = model_name
        self._init_model()

    def infer(self, img: Union[np.ndarray, PIL.Image.Image], question: str = None) -> str:
        return self._model.infer(img, question)

    @staticmethod
    def create_model_repository():
        for itt_model in ITTModelName:
            has_question = False
            if itt_model == ITTModelName.GIT:
                model = GIT()
            elif itt_model == ITTModelName.GITVQA:
                model = GIT(with_question=True)
                has_question = True
            elif itt_model == ITTModelName.ViTGPT2:
                model = ViTGPT2()

            height = model.processor.current_processor.crop_size["height"]
            width = model.processor.current_processor.crop_size["width"]
            dummy_input = torch.rand(3, height, width)
            pixel_values = model.processor(images=dummy_input, return_tensors="pt").pixel_values.to(device)
            if has_question:
                question = "What does the image show?"
                input_ids = model.processor(text=question, add_special_tokens=False).input_ids
                input_ids = [model.processor.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0)

                torch.onnx.export(model.model, (pixel_values, input_ids),
                                  f'model_repository/{itt_model.name}-onnx-model/1/model.onnx', verbose=False,
                                  input_names=["input0", "input1"], output_names=["output0"],
                                  dynamic_axes={"input0": {0: "batch_size"}, "input1": {0: "batch_size"},
                                                "output0": {0: "batch_size"}})
            else:
                torch.onnx.export(model.model, pixel_values,
                                  f'model_repository/{itt_model.name}-onnx-model/1/model.onnx', verbose=False,
                                  input_names=["input0"], output_names=["output0"],
                                  dynamic_axes={"input0": {0: "batch_size"}, "output0": {0: "batch_size"}})

            configuration = triton_config.TritonConfig(
                name=f"{itt_model.name}-onnx-model",
                platform="onnxruntime_onnx",
                max_batch_size=32,
                inputs=[
                    {
                        "name": '"input0"',
                        "data_type": "TYPE_FP32",
                        "format": "FORMAT_NCHW",
                        "dims": f"[ 3, {height}, {width} ]",
                    }
                ],
                outputs=[
                    {
                        "name": '"output0"',
                        "data_type": "TYPE_FP32",
                        "dims": f"[ {model.model.output.out_features} ]",
                    }
                ],
            )

            if has_question:
                configuration.inputs.append(
                    {
                        "name": '"input1"',
                        "data_type": "TYPE_INT32",
                        "format": "FORMAT_NCHW",
                        "dims": f"[ 3, {height}, {width} ]",
                    }
                )

            with open(f"model_repository/{itt_model.name}-onnx-model/config.pbtxt", "wt") as file:
                file.write(str(configuration))
