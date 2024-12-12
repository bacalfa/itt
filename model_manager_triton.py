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
            if itt_model == ITTModelName.GIT:
                model = GIT()
            elif itt_model == ITTModelName.GITVQA:
                model = GIT(with_question=True)
            elif itt_model == ITTModelName.ViTGPT2:
                model = ViTGPT2()

            height = model.processor.current_processor.crop_size["height"]
            width = model.processor.current_processor.crop_size["width"]
            dummy_input = torch.randn(1, 3, height, width).to(device)
            torch.onnx.export(model.model, dummy_input,
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

            with open(f"model_repository/{itt_model.name}-onnx-model/config.pbtxt", "wt") as file:
                file.write(str(configuration))


class GIT:
    def __init__(self, with_question: bool = False):
        model_name = ITTModelName.GIT if not with_question else ITTModelName.GITVQA
        self.model: Optional[AutoModelForCausalLM] & Callable = AutoModelForCausalLM.from_pretrained(model_name).to(
            device)
        self.processor: Optional[AutoProcessor] & Callable = AutoProcessor.from_pretrained(model_name)

    def infer(self, img: Union[np.ndarray, PIL.Image.Image], question: str = None) -> str:
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(device)
        input_ids = None
        if question is not None:
            input_ids = self.processor(text=question, add_special_tokens=False).input_ids
            input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_caption[0]


class ViTGPT2:
    def __init__(self):
        model_name = ITTModelName.ViTGPT2
        self.model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def infer(self, img: Union[np.ndarray, PIL.Image.Image], question: str = None) -> str:
        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

        generated_ids = self.model.generate(pixel_values=pixel_values)
        generated_caption = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_caption[0]
