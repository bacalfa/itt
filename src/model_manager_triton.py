import torch
import model_manager
from typing import Union
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
    @staticmethod
    def create_model_repository():
        #TODO: This approach does not work because converting these generative models to ONNX engine doesn't seem to be
        # supported. ONNX model is needed before converting to the final TensorRT engine.
        for itt_model in ITTModelName:
            has_question = False
            if itt_model == ITTModelName.GIT:
                model = model_manager.GIT()
            elif itt_model == ITTModelName.GITVQA:
                model = model_manager.GIT(with_question=True)
                has_question = True
            elif itt_model == ITTModelName.ViTGPT2:
                model = model_manager.ViTGPT2()

            from transformers import AutoModelForVision2Seq
            model.model = AutoModelForVision2Seq.from_pretrained(ITTModelName.GIT).to(device)

            height = model.processor.current_processor.crop_size["height"]
            width = model.processor.current_processor.crop_size["width"]
            dummy_input = torch.rand(1, 3, height, width)
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
