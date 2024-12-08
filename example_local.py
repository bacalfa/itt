from PIL import Image
import model_manager


def sample():
    img = Image.open("./assets/sample.jpg").convert("RGB")
    # img.show()

    model_inputs = [
        {
            "model": model_manager.ITTModelName.GIT,
            "question": None,
        },
        {
            "model": model_manager.ITTModelName.GITVQA,
            "question": "What is the person doing?",
        },
        {
            "model": model_manager.ITTModelName.ViTGPT2,
            "question": None,
        },
    ]

    mm = model_manager.ModelManager()
    for mi in model_inputs:
        mm.ModelName = mi["model"]
        generated_caption = mm.infer(img=img, question=mi["question"])
        print(mm.ModelName, generated_caption)


def main():
    sample()


if __name__ == "__main__":
    main()
