import model_manager_triton


def create_model_repository():
    mm = model_manager_triton.ModelManager()
    mm.create_model_repository()


def main():
    create_model_repository()


if __name__ == "__main__":
    main()
