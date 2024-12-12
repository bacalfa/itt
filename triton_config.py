class TritonConfig:
    def __init__(self, name: str, platform: str, max_batch_size: int, inputs: list, outputs: list, **kwargs):
        self.name: str = name
        self.platform: str = platform
        self.max_batch_size: int = max_batch_size
        self.inputs: list = inputs
        self.outputs: list = outputs
        self.kwargs: dict = kwargs

    def __str__(self) -> str:
        s = f"""
        name: "{self.name}"
        """

        s += f"""
        platform: "{self.platform}"
        """

        s += f"""
        max_batch_size: {self.max_batch_size}
        """

        s += f"""
        input
        [
        """
        for i_ind, i in enumerate(self.inputs):
            s += "\t{"
            for k, v in i.items():
                s += f"""
                \t\t{k}: {v}
                """
            s += "\t}"

            if i_ind < len(self.inputs) - 1:
                s += ","
        s += f"""
        ]
        """

        s += f"""
        output
        [
        """
        for i_ind, i in enumerate(self.outputs):
            s += "\t{"
            for k, v in i.items():
                s += f"""
                \t\t{k}: {v}
                """
            s += "\t}"

            if i_ind < len(self.outputs) - 1:
                s += ","
        s += f"""
        ]
        """

        return s
