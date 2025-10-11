import slangpy as spy

class ToneMapper:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("Passes/ToneMapper/ToneMapper.slang", ["main"])
        self.kernel = self.device.create_compute_kernel(self.program)

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        image: spy.Texture,
        args: dict
    ):
        self.kernel.dispatch(
            thread_count=[image.width, image.height, 1],
            vars={
                "image": image,
            },
            command_encoder=command_encoder,
        )
