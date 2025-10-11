import slangpy as spy
from typing import Optional

class Accumulator:
    def __init__(self, device: spy.Device):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("Passes/Accumulate/Accumulate.slang", ["main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.history: Optional[spy.Texture] = None
    
    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        image: spy.Texture,
        args: dict
    ):
        reset = not args["history_valid"]
        if (
            self.history == None
            or self.history.width != image.width
            or self.history.height != image.height
        ):
            self.history = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=image.width,
                height=image.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="history",
            )

        self.kernel.dispatch(
            thread_count=[image.width, image.height, 1],
            vars={
                "image": image,
                "history": self.history,
                "reset": reset,
            },
            command_encoder=command_encoder,
        )
