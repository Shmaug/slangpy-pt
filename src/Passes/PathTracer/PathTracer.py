import slangpy as spy
from Scene.Scene import Scene

class PathTracer:
    def __init__(self, device: spy.Device, scene: Scene):
        super().__init__()
        self.device = device
        self.scene = scene
        self.program = self.device.load_program("Passes/PathTracer/PathTracer.slang", ["main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.frame = 0
        self.device.register_shader_hot_reload_callback(self.on_shader_reload)

    def on_load_scene(self, scene):
        self.frame = 0
        self.scene = scene

    def on_shader_reload(self, e:spy.ShaderHotReloadEvent):
        self.frame = 0

    def execute(self,
        command_encoder: spy.CommandEncoder,
        renderTarget: spy.Texture,
        args: dict
    ):
        w = renderTarget.width
        h = renderTarget.height

        self.scene.camera.width = w
        self.scene.camera.height = h
        self.scene.camera.recompute()

        self.kernel.dispatch(
            thread_count=[renderTarget.width, renderTarget.height, 1],
            vars={
                "scene": self.scene.shader_parameters(),
                "renderTarget": renderTarget,
                "randomSeed": self.frame,
                "spp": 4
            },
            command_encoder=command_encoder,
        )

        self.frame += 1
