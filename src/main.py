import os
import slangpy as spy
# from Gui.Gui import Gui
from Scene.Scene import *
from Scene.Camera import CameraController
from Passes.Accumulate.Accumulate import Accumulator
from Passes.PathTracer.PathTracer import PathTracer
from Passes.ToneMapper.ToneMapper import ToneMapper

def demo_scene():
    stage = SceneBuilder()
    stage.camera.target   = spy.float3(0, 1, 0)
    stage.camera.position = spy.float3(2, 1, 2)

    floor_material  = stage.add_material(Material(base_color=spy.float3(0.5)))
    floor_mesh      = stage.add_mesh(Mesh.create_quad([5, 5]))
    floor_transform = stage.add_transform(Transform())
    stage.add_instance(floor_mesh, floor_material, floor_transform)

    cube_materials = []
    for _ in range(10):
        cube_materials.append(
            stage.add_material(
                Material(base_color=spy.float3(np.random.rand(3).astype(np.float32)))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
            )
        )
    cube_mesh = stage.add_mesh(Mesh.create_cube([0.1, 0.1, 0.1]))

    for i in range(1000):
        transform = Transform()
        transform.translation = spy.float3((np.random.rand(3) * 2 - 1).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
        transform.translation[1] += 1
        transform.scaling = spy.float3((np.random.rand(3) + 0.5).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
        transform.rotation = spy.float3((np.random.rand(3) * 10).astype(np.float32))  # type: ignore (TYPINGTODO: need explicit np->float conversion)
        transform.update_matrix()
        cube_transform = stage.add_transform(transform)
        stage.add_instance(cube_mesh, cube_materials[i % len(cube_materials)], cube_transform)
    
    return stage

class App:
    def __init__(self):
        super().__init__()
        self.window = spy.Window(width=1920, height=1080, title="PathTracer", resizable=True)
        self.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [os.path.abspath("")]},
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure({
            "width":  self.window.width,
            "height": self.window.height,
            "vsync": False
        })

        self.device.register_shader_hot_reload_callback(self.on_shader_reload)
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event    = self.on_mouse_event
        self.window.on_resize         = self.on_resize

        # self.gui = Gui(self.device, self.window.width, self.window.height)

        self.scene = Scene(self.device, demo_scene())

        self.camera_controller = CameraController(self.scene.camera)

        self.passes = [
            PathTracer(self.device, self.scene),
            Accumulator(self.device),
            ToneMapper(self.device)
        ]

        self.render_texture: spy.Texture = None  # type: ignore (will be set immediately)

        # self.gui.refresh_font_texture()

    def on_shader_reload(self, e:spy.ShaderHotReloadEvent):
        self.history_valid = False

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f1:
                if self.output_texture:
                    spy.tev.show_async(self.output_texture)
            elif event.key == spy.KeyCode.f2:
                if self.output_texture:
                    bitmap = self.output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")

        self.camera_controller.on_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent):
        self.camera_controller.on_mouse_event(event)

    def on_resize(self, width: int, height: int):
        self.device.wait()
        self.surface.configure({"width": width, "height": height, "vsync": False})
        # self.gui.on_resize(width, height)

    def render(self, command_encoder):
        args = {
            "history_valid": self.history_valid
        }
        for p in self.passes:
            p.execute(command_encoder, self.render_texture, args)
        self.history_valid = True

    def main_loop(self):
        timer = spy.Timer()
        while not self.window.should_close():
            dt = timer.elapsed_s()
            timer.reset()

            self.window.process_events()

            if self.camera_controller.update(dt):
                self.history_valid = False

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            # self.gui.new_frame()

            if (
                self.render_texture == None
                or self.render_texture.width != surface_texture.width
                or self.render_texture.height != surface_texture.height
            ):
                self.render_texture = self.device.create_texture(
                    format=spy.Format.rgba32_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.render_target,
                    label="render_texture",
                )
                self.history_valid = False

            command_encoder = self.device.create_command_encoder()

            self.render(command_encoder)

            # self.gui.render(command_encoder, self.render_texture)
            
            # self.gui.end_frame(command_encoder, self.render_texture)

            command_encoder.blit(surface_texture, self.render_texture)
            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

        self.device.wait()


app = App()
app.main_loop()
