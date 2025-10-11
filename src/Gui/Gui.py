import slangpy as spy
import numpy as np
import imgui
import ctypes

class Gui:
    def __init__(self, device: spy.Device, width, height):
        imgui.create_context()
        imgui.get_io().display_size = width, height
        
        self.c_index_type = None
        self.index_format = None
        match imgui.INDEX_SIZE:
            case 1:
                self.c_index_type = ctypes.c_int8
                self.index_format = spy.IndexFormat.uint8
            case 2:
                self.c_index_type = ctypes.c_int16
                self.index_format = spy.IndexFormat.uint16
            case 4:
                self.c_index_type = ctypes.c_int32
                self.index_format = spy.IndexFormat.uint32
            case _: assert("Invalid imgui.INDEX_SIZE: {}".format(imgui.INDEX_SIZE))

        self.device = device
        self.program = self.device.load_program("Gui/Gui.3d.slang", ["vertex_main", "fragment_main"])
        self.pipeline = None

        self.texture_array = [None]
        imgui.get_io().fonts.texture_id = 0
        self.linear_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        )

    def texture_id(self, texture):
        try:
            i = self.texture_array.index(texture)
        except ValueError as e:
            i = len(self.texture_array)
            self.texture_array.append(texture)
        return i

    def refresh_font_texture(self):
        io = imgui.get_io()
        width, height, pixels = io.fonts.get_tex_data_as_rgba32()
        pixels = np.frombuffer(pixels, dtype=np.uint8)
        pixel_data = np.copy(pixels)
        tex = self.device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="font_atlas",
            data=pixel_data
        )
        io.fonts.clear_tex_data()
        self.texture_array[0] = tex

    def on_resize(self, width, height):
        imgui.get_io().display_size = width, height

    def new_frame(self):
        imgui.new_frame()

    def render(self, command_encoder: spy.CommandEncoder, render_texture: spy.Texture):
        imgui.begin("Custom window", True)
        imgui.show_test_window()
        imgui.end()

    def end_frame(self, command_encoder: spy.CommandEncoder, render_texture: spy.Texture):
        imgui.render()
        
        draw_data = imgui.get_draw_data()

        if self.pipeline is None or self.pipeline.targets[0].format != render_texture.format:
            vertex_layout = self.device.create_input_layout(
                input_elements=[
                    {
                        "semantic_name": "POSITION",
                        "semantic_index": 0,
                        "format": spy.Format.rg32_float,
                        "offset": 4*0
                    },
                    {
                        "semantic_name": "TEXCOORD",
                        "semantic_index": 0,
                        "format": spy.Format.rg32_float,
                        "offset": 4*2
                    },
                    {
                        "semantic_name": "COLOR",
                        "semantic_index": 0,
                        "format": spy.Format.rgba32_float,
                        "offset": 4*4
                    }
                ],
                vertex_streams=[{"stride": imgui.VERTEX_SIZE}],
            )
            self.pipeline = self.device.create_render_pipeline(
                program=self.program,
                input_layout=vertex_layout,
                targets=[{"format": render_texture.format}],
            )

        # Execute command lists

        with command_encoder.begin_render_pass(
            {"color_attachments": [{"view": render_texture.create_view({})}]}
        ) as pass_encoder:
            pass_encoder.bind_pipeline(self.pipeline)

            for draw_list in draw_data.commands_lists:
                vertex_buffer = self.device.create_buffer(
                    element_count=draw_list.vtx_buffer_size,
                    struct_size=imgui.VERTEX_SIZE,
                    usage=spy.BufferUsage.vertex_buffer,
                    label="vertex_buffer"
                )

                index_buffer = self.device.create_buffer(
                    element_count=draw_list.idx_buffer_size,
                    struct_size=imgui.INDEX_SIZE,
                    usage=spy.BufferUsage.index_buffer,
                    label="index_buffer"
                )

                vertices = np.ctypeslib.as_array(
                    ctypes.cast(draw_list.vtx_buffer_data, ctypes.POINTER(ctypes.c_float)),
                    (draw_list.vtx_buffer_size * int(imgui.VERTEX_SIZE/4),))
                
                indices = np.ctypeslib.as_array(
                    ctypes.cast(draw_list.vtx_buffer_data, ctypes.POINTER(self.c_index_type)),
                    (draw_list.idx_buffer_size,))
                
                vertex_buffer.copy_from_numpy(vertices)
                index_buffer.copy_from_numpy(indices)

                idx_buffer_offset = 0
                for command in draw_list.commands:
                    print(command)
                    # pass_encoder.set_render_state({
                    #     "viewports": [spy.Viewport.from_size(render_texture.width, render_texture.height)],
                    #     "scissor_rects": [
                    #         spy.ScissorRect({
                    #             "min_x": command.clip_rect[0],
                    #             "min_y": command.clip_rect[1],
                    #             "max_x": command.clip_rect[2],
                    #             "max_y": command.clip_rect[3],
                    #         })
                    #     ],
                    #     "vertex_buffers": [vertex_buffer],
                    #     "index_buffer": index_buffer,
                    #     "index_format": self.index_format,
                    # })

                    # texture = self.texture_array[command.texture_id] if command.texture_id >= 0 and command.texture_id < len(self.texture_array) else None

                    # pass_encoder.draw_indexed({
                    #     "index_count": command.elem_count,
                    #     "start_index_location": idx_buffer_offset})
                    # idx_buffer_offset += command.elem_count

                del vertex_buffer
                del index_buffer
        imgui.end_frame()
