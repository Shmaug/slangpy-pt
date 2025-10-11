import slangpy as spy
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import struct
from Scene.Camera import Camera

class Material:
    def __init__(self, base_color: "spy.float3param" = spy.float3(0.5)):
        super().__init__()
        self.base_color = base_color


class Mesh:
    def __init__(
        self,
        vertices: npt.NDArray[np.float32],  # type: ignore
        indices: npt.NDArray[np.uint32],  # type: ignore
    ):
        super().__init__()
        assert vertices.ndim == 2 and vertices.dtype == np.float32
        assert indices.ndim == 2 and indices.dtype == np.uint32
        self.vertices = vertices
        self.indices = indices

    @property
    def vertex_count(self):
        return self.vertices.shape[0]

    @property
    def triangle_count(self):
        return self.indices.shape[0]

    @property
    def index_count(self):
        return self.triangle_count * 3

    @classmethod
    def create_quad(cls, size: "spy.float2param" = spy.float2(1)):
        vertices = np.array(
            [
                # position, normal, uv
                [-0.5, 0, -0.5, 0, 1, 0, 0, 0],
                [+0.5, 0, -0.5, 0, 1, 0, 1, 0],
                [-0.5, 0, +0.5, 0, 1, 0, 0, 1],
                [+0.5, 0, +0.5, 0, 1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        vertices[:, (0, 2)] *= [size[0], size[1]]
        indices = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
            ],
            dtype=np.uint32,
        )
        return Mesh(vertices, indices)

    @classmethod
    def create_cube(cls, size: "spy.float3param" = spy.float3(1)):
        vertices = np.array(
            [
                # position, normal, uv
                # left
                [-0.5, -0.5, -0.5, 0, -1, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, 0, -1, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, 0, -1, 0, 1.0, 1.0],
                [+0.5, -0.5, -0.5, 0, -1, 0, 0.0, 1.0],
                # right
                [-0.5, +0.5, +0.5, 0, +1, 0, 0.0, 0.0],
                [-0.5, +0.5, -0.5, 0, +1, 0, 1.0, 0.0],
                [+0.5, +0.5, -0.5, 0, +1, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, 0, +1, 0, 0.0, 1.0],
                # back
                [-0.5, +0.5, -0.5, 0, 0, -1, 0.0, 0.0],
                [-0.5, -0.5, -0.5, 0, 0, -1, 1.0, 0.0],
                [+0.5, -0.5, -0.5, 0, 0, -1, 1.0, 1.0],
                [+0.5, +0.5, -0.5, 0, 0, -1, 0.0, 1.0],
                # front
                [+0.5, +0.5, +0.5, 0, 0, +1, 0.0, 0.0],
                [+0.5, -0.5, +0.5, 0, 0, +1, 1.0, 0.0],
                [-0.5, -0.5, +0.5, 0, 0, +1, 1.0, 1.0],
                [-0.5, +0.5, +0.5, 0, 0, +1, 0.0, 1.0],
                # bottom
                [-0.5, +0.5, +0.5, -1, 0, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, -1, 0, 0, 1.0, 0.0],
                [-0.5, -0.5, -0.5, -1, 0, 0, 1.0, 1.0],
                [-0.5, +0.5, -0.5, -1, 0, 0, 0.0, 1.0],
                # top
                [+0.5, +0.5, -0.5, +1, 0, 0, 0.0, 0.0],
                [+0.5, -0.5, -0.5, +1, 0, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, +1, 0, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, +1, 0, 0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        vertices[:, 0:3] *= [size[0], size[1], size[2]]

        indices = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 6, 5],
                [4, 7, 6],
                [8, 10, 9],
                [8, 11, 10],
                [12, 14, 13],
                [12, 15, 14],
                [16, 18, 17],
                [16, 19, 18],
                [20, 22, 21],
                [20, 23, 22],
            ],
            dtype=np.uint32,
        )

        return Mesh(vertices, indices)


class Transform:
    def __init__(self):
        super().__init__()
        self.translation = spy.float3(0)
        self.scaling = spy.float3(1)
        self.rotation = spy.float3(0)
        self.matrix = spy.float4x4.identity()

    def update_matrix(self):
        T = spy.math.matrix_from_translation(self.translation)
        S = spy.math.matrix_from_scaling(self.scaling)
        R = spy.math.matrix_from_rotation_xyz(self.rotation)
        self.matrix = spy.math.mul(spy.math.mul(T, R), S)


class SceneBuilder:
    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.materials = []
        self.meshes = []
        self.transforms = []
        self.instances = []

    def add_material(self, material: Material):
        material_id = len(self.materials)
        self.materials.append(material)
        return material_id

    def add_mesh(self, mesh: Mesh):
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_transform(self, transform: Transform):
        transform_id = len(self.transforms)
        self.transforms.append(transform)
        return transform_id

    def add_instance(self, mesh_id: int, material_id: int, transform_id: int):
        instance_id = len(self.instances)
        self.instances.append((mesh_id, material_id, transform_id))
        return instance_id


class Scene:
    @dataclass
    class MaterialDesc:
        base_color: spy.float3

        def pack(self):
            return struct.pack("fff", self.base_color[0], self.base_color[1], self.base_color[2])

    @dataclass
    class MeshDesc:
        vertex_count: int
        index_count: int
        vertex_offset: int
        index_offset: int

        def pack(self):
            return struct.pack(
                "IIII",
                self.vertex_count,
                self.index_count,
                self.vertex_offset,
                self.index_offset,
            )

    @dataclass
    class InstanceDesc:
        mesh_id: int
        material_id: int
        transform_id: int

        def pack(self):
            return struct.pack("III", self.mesh_id, self.material_id, self.transform_id)

    def __init__(self, device: spy.Device, stage: SceneBuilder):
        super().__init__()
        self.device = device

        self.camera = stage.camera

        # Prepare material descriptors
        self.material_descs = [Scene.MaterialDesc(base_color=m.base_color) for m in stage.materials]
        material_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.material_descs), dtype=np.uint8
        ).flatten()
        self.material_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="material_descs_buffer",
            data=material_descs_data,
        )

        # Prepare mesh descriptors
        vertex_count = 0
        index_count = 0
        self.mesh_descs = []
        for mesh in stage.meshes:
            self.mesh_descs.append(
                Scene.MeshDesc(
                    vertex_count=mesh.vertex_count,
                    index_count=mesh.index_count,
                    vertex_offset=vertex_count,
                    index_offset=index_count,
                )
            )
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count

        # Prepare instance descriptors
        self.instance_descs = []
        for mesh_id, material_id, transform_id in stage.instances:
            self.instance_descs.append(Scene.InstanceDesc(mesh_id, material_id, transform_id))

        # Create vertex and index buffers
        vertices = np.concatenate([mesh.vertices for mesh in stage.meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in stage.meshes], axis=0)
        assert vertices.shape[0] == vertex_count
        assert indices.shape[0] == index_count // 3

        self.vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_buffer",
            data=vertices,
        )

        self.index_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="index_buffer",
            data=indices,
        )

        mesh_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.mesh_descs), dtype=np.uint8
        ).flatten()
        self.mesh_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_descs_buffer",
            data=mesh_descs_data,
        )

        instance_descs_data = np.frombuffer(
            b"".join(d.pack() for d in self.instance_descs), dtype=np.uint8
        ).flatten()
        self.instance_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="instance_descs_buffer",
            data=instance_descs_data,
        )

        # Prepare transforms
        self.transforms = [t.matrix for t in stage.transforms]
        self.inverse_transpose_transforms = [
            spy.math.transpose(spy.math.inverse(t)) for t in self.transforms
        ]
        self.transform_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="transform_buffer",
            data=np.stack([t.to_numpy() for t in self.transforms]),
        )
        self.inverse_transpose_transforms_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="inverse_transpose_transforms_buffer",
            data=np.stack([t.to_numpy() for t in self.inverse_transpose_transforms]),
        )

        # Build BLASes
        self.blases = [self.build_blas(mesh_desc) for mesh_desc in self.mesh_descs]

        # Build TLAS
        self.tlas = self.build_tlas()

    def build_blas(self, mesh_desc: MeshDesc):
        build_input = spy.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [
                    {
                        "buffer": self.vertex_buffer,
                        "offset": mesh_desc.vertex_offset * 32,
                    }
                ],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": 32,
                "index_buffer": {
                    "buffer": self.index_buffer,
                    "offset": mesh_desc.index_offset * 4,
                },
                "index_format": spy.IndexFormat.uint32,
                "index_count": mesh_desc.index_count,
                "flags": spy.AccelerationStructureGeometryFlags.opaque,
            }
        )

        build_desc = spy.AccelerationStructureBuildDesc({"inputs": [build_input]})

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        blas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="blas_scratch_buffer",
        )

        blas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="blas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=blas, src=None, scratch_buffer=blas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return blas

    def build_tlas(self):
        instance_list = self.device.create_acceleration_structure_instance_list(
            size=len(self.instance_descs)
        )
        for instance_id, instance_desc in enumerate(self.instance_descs):
            instance_list.write(
                instance_id,
                {
                    "transform": spy.float3x4(self.transforms[instance_desc.transform_id]),
                    "instance_id": instance_id,
                    "instance_mask": 0xFF,
                    "instance_contribution_to_hit_group_index": 0,
                    "flags": spy.AccelerationStructureInstanceFlags.none,
                    "acceleration_structure": self.blases[instance_desc.mesh_id].handle,
                },
            )

        build_desc = spy.AccelerationStructureBuildDesc(
            {
                "inputs": [instance_list.build_input_instances()],
            }
        )

        sizes = self.device.get_acceleration_structure_sizes(build_desc)

        tlas_scratch_buffer = self.device.create_buffer(
            size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        tlas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=build_desc, dst=tlas, src=None, scratch_buffer=tlas_scratch_buffer
        )
        self.device.submit_command_buffer(command_encoder.finish())

        return tlas
    
    def parameters(self):
        return {
            "tlas": self.tlas,
            "material_descs": self.material_descs_buffer,
            "mesh_descs": self.mesh_descs_buffer,
            "instance_descs": self.instance_descs_buffer,
            "vertices": self.vertex_buffer,
            "indices": self.index_buffer,
            "transforms": self.transform_buffer,
            "inverse_transpose_transforms": self.inverse_transpose_transforms_buffer,
            "camera": self.camera.parameters()
        }
