import slangpy as spy
from pygltflib import GLTF2
from Scene.Scene import *
import numpy as np
import os
import struct
from pathlib import Path


def _get_buffer_data(gltf: GLTF2, file_path: str):
    """Extract buffer data from GLTF file."""
    base_dir = os.path.dirname(file_path)
    buffer_data = []
    
    for buffer in gltf.buffers:
        if buffer.uri:
            # External buffer file
            buffer_path = os.path.join(base_dir, buffer.uri)
            with open(buffer_path, 'rb') as f:
                buffer_data.append(f.read())
        elif buffer.byteLength:
            # Data URI or embedded data
            if buffer.uri and buffer.uri.startswith('data:'):
                import base64
                header, data = buffer.uri.split(',', 1)
                buffer_data.append(base64.b64decode(data))
            else:
                buffer_data.append(b'\x00' * buffer.byteLength)
    
    return buffer_data


def _get_accessor_data(gltf: GLTF2, accessor_id: int, buffer_data: list):
    """Extract data from an accessor."""
    if accessor_id is None or accessor_id >= len(gltf.accessors):
        return None
    
    accessor = gltf.accessors[accessor_id]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = buffer_data[buffer_view.buffer]
    
    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    
    # Determine component type and count
    component_type_sizes = {
        5120: 1,  # BYTE
        5121: 1,  # UNSIGNED_BYTE
        5122: 2,  # SHORT
        5123: 2,  # UNSIGNED_SHORT
        5125: 4,  # UNSIGNED_INT
        5126: 4,  # FLOAT
    }
    
    type_sizes = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }
    
    component_byte_size = component_type_sizes.get(accessor.componentType, 4)
    type_count = type_sizes.get(accessor.type, 1)
    element_byte_size = component_byte_size * type_count
    
    # Extract data
    stride = buffer_view.byteStride or element_byte_size
    data = []
    
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    
    dtype = dtype_map.get(accessor.componentType, np.float32)
    
    for i in range(accessor.count):
        pos = offset + i * stride
        element = buffer[pos:pos + element_byte_size]
        values = np.frombuffer(element, dtype=dtype, count=type_count)
        data.extend(values)
    
    return np.array(data, dtype=dtype).reshape((accessor.count, type_count))


def _load_mesh_from_gltf(gltf: GLTF2, mesh_id: int, buffer_data: list) -> Mesh:
    """Convert a GLTF mesh to a Mesh object."""
    gltf_mesh = gltf.meshes[mesh_id]
    
    all_vertices = []
    all_indices = []
    
    for primitive in gltf_mesh.primitives:
        # Get position data
        positions = _get_accessor_data(gltf, primitive.attributes.POSITION, buffer_data)
        
        # Get normal data (or compute if not present)
        normals = _get_accessor_data(gltf, primitive.attributes.NORMAL, buffer_data)
        if normals is None:
            # Compute normals from indices later
            normals = np.zeros_like(positions)
        
        # Get texture coordinates (or use zeros)
        texcoords = _get_accessor_data(gltf, primitive.attributes.TEXCOORD_0, buffer_data)
        if texcoords is None:
            texcoords = np.zeros((len(positions), 2), dtype=np.float32)
        
        # Get indices
        indices = _get_accessor_data(gltf, primitive.indices, buffer_data)
        if indices is not None:
            indices = indices.flatten().astype(np.uint32)
        else:
            indices = np.arange(len(positions), dtype=np.uint32)
        
        # Create vertices
        vertex_offset = len(all_vertices)
        vertices = []
        for i in range(len(positions)):
            pos = spy.float3(positions[i][0], positions[i][1], positions[i][2])
            normal = spy.float3(normals[i][0], normals[i][1], normals[i][2])
            uv = spy.float2(texcoords[i][0], texcoords[i][1] if len(texcoords[i]) > 1 else 0)
            vertices.append(MeshVertex(pos, normal, uv))
        
        all_vertices.extend(vertices)
        
        # Adjust indices and add them
        indices_adjusted = indices.reshape(-1, 3) + vertex_offset
        all_indices.extend(indices_adjusted)
    
    vertices_array = np.array(all_vertices, dtype=MeshVertex)
    indices_array = np.array(all_indices, dtype=np.uint32)
    
    return Mesh(vertices_array, indices_array)


def _load_material_from_gltf(gltf: GLTF2, material_id: int, texture_loader: spy.TextureLoader, file_path: str) -> Material:
    """Convert a GLTF material to a Material object."""
    if material_id is None or material_id >= len(gltf.materials):
        return Material()
    
    gltf_material = gltf.materials[material_id]
    
    # Get base color
    base_color = spy.float3(0.5, 0.5, 0.5)
    if hasattr(gltf_material, 'pbrMetallicRoughness') and gltf_material.pbrMetallicRoughness:
        pbr = gltf_material.pbrMetallicRoughness
        if pbr.baseColorFactor:
            base_color = spy.float3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2])
    
    # Get base color texture
    base_color_texture = None
    if hasattr(gltf_material, 'pbrMetallicRoughness') and gltf_material.pbrMetallicRoughness:
        pbr = gltf_material.pbrMetallicRoughness
        if pbr.baseColorTexture:
            texture_index = pbr.baseColorTexture.index
            if texture_index < len(gltf.textures):
                gltf_texture = gltf.textures[texture_index]
                if gltf_texture.source < len(gltf.images):
                    image = gltf.images[gltf_texture.source]
                    if image.uri:
                        image_path = os.path.join(os.path.dirname(file_path), image.uri)
                        try:
                            base_color_texture = texture_loader.loadTexture(image_path)
                        except:
                            pass
    
    return Material(base_color=base_color, base_color_texture=base_color_texture)


def _build_scene_from_gltf(gltf: GLTF2, stage: SceneBuilder, buffer_data: list, texture_loader: spy.TextureLoader, file_path: str):
    """Populate a SceneBuilder from GLTF data."""
    
    # Material map (GLTF material ID -> stage material ID)
    material_map = {}
    
    # Load materials
    for i, gltf_material in enumerate(gltf.materials):
        material = _load_material_from_gltf(gltf, i, texture_loader, file_path)
        material_id = stage.add_material(material)
        material_map[i] = material_id
    
    # Default material if none specified
    default_material_id = stage.add_material(Material())
    
    # Mesh map (GLTF mesh ID -> stage mesh ID)
    mesh_map = {}
    
    # Load meshes
    for i, gltf_mesh in enumerate(gltf.meshes):
        mesh = _load_mesh_from_gltf(gltf, i, buffer_data)
        mesh_id = stage.add_mesh(mesh)
        mesh_map[i] = mesh_id
    
    # Process nodes and build hierarchy
    def process_node(node_id: int, parent_transform: Transform = None):
        gltf_node = gltf.nodes[node_id]
        
        # Create transform for this node
        transform = Transform()
        
        # Apply node transformation
        if gltf_node.translation:
            transform.translation = spy.float3(gltf_node.translation[0], gltf_node.translation[1], gltf_node.translation[2])
        
        if gltf_node.scale:
            transform.scaling = spy.float3(gltf_node.scale[0], gltf_node.scale[1], gltf_node.scale[2])
        
        if gltf_node.rotation:
            # Convert quaternion to euler angles (simplified)
            # For now, keep as identity - proper conversion would be complex
            transform.rotation = spy.float3(0, 0, 0)
        
        # Combine with parent transform if needed
        transform.update_matrix()
        transform_id = stage.add_transform(transform)
        
        # Create instances for meshes at this node
        if gltf_node.mesh is not None:
            mesh_id = mesh_map[gltf_node.mesh]
            # Get material for this mesh
            material_id = default_material_id
            if gltf_node.mesh < len(gltf.meshes):
                gltf_mesh = gltf.meshes[gltf_node.mesh]
                if gltf_mesh.primitives and gltf_mesh.primitives[0].material is not None:
                    material_id = material_map.get(gltf_mesh.primitives[0].material, default_material_id)
            
            stage.add_instance(mesh_id, material_id, transform_id)
        
        # Process children
        if gltf_node.children:
            for child_id in gltf_node.children:
                process_node(child_id, transform)
    
    # Process all root nodes
    if gltf.scenes and len(gltf.scenes) > 0:
        scene = gltf.scenes[0]
        if scene.nodes:
            for node_id in scene.nodes:
                process_node(node_id)


def load_gltf(device : spy.Device, file):
    stage = SceneBuilder()

    texture_loader = spy.TextureLoader(device)
    
    gltf : GLTF2 = GLTF2().load(file)
    
    # Extract buffer data
    buffer_data = _get_buffer_data(gltf, file)
    
    # Build scene from GLTF
    _build_scene_from_gltf(gltf, stage, buffer_data, texture_loader, file)

    print(f"Loaded {len(gltf.nodes)} nodes, {len(stage.meshes)} meshes, {len(stage.materials)} materials")

    return Scene(device, stage)