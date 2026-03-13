import MathUtils;

static const uint SCENE_TEXTURE_ARRAY_SIZE = 1024;

struct Camera {
    float3 position;
    float3 image_u;
    float3 image_v;
    float3 image_w;

    RayDesc get_ray(float2 uv)
    {
        uv = uv * 2 - 1;
        float3 dir = normalize(uv.x * image_u + uv.y * image_v + image_w);
        return RayDesc(position, 0, dir, FLT_MAX);
    }
};

struct MaterialDesc {
    float3 base_color;
    uint   base_color_texture_id;
};

struct MeshVertex {
    float3 position;
    float  u;
    float3 normal;
    float  v;

    property float2 uv { get { return float2(u, v); } }
};

struct MeshDesc {
    uint vertex_count;
    uint index_count;
    uint vertex_offset;
    uint index_offset;
};

struct InstanceDesc {
    uint mesh_id;
    uint material_id;
    uint transform_id;
    uint padding;
};

struct HitInfo {
    bool valid;
    float2 bary;
    float t;
    uint instance_id;
    uint primitive_index;
};

struct SceneVertex {
    float3 position;
    float3 normal;
    float2 uv;
}

struct Scene {
    RaytracingAccelerationStructure tlas;

    StructuredBuffer<MaterialDesc> material_descs;
    StructuredBuffer<MeshDesc>     mesh_descs;
    StructuredBuffer<InstanceDesc> instance_descs;
    StructuredBuffer<MeshVertex>   vertices;
    StructuredBuffer<uint>         indices;
    StructuredBuffer<float4x4>     transforms;
    StructuredBuffer<float4x4>     inverse_transpose_transforms;

    SamplerState sampler;
    Texture2D<float4> textures[SCENE_TEXTURE_ARRAY_SIZE];

    Camera camera;

    MaterialDesc get_material(HitInfo hit_info)
    {
        InstanceDesc instance_desc = instance_descs[hit_info.instance_id];
        return material_descs[instance_desc.material_id];
    }

    SceneVertex get_vertex(HitInfo hit_info)
    {
        return get_vertex(hit_info.instance_id, hit_info.primitive_index, hit_info.bary);
    }

    SceneVertex get_vertex(uint instance_id, uint primitive_index, float2 bary)
    {
        InstanceDesc instance_desc = instance_descs[instance_id];
        MeshDesc mesh_desc = mesh_descs[instance_desc.mesh_id];
        float4x4 transform = transforms[instance_desc.transform_id];
        float4x4 inverse_transpose_transform = inverse_transpose_transforms[instance_desc.transform_id];
        uint i0 = indices[mesh_desc.index_offset + primitive_index * 3 + 0];
        uint i1 = indices[mesh_desc.index_offset + primitive_index * 3 + 1];
        uint i2 = indices[mesh_desc.index_offset + primitive_index * 3 + 2];
        MeshVertex v0 = vertices[mesh_desc.vertex_offset + i0];
        MeshVertex v1 = vertices[mesh_desc.vertex_offset + i1];
        MeshVertex v2 = vertices[mesh_desc.vertex_offset + i2];
        float3 b = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        return {
            mul(transform, float4(b.x * v0.position + b.y * v1.position + b.z * v2.position, 1.0)).xyz,
            normalize(
                mul(inverse_transpose_transform, float4(b.x * v0.normal + b.y * v1.normal + b.z * v2.normal, 0)).xyz
            ),
            b.x * v0.uv + b.y * v1.uv + b.z * v2.uv,
        };
    }

    HitInfo intersect(RayDesc ray)
    {
        HitInfo hit = {};

        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(tlas, 0, 0xff, ray);
        while (q.Proceed()) {
            // TODO: non-opaque geometry
        }
            
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            hit = {
                true,
                q.CommittedTriangleBarycentrics(),
                q.CommittedRayT(),
                q.CommittedInstanceIndex(),
                q.CommittedPrimitiveIndex(),
            };
        }
        return hit;
    }
};