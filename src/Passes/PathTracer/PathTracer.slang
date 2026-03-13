import MathUtils;
import Scene.Scene;
import RNG;

struct Frame {
    float3 n;
    float3 t;
    float3 b;

    __init(float3 n)
    {
        this.n = n;
        ONB(this.n, t, b);
    }

    float3 to_local(float3 w) { return float3(dot(t, w), dot(b, w), dot(n, w)); }
    float3 to_global(float3 w) { return w.x * t + w.y * b + w.z * n; }
};

struct Path {
    uint2 pixel;
    uint vertex_index;
    RayDesc ray;
    float3 thp;
    float3 L;
    RNG rng;

    __init(uint2 pixel, RayDesc ray, RNG rng)
    {
        this.pixel = pixel;
        this.vertex_index = 0;
        this.ray = ray;
        this.thp = float3(1);
        this.L = float3(0);
        this.rng = rng;
    }
};

float3 diffuse_brdf_sample(float3 wi, float2 uv, out float pdf)
{
    float phi = uv.y * M_2PI;
    float cos_theta = sqrt(1.f - uv.x);
    float sin_theta = sqrt(1.f - cos_theta * cos_theta);
    pdf = cos_theta * M_1_PI;
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float3 envmap(float3 dir)
{
    return float3(1);
    // float3 env = float3(0);
    // env += saturate(dot(dir, float3(0, 1, 0))) * float3(0.5, 0.5, 1);
    // const float3 LIGHT_DIR = normalize(float3(0.5, 1, -0.8));
    // env += saturate(smoothstep(0.95, 0.96, dot(dir, LIGHT_DIR))) * float3(10);
    // return env;
}

void trace_path(inout Path path)
{
    while (path.vertex_index < 5) {
        HitInfo hit = scene.intersect(path.ray);
        if (!hit.valid) {
            path.L += path.thp * envmap(path.ray.Direction);
            break;
        }
        SceneVertex vertex = scene.get_vertex(hit);
        // Flip normal if we hit the backside of the triangle
        vertex.normal *= sign(-dot(vertex.normal, path.ray.Direction));
        Frame frame = Frame(vertex.normal);
        float3 wi = frame.to_local(-path.ray.Direction);
        float pdf;
        float3 wo = diffuse_brdf_sample(wi, path.rng.next_2d(), pdf);
        MaterialDesc material = scene.get_material(hit);
        float3 f = material.base_color * M_1_PI;
        path.thp *= f * abs(dot(wo, frame.n)) / pdf;
        path.vertex_index++;
        path.ray = RayDesc(vertex.position + vertex.normal * 1e-6f, 0, frame.to_global(wo), FLT_MAX);
    }
}

ParameterBlock<Scene> scene;
RWTexture2D<float4> renderTarget;
uniform uint randomSeed;
uniform uint spp;

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    uint2 dim;
    renderTarget.GetDimensions(dim.x, dim.y);
    if (any(pixel >= dim))
        return;

    float3 L = float3(0);
    for (uint i = 0; i < spp; i++) {
        RNG rng = RNG(pixel, randomSeed * spp + i);
        float2 uv = float2(pixel + rng.next_2d()) / float2(dim);
        uv.y = 1 - uv.y;
        RayDesc ray = scene.camera.get_ray(uv);
        Path path = Path(pixel, ray, rng);
        trace_path(path);
        L += path.L;
    }
    L /= spp;

    renderTarget[pixel] = float4(L, 1);
}
