float3 aces_film(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

RWTexture2D<float4> image;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pixel = tid.xy;
    image[pixel] = float4(aces_film(image[pixel].rgb), 1.0);
}
