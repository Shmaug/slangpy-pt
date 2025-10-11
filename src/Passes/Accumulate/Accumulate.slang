RWTexture2D<float4> image;
RWTexture2D<float4> history;
bool reset;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pixel = tid.xy;

    float4 cur = float4(image[pixel].rgb, 1.0);
    if (!reset) cur += history[pixel];

    history[pixel] = cur;
    image[pixel] = float4(cur.rgb / cur.w, 1.0);
}
