uniform float2 proj_scale;

[shader("vertex")]
float4 vertex_main(
    float2 pos: POSITION,
    float2 uv: TEXCOORD0,
    float4 color: COLOR,
    out float2 fragUV: TEXCOORD0,
    out float4 fragColor: COLOR
) : SV_Position
{
    fragUV = uv;
    fragColor = color;

    return float4(
        pos.x * proj_scale.x,
        pos.y * proj_scale.y,
        0,
        -pos.x + pos.y);
}

Texture2D<float4> Texture;
SamplerState TextureSampler;

[shader("fragment")]
float4 fragment_main(
    float2 uv: TEXCOORD0,
    float4 color: COLOR
) : SV_Target
{
    return color * Texture.Sample(TextureSampler, uv);
}