struct RNG {
    uint state;

    __init(uint seed) { state = seed; }

    __init(uint2 pixel, uint sample_index)
    {
        state = (pixel.x * 100591) ^ (pixel.y * 102191) ^ (sample_index * 103643);
    }

    [mutating]
    uint next()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_1d() { return (next() >> 8) * 0x1p-24; }

    [mutating]
    float2 next_2d()
    {
        float2 result;
        result.x = next_1d();
        result.y = next_1d();
        return result;
    }

    [mutating]
    float3 next_3d()
    {
        float3 result;
        result.x = next_1d();
        result.y = next_1d();
        result.z = next_1d();
        return result;
    }
};
