#pragma once

static const uint UINT32_MAX = 0xFFFFFFFF;

static const float FLT_MAX             = 3.402823466e+38F;         // max value
static const float FLT_MIN             = 1.175494351e-38F;         // min normalized positive value
static const float M_E                 = 2.71828182845904523536;   // e
static const float M_LOG2E             = 1.44269504088896340736;   // log2(e)
static const float M_LOG10E            = 0.434294481903251827651;  // log10(e)
static const float M_LN2               = 0.693147180559945309417;  // ln(2)
static const float M_LN10              = 2.30258509299404568402;   // ln(10)
static const float M_PI                = 3.14159265358979323846;   // pi
static const float M_PI_2              = 1.57079632679489661923;   // pi/2
static const float M_PI_4              = 0.785398163397448309616;  // pi/4
static const float M_1_PI              = 0.318309886183790671538;  // 1/pi
static const float M_2_PI              = 0.636619772367581343076;  // 2/pi
static const float M_2_SQRTPI          = 1.12837916709551257390;   // 2/sqrt(pi)
static const float M_SQRT2             = 1.41421356237309504880;   // sqrt(2)
static const float M_SQRT1_2           = 0.707106781186547524401;  // 1/sqrt(2)
static const float M_2PI               = 6.28318530717958647693;   // 2pi
static const float M_4PI               = 12.5663706143591729539;   // 4pi
static const float M_4_PI              = 1.27323954473516268615;   // 4/pi
static const float M_1_2PI             = 0.159154943091895335769;  // 1/2pi
static const float M_1_4PI             = 0.079577471545947667884;  // 1/4pi
static const float M_SQRT3             = 1.732050807568877293527;  // sqrt(3)
static const float M_SQRTPI            = 1.77245385090551602730;   // sqrt(pi)
static const float M_1_SQRT2           = 0.707106781186547524401;  // 1/sqrt(2)

float2x2 inverse(float2x2 m) {
	const float invdet = 1 / determinant(m);
    float2x2 r;
    r[0][0]  = m[1][1] * invdet;
    r[1][1]  = m[0][0] * invdet;
    r[0][1] = -m[0][1] * invdet;
    r[1][0] = -m[1][0] * invdet;
    return r;
}

float3x3 inverse(float3x3 m) {
	float invdet = 1 / determinant(m);
    float3x3 r;
    r[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    r[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    r[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    r[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    r[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;
    return r;
}

inline float luminance(const float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }
inline float atan2_stable(const float y, const float x) { return x == 0.0 ? (y == 0 ? 0 : (y < 0 ? -M_PI / 2 : M_PI / 2)) : atan2(y, x); }

// cartesian to spherical UV [0-1]
inline float2 xyz2sphuv(const float3 v) {
	const float theta = atan2_stable(v.z, v.x);
	return float2(theta*M_1_PI*.5 + .5, acos(clamp(v.y, -1.f, 1.f))*M_1_PI);
}
// spherical UV [0-1] to cartesian
inline float3 sphuv2xyz(float2 uv) {
	uv.x = uv.x*2 - 1;
	uv *= M_PI;
	const float sinPhi = sin(uv.y);
	return float3(sinPhi*cos(uv.x), cos(uv.y), sinPhi*sin(uv.x));
}

// Octahedral mapping
float2 xyz2oct(const float3 v) {
	const float3 n = v / (abs(v.x) + abs(v.y) + abs(v.z));
	float2 xy = float2(n.x, n.y);
    xy = n.z >= 0.f ? xy : (1.f - abs(float2(n.y, n.x))) * float2(xy.x >= 0.f ? 1.f : -1.f, xy.y >= 0.f ? 1.f : -1.f);
    return xy * 0.5f + float2(0.5f);
}
float3 oct2xyz(const float2 p) {
    float2 f = p * 2.f - float2(1.f);
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    const float z = 1.f - abs(f.x) - abs(f.y);
    const float t = saturate(-z);
    f.x += f.x >= 0.f ? -t : t;
    f.y += f.y >= 0.f ? -t : t;
    return normalize(float3(f, z));
}

inline float3 srgb2rgb(const float3 srgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
	float3 rgb;
	for (int i = 0; i < 3; i++)
		rgb[i] = srgb[i] <= 0.04045 ? srgb[i] / 12.92 : pow((srgb[i] + 0.055) / 1.055, 2.4);
	return rgb;
}
inline float3 rgb2srgb(const float3 rgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
	float3 srgb;
	for (int i = 0; i < 3; i++)
		srgb[i] = rgb[i] <= 0.0031308 ? rgb[i] * 12.92 : pow(rgb[i] * 1.055, 1/2.4) - 0.055;
	return srgb;
}

inline float3 viridis(const float x) {
	// from https://www.shadertoy.com/view/XtGGzG
	float4 x1 = float4(1, x, x*x, x*x*x); // 1 x x2 x3
	float2 x2 = float2(x1.y, x1.z) * x1[3]; // x4 x5
	return float3(
		dot(x1, float4( 0.280268003, -0.143510503,   2.225793877, -14.815088879)) + dot(x2, float2( 25.212752309, -11.772589584)),
		dot(x1, float4(-0.002117546,  1.617109353,  -1.909305070,   2.701152864)) + dot(x2, float2(-1.685288385 ,   0.178738871)),
		dot(x1, float4( 0.300805501,  2.614650302, -12.019139090,  28.933559110)) + dot(x2, float2(-33.491294770,  13.762053843)));
}

inline void ONB(const float3 n, out float3 b1, out float3 b2) {
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = n.z < 0 ? -1 : 1;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}
