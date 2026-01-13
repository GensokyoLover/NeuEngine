/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
__exported import Scene.Shading;
import Scene.Impostor.Impostor;
static const uint objectCount = 11;
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float> gDepth;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gMask;

RWTexture2D<float4> gAlbedo0[objectCount];
RWTexture2D<float4> gAlbedo1[objectCount];
RWTexture2D<float4> gAlbedo2[objectCount];
RWTexture2D<float4> gDepth0[objectCount];
RWTexture2D<float4> gDepth1[objectCount];
RWTexture2D<float4> gDepth2[objectCount];
RWTexture2D<float4> gDir1[objectCount];
RWTexture2D<float4> gDir2[objectCount];
RWTexture2D<float4> gDir0[objectCount];
RWTexture2D<float4> gPrePosition;
RWTexture2D<float4> gPreDirection;
RWTexture2D<float4> gPreRoughness;

#define is_valid(name) (is_valid_##name != 0)

#if !defined(COMPUTE_DEPTH_OF_FIELD) || !defined(USE_ALPHA_TEST) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif
struct ImageRay
{
    float2 uv0;    // ray 进入 impostor 时的 uv
    float2 duv_dt; // uv 对 t 的导数
    float tEnter;  // ray 进入 impostor 的 t
};

struct ViewDDA
{
    uint viewIdx;

    int ix, iy;
    int stepX, stepY;

    float tMaxX, tMaxY;
    float tDeltaX, tDeltaY;

    float2 uvEnter;
    bool valid;
};
static float deg[101] = { 0,      0.057,  0.228,  0.513,  0.913,  1.427,  2.057,  2.805,  3.673,  4.664,  5.782,  7.033,  8.423, 9.959,  11.651, 13.507, 15.538, 17.752, 20.159, 22.764, 25.57,  28.572, 31.756, 35.101, 38.569, 42.115, 45.683, 49.215, 52.651, 55.94,  59.042, 61.927, 64.582, 67.001, 69.191, 71.163, 72.932, 74.514, 75.928, 77.191, 78.318, 79.326, 80.228, 81.035, 81.76,  82.411, 82.997, 83.525, 84.002, 84.434, 84.825, 85.181, 85.505, 85.8,   86.069, 86.316, 86.541, 86.749, 86.939, 87.115, 87.276, 87.426, 87.564, 87.691, 87.81, 87.92,  88.023, 88.118, 88.207, 88.29,  88.367, 88.439, 88.507, 88.571, 88.631, 88.687, 88.74,  88.789, 88.836, 88.88,  88.922, 88.961, 88.999, 89.034, 89.067, 89.099, 89.129, 89.158, 89.185, 89.211, 89.235, 89.259, 89.281, 89.302, 89.323, 89.342, 89.361, 89.379, 89.395, 89.412, 89.427 };
// static float half_tan[101] = {0.0,   0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.011, 0.014, 0.017, 0.021, 0.025, 0.029, 0.034, 0.039, 0.045, 0.05, 0.057, 0.063, 0.07,  0.077, 0.085, 0.093, 0.102, 0.111, 0.12,  0.13,  0.141, 0.152, 0.163, 0.175, 0.187, 0.2,   0.214, 0.228, 0.243, 0.259, 0.275, 0.292, 0.31,  0.328, 0.348, 0.368, 0.389, 0.411, 0.435, 0.459, 0.484, 0.51,  0.538, 0.567, 0.597, 0.629, 0.661, 0.695, 0.731, 0.768, 0.807, 0.847, 0.889, 0.932, 0.977, 1.024, 1.072, 1.122, 1.173, 1.227, 1.282, 1.338, 1.397, 1.457, 1.519, 1.582, 1.647, 1.714, 1.783, 1.853, 1.924, 1.998, 2.072, 2.149, 2.227, 2.306, 2.387, 2.47, 2.554, 2.639, 2.726, 2.814, 2.904, 2.995, 3.087, 3.181, 3.276, 3.372, 3.47,  3.569, 3.669, 3.77,  3.873 };
static float half_tan[101] = {0.0,   0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.011, 0.014, 0.017, 0.021, 0.025, 0.029, 0.034, 0.039, 0.045, 0.05, 0.057, 0.063, 0.07,  0.077, 0.085, 0.093, 0.102, 0.111, 0.12,  0.13,  0.141, 0.152, 0.163, 0.175, 0.187, 0.2,   0.214, 0.228, 0.243, 0.259, 0.275, 0.292, 0.31,  0.328, 0.348, 0.368, 0.389, 0.411, 0.435, 0.459, 0.484, 0.51,  0.538, 0.567, 0.597, 0.629, 0.661, 0.695, 0.731, 0.768, 0.807, 0.847, 0.889, 0.932, 0.977, 1.024, 1.072, 1.122, 1.173, 1.227, 1.282, 1.338, 1.397, 1.457, 1.519, 1.582, 1.647, 1.714, 1.783, 1.853, 1.924, 1.998, 2.072, 2.149, 2.227, 2.306, 2.387, 2.47, 2.554, 2.639, 2.726, 2.814, 2.904, 2.995, 3.087, 3.181, 3.276, 3.372, 3.47,  3.569, 3.669, 3.77,  3.873 };
//static float half_tan[101] = {0.001,  0.004,  0.009,  0.016,  0.025,  0.036,  0.049,  0.064,  0.082,  0.101,  0.123,  0.148,  0.176,  0.206,  0.24,   0.278,  0.32, 0.367,  0.42,   0.478,  0.545,  0.619,  0.703,  0.797,  0.904,  1.024,  1.159,  1.31,   1.479,  1.667,  1.875,  2.104,  2.356,  2.631, 2.931,  3.257,  3.609,  3.989,  4.398,  4.837,  5.306,  5.806,  6.339,  6.905,  7.505,  8.14,   8.811,  9.517,  10.261, 11.042, 11.861, 12.72,  13.616, 14.553, 15.53,  16.545, 17.604, 18.701, 19.84,  21.019, 22.24,  23.503, 24.806, 26.148, 27.535, 28.965, 30.427, 31.94, 33.487, 35.075, 36.705, 38.376, 40.094, 41.833, 43.626, 45.449, 47.325, 49.229, 51.168, 53.134, 55.16,  57.218, 59.297, 61.414, 63.593, 65.795, 68.01,  70.262, 72.586, 74.892, 77.302, 79.672, 82.087, 84.596, 87.085, 89.597, 92.193, 94.731, 97.413, 100.015};
struct VBufferRT
{
    static const bool kComputeDepthOfField = COMPUTE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kRayFlags = RAY_FLAGS;
    static const bool kMultiView = false;

    uint2 frameDim;
    uint frameCount;
    Impostor gImpostor0;
    Impostor gImpostor1;
    Impostor gImpostor2;
    Impostor gImpostor3;
    Impostor gImpostor4;
    Impostor gImpostor5;
    Impostor gImpostor6;
    Impostor gImpostor7;
    Impostor gImpostor8;
    Impostor gImpostor9;
    Impostor gImpostor10;
    int roughness;
    int impostorCnt;
    
    Ray generateRay(uint2 pixel)
    {
        if (kMultiView){
            return gScene.camera.computeRayMultiView(pixel, frameDim, false);
        }
        if (gScene.camera.data.focalLength == 0)
            return gScene.camera.computeRayOrthographic(pixel, frameDim);
        if (kComputeDepthOfField)
        {
            SampleGenerator sg = SampleGenerator(pixel, frameCount);
            return gScene.camera.computeRayThinlens(pixel, frameDim, sampleNext2D(sg));
        }
        else
        {
            return gScene.camera.computeRayPinhole(pixel, frameDim,false);
        }
    }
    float3x3 RotationMatrixFromVectors(float3 from, float3 to) {
        // 假设 from 和 to 都是单位向量
        float3 fromNorm = normalize(from);
        float3 toNorm = normalize(to);

        float dotProd = dot(fromNorm, toNorm);

        // 如果平行，返回单位矩阵
        if (abs(dotProd - 1.0) < 1e-6) {
            return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
        }
        if (abs(dotProd + 1.0) < 1e-6) {
            // 反向：需要特殊处理，但这里假设角度 < 180
            return float3x3(1, 0, 0, 0, -1, 0, 0, 0, -1); // 示例，实际需根据情况
        }

        // 计算旋转轴（叉积）
        float3 axis = cross(fromNorm, toNorm);
        axis = normalize(axis);

        // 计算旋转角度（通过余弦）
        float angle = acos(dotProd);

        // 构建旋转矩阵 (Rodrigues' rotation formula)
        float3 axisSq = axis * axis;
        float sinA = sin(angle);
        float cosA = cos(angle);

        float3x3 rotMat;
        rotMat[0][0] = cosA + axisSq.x * (1.0 - cosA);
        rotMat[0][1] = axis.x * axis.y * (1.0 - cosA) - axis.z * sinA;
        rotMat[0][2] = axis.x * axis.z * (1.0 - cosA) + axis.y * sinA;

        rotMat[1][0] = axis.y * axis.x * (1.0 - cosA) + axis.z * sinA;
        rotMat[1][1] = cosA + axisSq.y * (1.0 - cosA);
        rotMat[1][2] = axis.y * axis.z * (1.0 - cosA) - axis.x * sinA;

        rotMat[2][0] = axis.z * axis.x * (1.0 - cosA) - axis.y * sinA;
        rotMat[2][1] = axis.z * axis.y * (1.0 - cosA) + axis.x * sinA;
        rotMat[2][2] = cosA + axisSq.z * (1.0 - cosA);

        return rotMat;
    }

    // 在给定的圆锥内均匀采样光线（8x8 = 64 样本）
    // 假设本地圆锥以 +Z 轴为中心（float3(0,0,1)）
    void SampleRaysInCone(Ray inputRay, float half_deg, out Ray newRays[64]) {
        // 转换为弧度
        float half_rad = half_deg * (3.14159265359 / 180.0);
        float cosHalf = cos(half_rad);
        float sinHalf = sin(half_rad);

        // 预计算旋转矩阵：从本地 Z 轴旋转到 inputRay.direction
        float3 localZ = float3(0, 0, 1);
        float3x3 rotMat = RotationMatrixFromVectors(localZ, inputRay.dir);

        int sampleCount = 64;
        int gridSize = 8;

        // 均匀采样：在圆锥截面（圆盘）上均匀采样，然后投影到单位球面
        // 这确保了方向的均匀分布（面积加权）
        for (int i = 0; i < sampleCount; ++i) {
            // 使用网格索引生成准随机 u, v (简单均匀网格，不是真正随机，但适合 shader)
            int ix = i % gridSize;
            int iy = i / gridSize;

            // Hammersley-like 低差异序列（可选，提升均匀性）
            // float u = (ix + 0.5) / float(gridSize);  // 中心采样
            float u = float(ix) / float(gridSize - 1); // 简单均匀
            float v = float(iy) / float(gridSize - 1);

            // 对于圆锥内均匀方向采样：
            // 1. 在单位圆盘上均匀采样 (r = sqrt(v), theta = 2*pi*u)
            // 2. 投影到锥面：local_dir.z = cosHalf, xy 缩放 sinHalf
            //    但为了方向均匀，使用 cos_phi 线性插值
            float theta = 2.0 * 3.14159265359 * u;

            // phi 从 0 到 half_rad，cos_phi 从 1 到 cos(half_rad)，线性在 cos 空间（均匀投影）
            float cos_phi = lerp(cosHalf, 1.0, v); // v=0: phi=half_rad (边缘), v=1: phi=0 (中心)
            float sin_phi = sqrt(1.0 - cos_phi * cos_phi);

            // 本地笛卡尔坐标（相对于 Z 轴）
            float3 localDir;
            localDir.x = sin_phi * cos(theta);
            localDir.y = sin_phi * sin(theta);
            localDir.z = cos_phi;

            // 归一化（虽然已经是单位向量）
            localDir = normalize(localDir);

            // 应用旋转：将本地方向旋转到 inputRay.direction
            float3 worldDir = mul(rotMat, localDir);

            // 生成新光线
            newRays[i].origin = inputRay.origin;        // 起点相同
            newRays[i].dir = normalize(worldDir); // 确保单位向量
        }
    }
    
    

    float2 OctEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
    return n.xy * 0.5 + 0.5;
}

uint3 getNearestImpostorView(in Impostor impostor,float3 dir,int level)
{
    float2 uv = OctEncode(dir); // [-1,1] → [0,1]
    uint viewIdx = impostor.texFaceIndex[level].Load(int3(uv * impostor.baseCameraResolution[level], 0));

    // 然后直接采样对应层的 impostor 纹理
    return impostor.cFace[level][viewIdx];
}

float3 hash3(uint n)
{
    // 类似 GLSL 的黄金比例 hash
    n = (n << 13u) ^ n;
    float x = frac((n * 0.1031));
    float y = frac((n * 0.11369));
    float z = frac((n * 0.13787));
    return float3(x, y, z);
}

bool BuildImageRay(
    in Impostor impostor,
    uint viewIdx,
    float3 rayPosW,
    float3 rayDirW,
    out ImageRay imgRay,
    float tEnter,
    int level
)
{
    // -----------------------------
    // 1. impostor view parameters
    // -----------------------------
    float3 camPos = impostor.cPosition[level][viewIdx];
    float3 camR = impostor.cRight[level][viewIdx];
    float3 camU = impostor.cUp[level][viewIdx];
    float3 camF = impostor.cForward[level][viewIdx];
    float radius = impostor.cRadius[level][viewIdx];

    // -----------------------------
    // 2. ray-sphere entry
    // -----------------------------
    float tEnter;
    if (!RaySphereIntersect(rayPosW - impostor.centerWS[level].xyz,
                            rayDirW,
                            1,
                            tEnter))
            return false;

    // -----------------------------
    // 3. ray in view space
    // -----------------------------
    float3 O = rayPosW - camPos;
    float3 D = rayDirW;

    float ox = dot(O, camR);
    float oy = dot(O, camU);
    float oz = dot(O, camF);

    float dx = dot(D, camR);
    float dy = dot(D, camU);
    float dz = dot(D, camF);

    // -----------------------------
    // 4. orthographic impostor projection
    //    u = 0.5 + x / (2R)
    // -----------------------------
    float invScale = 0.5 / radius;

    imgRay.tEnter = tEnter;
    imgRay.uv0 = float2(
        0.5 + (ox + tEnter * dx) * invScale,
        0.5 + (oy + tEnter * dy) * invScale
    );

    imgRay.duv_dt = float2(
        dx * invScale,
        dy * invScale
    );

    return true;
}

bool BuildImageRay2(
    in Impostor impostor,
    uint viewIdx,
    float3 rayPosW,
    float3 rayDirW,
    out ImageRay imgRay,
    int level
)
{
    // -----------------------------
    // 1. impostor view parameters
    // -----------------------------
    float3 camPos = impostor.cPosition[level][viewIdx];
    float3 camR = impostor.cRight[level][viewIdx];
    float3 camU = impostor.cUp[level][viewIdx];
    float3 camF = impostor.cForward[level][viewIdx];
    float radius = impostor.cRadius[level][viewIdx];

    // -----------------------------
    // 2. ray-sphere entry
    // -----------------------------

    // -----------------------------
    // 3. ray in view space
    // -----------------------------
    float3 O = rayPosW - camPos;
    float3 D = rayDirW;

    float ox = dot(O, camR);
    float oy = dot(O, camU);
    float oz = dot(O, camF);

    float dx = dot(D, camR);
    float dy = dot(D, camU);
    float dz = dot(D, camF);

    // -----------------------------
    // 4. orthographic impostor projection
    //    u = 0.5 + x / (2R)
    // -----------------------------
    float invScale = 0.5 / radius;

    imgRay.uv0 = float2(
        0.5 + ox  * invScale,
        0.5 + oy  * invScale
    );

    imgRay.duv_dt = float2(
        dx * invScale,
        dy * invScale
    );

    return true;
}




// bool RecursiveTraverseImpostorImageDDA(
//     in Impostor impostor,
//     uint3 triView[],
//     float3 rayPosW,
//     float3 rayDirW,
//     out float hitDepth,
//     float tNear,
//     float tFar,
//     out int iter_cnt,
// )
// {
//     ImageRay imgRay[4][3];
//     ViewDDA view[4][3];
//     for (int l = 0; l < 4; l++) {
//         for (int i = 0; i < 3; i++) {
//             if (!BuildImageRay(impostor, triView[l][i], rayPosW, rayDirW, imgRay[l][i], tNear, l))
//                 return false;
            
//         }
//     }

//     uint level = 3;
//     const float eps = 1e-6;

//     const int MAX_PIXEL_STEPS = 1024 * 3;
//     float t = tNear;
//     float tEnter[4];
//     float tExit[4];
//     for (int i = 0; i < 4; i++) {
//         tEnter[i] = tNear;
//         tExit[i] = tFar;
//     } 
    
    
//     for (int iter = 0; iter < MAX_PIXEL_STEPS; ++iter)
//     {
//         iter_cnt = iter;
//         float tNext = 1e30;
//         int nextView = -1;
//         bool stepX = false; 
        
        
//         for (int i = 0; i < 3; ++i)
//         {
//             InitViewDDA(impostor, triView[level][i], imgRay[level][i], t, view[level][i], level);
//             if (!view[level][i].valid) continue;

//             if (view[level][i].tMaxX < tNext)
//             {
//                 tNext = view[level][i].tMaxX;
//                 nextView = i;
//                 stepX = true;
//             }
//             if (view[level][i].tMaxY < tNext)
//             {
//                 tNext = view[level][i].tMaxY;
//                 nextView = i;
//                 stepX = false;
//             }
//         }

//         float tWorldEnter = t;
//         float tWorldExit = t + tNext;
        
//         float tMid = 0.5 * (tWorldEnter + tWorldExit);
        
//         ViewDDA s = view[level][nextView];
//         ImageRay ray = imgRay[level][nextView];
//         // ✅ 代表点对应的 uv（严格沿 ray 的投影线）
//         float2 uv =
//             s.uvEnter + ray.duv_dt * (tMid - t);

//         float viewDepth =
//         SampleImpostorPoint(
//             impostor.texDepth,
//             impostor.samplerPoint[level],
//             s.viewIdx,
//             uv, level).r;
//         float3 camPos = impostor.cPosition[level][s.viewIdx];
//         float3 camR = impostor.cRight[level][s.viewIdx];
//         float3 camU = impostor.cUp[level][s.viewIdx];
//         float3 camF = impostor.cForward[level][s.viewIdx];
//         float rView = impostor.cRadius[level][s.viewIdx];
//         float depthRayEnter = dot(
//             (rayPosW + tWorldEnter * rayDirW) - camPos,
//             camF
//         );

//         float depthRayExit = dot(
//             (rayPosW + tWorldExit * rayDirW) - camPos,
//             camF
//         );
//         float rayMinDepth = min(depthRayEnter, depthRayExit);
//         float rayMaxDepth = max(depthRayEnter, depthRayExit);
//         if (viewDepth - 1e-3 < rayMaxDepth)
//         {
//             if (level == 0) {
//                 float rView = impostor.cRadius[level][s.viewIdx];

//                 // 用 uv 重建 view 平面上的 x,y（与你原先的定义一致）
//                 float x = (uv.x * 2.0 - 1.0) * rView;
//                 float y = (uv.y * 2.0 - 1.0) * rView;

//                 float3 Pview = camPos + x * camR + y * camU + viewDepth * camF;

//                 float3 samplePos = rayPosW + tMid * rayDirW;
//                 float dist = length(samplePos - Pview);

//                 // TODO：把这个阈值换成你原来那套更严谨的判定
//                 if (dist < impostor.radius[level] * pow(2, level) / 128.0)
//                 {
//                     hitDepth = tMid;
//                     return true;
//                 }
//             }
//             else {
//                 level -= 1;
//                 t = tWorldEnter;
//                 tEnter[level] = tWorldEnter;
//                 tExit[level] = tWorldExit;
//                 continue; // 重新开始遍历更高分辨率的层
//             }
//         }
//         else {
//             if (stepX)
//             {
//                 view[level][nextView].ix += view[level][nextView].stepX;
//                 view[level][nextView].tMaxX += view[level][nextView].tDeltaX;
//             }
//             else
//             {
//                 view[level][nextView].iy += view[level][nextView].stepY;
//                 view[level][nextView].tMaxY += view[level][nextView].tDeltaY;
//             }

//             // 越界则该 view 失效
//             if (view[level][nextView].ix < 0 || view[level][nextView].ix >= impostor.texDim[level].x ||
//         view[level][nextView].iy < 0 || view[level][nextView].iy >= impostor.texDim[level].x)
//             {
//                 view[level][nextView].valid = false;
//             }

//             t = tWorldExit;
//             if (level < 3 && t > tExit[level]) {
//                 level = level + 1;
//                 tEnter[level + 1] = t;
//                 continue;
//             }
//             else if (level == 3 && t > tExit[3])
//                 return false;
//         }
        
//         // ---------------------------------------
//         // 4) 前进到下一个 cell：比较 tMaxX 和 tMaxY
//         // ---------------------------------------
        

//         // 处理极端：如果 du/dv 很小，tDelta 会非常大，导致 tWorld 直接超过 tExit
//         // 上面已经用 tWorld>tExit 退出了
//     }

//     return false;
// }

// 辅助函数：计算当前 t 距离下一个像素边界的距离
// 返回到达下一个 grid line 需要增加的 t (dt)
float GetDistToNextBoundary(float t, float2 uv0, float2 duvdt, float res) {
    float2 currentUV = uv0 + duvdt * t;
    float2 pixelPos = currentUV * res;

    // 当前像素索引
    float2 iPos = floor(pixelPos);

    // 确定下一个边界的像素坐标
    // 如果 duvdt > 0，下一个边界是 iPos + 1
    // 如果 duvdt < 0，下一个边界是 iPos
    //float2 nextBoundaryPixel = (duvdt > 0.0) ? (iPos + 1.0) : iPos;
    float2 nextBoundaryPixel =
    select(duvdt > 0.0, iPos + 1.0, iPos);
    // 计算到达边界的 UV 差值
    float2 deltaUV = (nextBoundaryPixel - pixelPos) / res;

    // 转换为 t 的差值 (dt = du / (du/dt))
    // 加一个微小的 epsilon 防止除零和卡在边界上
    float dtX = (abs(duvdt.x) > 1e-8) ? deltaUV.x / duvdt.x : 1e30;
    float dtY = (abs(duvdt.y) > 1e-8) ? deltaUV.y / duvdt.y : 1e30;

    // 必须是正数（向前走）
    return min(max(dtX, 0.0), max(dtY, 0.0));
}

// 输入：点 P，线段端点 A 和 B (均为 float3)
// 输出：距离 (float)
float DistancePointToSegment(float3 P, float3 A, float3 B) {
    float3 v = B - A; // 线段方向向量
    float3 w = P - A; // 从 A 到 P 的向量

    // 计算投影参数 t
    float lenSq = dot(v, v); // ||v||²，避免开方
    if (lenSq == 0.0) {
        // 特殊情况：A 和 B 重合，退化为点到点的距离
        return length(P - A);
    }

    float t = dot(w, v) / lenSq;

    // 钳位 t 到 [0, 1] 范围
    float3 Q;
    if (t < 0.0) {
        Q = A; // 最近点是 A
    } else if (t > 1.0) {
        Q = B; // 最近点是 B
    } else {
        Q = A + t * v; // 投影点在段内
    }

    // 返回距离
    return length(P - Q);
}
        
bool CheckImpostorHit(
    in Impostor impostor,
    uint viewIndex,   // 传入 triView[level][sellect_view]
    ImageRay rayData, // 传入 imgRay[level][sellect_view]
    float3 rayPosW,
    float3 rayDirW,
    float tEnter,
    float tExit,
    uint level,
    float depth_tanh
)
{
    // 1. 计算阈值
    float threshold = (1 / 128.0) * float(1 << level) ;

    // 2. 计算 UV (基于进入点)
    float2 uv = rayData.uv0 + rayData.duv_dt * tEnter;

    // 3. 采样纹理深度 (假设 rg 通道分别存储 MinDepth 和 MaxDepth)
    float2 textureDepth = SampleImpostorPoint(
        impostor.texDepthArray,
        impostor.samplerPoint[level],
        viewIndex,
        uv,
        level
    ).rg;
    // float2 textureDepth = SampleImpostorPoint(
    //     impostor.texDepth, 
    //     viewIndex, 
    //     uv, 
    //     level,
    //     512 / (1 << level)
    // ).rg;
    // 4. 获取当前视角的相机参数
    float3 camPos = impostor.cPosition[level][viewIndex];
    float3 camF = impostor.cForward[level][viewIndex];

    // 5. 计算光线在 View 空间下的深度区间
    float rayDepthAtBegin = dot((rayPosW + tEnter * rayDirW) - camPos, camF);
    float rayDepthAtExit = dot((rayPosW + tExit * rayDirW) - camPos, camF);

    float rayDepthMax = max(rayDepthAtBegin, rayDepthAtExit);
    float rayDepthMin = min(rayDepthAtBegin, rayDepthAtExit);
    float eps_min = rayDepthMin * depth_tanh;
    float eps_max = rayDepthMax * depth_tanh;
    // 6. 深度重叠测试 (Intersection Test)
    // 逻辑：光线的最大深度大于物体的最小深度，且光线的最小深度小于物体的最大深度(带阈值)
    
    if (textureDepth.x - eps_min < rayDepthMax && rayDepthMin - (1 / 128.0) < textureDepth.y + eps_max)
    {
        return true;
    }
    return false;
}
// 计算圆锥截锥内部到平面的最小和最大距离
float2 CalculateMinMaxDistancesInFrustum( float3 position, float3 direction, float halfAngle, float near, float far, float3 planePoint, float3 planeNormal )
{
    // 归一化平面法向量（假设用户输入未归一化；如果已归一化，可跳过）
    float3 N = normalize(planeNormal);

    // direction 已归一化（用户保证）
    float cosPhi = dot(direction, N);
    float sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi)); // 避免负值
    float tanHalfAngle = tan(halfAngle);

    // 上边界和下边界系数
    float coeff_upper = cosPhi + tanHalfAngle * sinPhi;
    float coeff_lower = cosPhi - tanHalfAngle * sinPhi;

    // 计算 max_proj = max_t (t * coeff_upper)
    float max_proj;
    if (coeff_upper >= 0.0) {
        max_proj = coeff_upper * far;
    } else {
        max_proj = coeff_upper * near;
    }

    // 计算 min_proj = min_t (t * coeff_lower)
    float min_proj;
    if (coeff_lower >= 0.0) {
        min_proj = coeff_lower * near;
    } else {
        min_proj = coeff_lower * far;
    }

    // 计算带符号距离范围
    float pDotN = dot(position - planePoint, N); // 使用平面内部点来计算
    float s_min = min_proj + pDotN;   // 带符号的最小距离
    float s_max = max_proj + pDotN;   // 带符号的最大距离

    // 计算最小和最大距离
    float min_dist = min(s_min, s_max); // 取最小值
    float max_dist = max(s_min, s_max); // 取最大值

    return float2(min_dist, max_dist);
}

float2 SampleImpostorDepth4Texels(
    Texture2DArray<float4> texArray,
    uint viewIndex,
    float2 uv,
    uint level,
    uint2 baseResolution // mip0 分辨率，例如 (512,512)
) {
    // 当前 mip 的分辨率
    uint2 texSize = baseResolution >> level;

    // uv -> texel space（对齐 bilinear 规则）
    float2 texelPos = uv * texSize - 0.5;
    int2 p0 = int2(floor(texelPos));

    int2 p00 = p0;
    int2 p10 = p0 + int2(1, 0);
    int2 p01 = p0 + int2(0, 1);
    int2 p11 = p0 + int2(1, 1);

    int2 maxP = int2(texSize) - 1;
    p00 = clamp(p00, int2(0, 0), maxP);
    p10 = clamp(p10, int2(0, 0), maxP);
    p01 = clamp(p01, int2(0, 0), maxP);
    p11 = clamp(p11, int2(0, 0), maxP);

    float2 d00 = texArray.Load(int4(p00, viewIndex,0)).rg;
    float2 d10 = texArray.Load(int4(p10, viewIndex,0)).rg;
    float2 d01 = texArray.Load(int4(p01, viewIndex,0)).rg;
    float2 d11 = texArray.Load(int4(p11, viewIndex,0)).rg;

    // 合并为“大区间”
    float minDepth = min(min(d00.x, d10.x), min(d01.x, d11.x));
    float maxDepth = max(max(d00.y, d10.y), max(d01.y, d11.y));

    return float2(minDepth, maxDepth);
}
bool SingleCheckImpostorHit(
    in Impostor impostor,
    uint viewIndex,   // 传入 triView[level][sellect_view]
    ImageRay rayData, // 传入 imgRay[level][sellect_view]
    float3 rayPosW,
    float3 rayDirW,
    float tEnter,
    float tExit,
    uint level,
    float depth_tanh,
    out float4 sample,
)
{
    // 1. 计算阈值
    float threshold = (1 / 128.0) * float(1 << level);

    // 2. 计算 UV (基于进入点)
    float2 uv = rayData.uv0 + rayData.duv_dt * (tEnter + tExit)/2;

    // 3. 采样纹理深度 (假设 rg 通道分别存储 MinDepth 和 MaxDepth)
    float2 textureDepth = SingleSampleImpostorPoint(
        impostor.texDepthArray,
        impostor.samplerPoint[0],
        viewIndex,
        uv,
        level
    ).rg;
    // float2 textureDepth = SampleImpostorDepth4Texels(
    //     impostor.texDepthArray[level],
    //     viewIndex,
    //     uv,
    //     level,
    //     uint2(512,512)
    // );
    // float2 textureDepth = SampleImpostorPoint(
    //     impostor.texDepth,
    //     viewIndex,
    //     uv,
    //     level,
    //     512 / (1 << level)
    // ).rg;
    // 4. 获取当前视角的相机参数
    float3 camPos = impostor.cPosition[0][viewIndex];
    float3 camF = impostor.cForward[0][viewIndex];

    // 5. 计算光线在 View 空间下的深度区间
    // float rayDepthAtBegin = dot((rayPosW + tEnter * rayDirW) - camPos, camF);
    // float rayDepthAtExit = dot((rayPosW + tExit * rayDirW) - camPos, camF);
    // float rayDepthMax = max(rayDepthAtBegin, rayDepthAtExit);
    // float rayDepthMin = min(rayDepthAtBegin, rayDepthAtExit);

    float2 project = CalculateMinMaxDistancesInFrustum(rayPosW, rayDirW, depth_tanh,tEnter,tExit,camPos,camF);
    float rayDepthMin = project.x;
    float rayDepthMax = project.y;
    float eps_min = rayDepthMin * depth_tanh;
    float eps_max = rayDepthMax * depth_tanh;
    // 6. 深度重叠测试 (Intersection Test)
    // 逻辑：光线的最大深度大于物体的最小深度，且光线的最小深度小于物体的最大深度(带阈值)
    sample = float4(textureDepth.x, textureDepth.y, 0, 1);
    if (textureDepth.x < rayDepthMax + (1 / 128.0) && rayDepthMin - (1 / 128.0) < textureDepth.y )
    {
        //sample = float4(0);
        return true;
    }
    //sample = float4(1);
    return false;
}

// 定义点采样器：必须使用 Point 模式，防止硬件自动插值
SamplerState PointClampSampler; // Filter = MIN_MAG_MIP_POINT, Address = Clamp/Wrap

float2 GetMipSize(float2 baseSize, uint mipLevel) {
    return max(float2(1.0, 1.0), baseSize * exp2(-float(mipLevel)));
}

// 输出结构体，包含处理后的 R 和 G
struct MinMaxResult {
    float minR;
    float maxG;
};

float2 SmartMinMaxSample(
    Impostor impostor,     // 纹理
    uint viewIdx,
    float2 uinterval,         // 原始 UV
    float2 vinterval,    // 采样半径
    float2 texBaseSize, // 纹理 Base 分辨率
    out uint level,
)
{
    MinMaxResult result;

    // -----------------------------------------------------------
    // 1. 计算合适的 Mip Level
    // -----------------------------------------------------------
    // 计算 uvLength 对应的像素覆盖范围
    float2 uv = float2(uinterval.x + uinterval.y, vinterval.x + vinterval.y) * 0.5;
    float uvLength = max(uinterval.y - uinterval.x, vinterval.y - vinterval.x);
    float pixelFootprint = uvLength * max(texBaseSize.x, texBaseSize.y);
    // 算出 mip 层级
    uint mipLevel = (uint)min(max(0.0, ceil(log2(pixelFootprint))),9);

    // -----------------------------------------------------------
    // 2. 准备坐标数据
    // -----------------------------------------------------------
    float2 mipSize = GetMipSize(texBaseSize, mipLevel);
    float2 oneTexelSize = 1.0 / mipSize;
    level = mipLevel;
    // 像素坐标 (Texel Space): 比如 (10.5, 20.3)
    float2 texelPos = uv * mipSize;

    // -----------------------------------------------------------
    // 3. 判断是否需要 4-Tap
    // -----------------------------------------------------------
    // 计算当前采样点相对于像素中心的偏移量 (-0.5 到 0.5)
    // frac(texelPos) -> 0..1, 中心是 0.5
    float2 distToCenter = abs(frac(texelPos) - 0.5);

    // 计算到最近边缘的距离 (0..0.5)
    float2 distToEdge = 0.5 - distToCenter;

    // 采样半径 (转为像素单位)
    float texelRadius = uvLength * max(mipSize.x, mipSize.y);

    // 判定：如果半径小于到边缘的距离，说明圆完全在像素内 -> 1 Tap
    // 加一个小的 epsilon 防止浮点误差
    bool isInside = texelRadius < (min(distToEdge.x, distToEdge.y) - 0.001);

    if (isInside)
    {
        // --- 情况 A: 单次采样 ---
        // 直接采样最近的一个点
        float4 val = impostor.texDepthArray[mipLevel].SampleLevel(impostor.samplerPoint[0], float3(uv,viewIdx), 0);
        result.minR = val.r;
        result.maxG = val.g;
    }
    else
    {
        // --- 情况 B: 4次采样 (模拟 Bilinear 的足迹) ---
        // 我们需要找到参与 Bilinear 插值的 4 个邻近像素中心
        // 算法：找到左上角的那个像素中心 UV，然后 +1 Texel

        // 这里的数学原理：
        // 标准 Bilinear 的基准点是 floor(pos - 0.5)
        // 我们要取这个基准点对应的像素中心 UV
        float2 baseTexelIndex = floor(texelPos - 0.5);

        // 转换回 UV 空间，并偏移到像素中心 (+0.5)
        float2 baseUV = (baseTexelIndex + 0.5) * oneTexelSize;

        // 采样 4 个点
        // TL: Top-Left, TR: Top-Right, BL: Bottom-Left, BR: Bottom-Right
        float4 c00 = impostor.texDepthArray[mipLevel].SampleLevel(impostor.samplerPoint[0], float3(baseUV, viewIdx), 0);
        float4 c10 = impostor.texDepthArray[mipLevel].SampleLevel(impostor.samplerPoint[0], float3(baseUV + float2(oneTexelSize.x, 0), viewIdx), 0);
        float4 c01 = impostor.texDepthArray[mipLevel].SampleLevel(impostor.samplerPoint[0], float3(baseUV + float2(0, oneTexelSize.y), viewIdx), 0);
        float4 c11 = impostor.texDepthArray[mipLevel].SampleLevel(impostor.samplerPoint[0], float3(baseUV + oneTexelSize,viewIdx), 0);

        // --- 执行自定义的 Min/Max 聚合逻辑 ---

        // 计算 R 通道的最小值 (Conservative Empty Space)
        result.minR = min(min(c00.r, c10.r), min(c01.r, c11.r));

        // 计算 G 通道的最大值 (Conservative Object Thickness/Height)
        result.maxG = max(max(c00.g, c10.g), max(c01.g, c11.g));
    }

    return float2(result.minR,result.maxG);
}

bool SphereCheckImpostorHit(// 球体假设
    in Impostor impostor,
    uint viewIndex,   // 传入 triView[level][sellect_view]
    ImageRay rayData, // 传入 imgRay[level][sellect_view]
    float3 rayPosW,
    float3 rayDirW,
    float tEnter,
    float tExit,
    out uint level,
    float depth_tanh,
    out float4 sample,
    float2 uinterval,
    float2 vinterval,
)
{


    // 2. 计算 UV (基于进入点)
    float2 uv = rayData.uv0 + rayData.duv_dt * tEnter;
    float2 textureDepth;
    textureDepth = SmartMinMaxSample(impostor, viewIndex, uinterval, vinterval, float2(512, 512), level);
  
    float3 camPos = impostor.cPosition[0][viewIndex];
    float3 camF = impostor.cForward[0][viewIndex];

    // float rayMidDepth = dot((rayPosW + tEnter * rayDirW) - camPos, camF);
    // float rayDepthMax = rayMidDepth + tEnter * depth_tanh ;
    // float rayDepthMin = rayMidDepth - tEnter * depth_tanh ;

    float2 project = CalculateMinMaxDistancesInFrustum(rayPosW, rayDirW, depth_tanh, tEnter, tExit, camPos, camF);
    float rayDepthMin = project.x;
    float rayDepthMax = project.y;
    // float eps_min = rayDepthMin * depth_tanh;
    // float eps_max = rayDepthMax * depth_tanh;
    // 6. 深度重叠测试 (Intersection Test)
    // 逻辑：光线的最大深度大于物体的最小深度，且光线的最小深度小于物体的最大深度(带阈值)
    
    if (textureDepth.x < rayDepthMax + (1 / 128.0) && rayDepthMin - (1 / 128.0) < textureDepth.y)
    {
        // sample = float4(0);
        return true;
    }
    // sample = float4(1);
    return false;
}
static int res_list[5] = { 512, 182, 64, 24, 8 } ; 


bool RecursiveTraverseImpostorImageDDA(
    in Impostor impostor,
    uint3 triView[5],
float3 rayPosW,
    float3 rayDirW,
    out float hitDepthEnter,
    out float hitDepthExit,
    out float3 compare_pos,
    float tNear,
    float tFar,
    out float iter_cnt,
    out float4 sampleUV,
    float depth_tanh,
)
{
    sampleUV = float4(1,1,1,1);
    ImageRay imgRay[5][3];
    for (int l = 0; l < 5; l++) {
        for (int i = 0; i < 3; i++) {
            if (!BuildImageRay2(impostor, triView[l][i], rayPosW, rayDirW, imgRay[l][i], l)) {
                sampleUV = float4(0, 0, 1, 1);
                return false;
            }
        }
    }

    uint level = 4;
    float tCurrent = tNear;

    float tExitStack[5];
    tExitStack[4] = tFar;

    const int MAX_STEPS = 2048;
    float4 ot = float4(0, 0, 0, 1);
    uint endlevel = 0;
    float inv_scale = (tCurrent * depth_tanh) / 1 ;
    if (inv_scale <= 1.0 / 512)
        endlevel = 0;
    else if (inv_scale <= 1.0 / 182)
        endlevel = 1;
    else if (inv_scale <= 1.0 / 64)
        endlevel = 2;
    else if (inv_scale <= 1.0 / 24)
        endlevel = 3;
    else
        endlevel = 4;

    for (int iter = 0; iter < MAX_STEPS; ++iter)
    {
        
        iter_cnt = iter ;

        float stepSize = 1e30;

        float res = res_list[level];
        int sellect_view = -1;
        for (int i = 0; i < 3; ++i) {
            float dt = GetDistToNextBoundary(tCurrent, imgRay[level][i].uv0, imgRay[level][i].duv_dt, res);
            if (dt < stepSize) {
                sellect_view = i;
                stepSize = dt;
            }
        }
        
        stepSize += 1e-5;
        
        float tWorldEnter = tCurrent;
        float tWorldExit = tCurrent + stepSize;
        if (tWorldExit > tExitStack[level]) {
            tWorldExit = tExitStack[level];
        }


        uint currentViewIdx = triView[level][0];
        ImageRay currentRayData = imgRay[level][0];
        uint currentViewIdx2 = triView[level][1];
        ImageRay currentRayData2 = imgRay[level][1];
        uint currentViewIdx3 = triView[level][2];
        ImageRay currentRayData3 = imgRay[level][2];
        bool isHitCandidate = CheckImpostorHit(
            impostor, 
            currentViewIdx, 
            currentRayData, 
            rayPosW, 
            rayDirW, 
            tWorldEnter, 
            tWorldExit, 
            level, depth_tanh
        ) || CheckImpostorHit(
            impostor, 
            currentViewIdx2, 
            currentRayData2, 
            rayPosW, 
            rayDirW, 
            tWorldEnter, 
            tWorldExit, 
            level, depth_tanh
        ) || CheckImpostorHit(
            impostor, 
            currentViewIdx3, 
            currentRayData3, 
            rayPosW, 
            rayDirW, 
            tWorldEnter, 
            tWorldExit, 
            level, depth_tanh
        );
        if (isHitCandidate)
        {
            if (level == endlevel) {
                    hitDepthEnter = tWorldEnter;
                    hitDepthExit = tWorldExit;
                    compare_pos = float3(0);
                sampleUV = float4(1,0,0,1);
                return true;
            }
            else {
                level--;
                tExitStack[level] = tWorldExit;
                continue;
            }
        }

        tCurrent = tWorldExit;
        inv_scale = (tCurrent * depth_tanh) / 1;
        if (inv_scale <= 1.0 / 512)
            endlevel = 0;
        else if (inv_scale <= 1.0 / 182)
            endlevel = 1;
        else if (inv_scale <= 1.0 / 64)
            endlevel = 2;
        else if (inv_scale <= 1.0 / 24)
            endlevel = 3;
        else
            endlevel = 4;

        while ((level < 4 && tCurrent >= tExitStack[level] - 1e-4) || level < endlevel) {
            level = level + 1;
        }
        if (level == 4 && tCurrent >= tExitStack[4]) {
            sampleUV = float4(0, 1, 0, 1);
            iter_cnt = imgRay[4][0].duv_dt.x;
            return false;
        }
    }
    sampleUV = ot;
    return false;
}

// bool RecursiveTraverseImpostorImageDDASphere(
//     in Impostor impostor,
//     uint3 triView[4],
//     float3 rayPosW,
//     float3 rayDirW,
//     out float hitDepthEnter,
//     out float hitDepthExit,
//     out float3 compare_pos,
//     float tNear,
//     float tFar,
//     out int iter_cnt,
//     out float4 sampleUV,
// )
// {
//     sampleUV = float4(1, 1, 1, 1);
//     ImageRay imgRay[4][3];
//     for (int l = 0; l < 4; l++) {
//         for (int i = 0; i < 3; i++) {
//             if (!BuildImageRay2(impostor, triView[l][i], rayPosW, rayDirW, imgRay[l][i], l)) {
//                 sampleUV = float4(0, 0, 1, 1);
//                 return false;
//             }
//         }
//     }

//     uint level = 3;
//     float tCurrent = tNear;

//     float tExitStack[4];
//     tExitStack[3] = tFar;

//     const int MAX_STEPS = 512;
//     float4 ot = float4(0, 0, 0, 1);

//     for (int iter = 0; iter < MAX_STEPS; ++iter)
//     {
//         iter_cnt = iter;

//         float stepSize = 1e30;

//         float res = impostor.texDim[0].x / float(1 << level);
//         int sellect_view = -1;
//         for (int i = 0; i < 3; ++i) {
//             float dt = GetDistToNextBoundary(tCurrent, imgRay[level][i].uv0, imgRay[level][i].duv_dt, res);
//             if (dt < stepSize) {
//                 sellect_view = i;
//                 stepSize = dt;
//             }
//         }

//         stepSize += 1e-5;

//         float tWorldEnter = tCurrent;
//         float tWorldExit = tCurrent + stepSize;

//         if (tWorldExit > tExitStack[level]) {
//             tWorldExit = tExitStack[level];
//         }

//         uint currentViewIdx = triView[level][0];
//         ImageRay currentRayData = imgRay[level][0];
//         uint currentViewIdx2 = triView[level][1];
//         ImageRay currentRayData2 = imgRay[level][1];
//         uint currentViewIdx3 = triView[level][2];
//         ImageRay currentRayData3 = imgRay[level][2];
//         bool isHitCandidate = CheckImpostorHit(
//             impostor, 
//             currentViewIdx, 
//             currentRayData, 
//             rayPosW, 
//             rayDirW, 
//             tWorldEnter, 
//             tWorldExit, 
//             level
//         ) || CheckImpostorHit(
//             impostor, 
//             currentViewIdx2, 
//             currentRayData2, 
//             rayPosW, 
//             rayDirW, 
//             tWorldEnter, 
//             tWorldExit, 
//             level
//         ) || CheckImpostorHit(
//             impostor, 
//             currentViewIdx3, 
//             currentRayData3, 
//             rayPosW, 
//             rayDirW, 
//             tWorldEnter, 
//             tWorldExit, 
//             level
//         );
//         if (isHitCandidate)
//         {
//             if (level == 0) {
//                 hitDepthEnter = tWorldEnter;
//                 hitDepthExit = tWorldExit;
//                 compare_pos = float3(0);
//                 sampleUV = float4(1, 0, 0, 1);
//                 return true;
//             }
//             else {
//                 level--;
//                 tExitStack[level] = tWorldExit;
//                 continue;
//             }
//         }

//         tCurrent = tWorldExit;

//         while (level < 3 && tCurrent >= tExitStack[level] - 1e-4) {
//             level = level + 1;
//         }
//         if (level == 3 && tCurrent >= tExitStack[3]) {
//             sampleUV = float4(0, 1, 0, 1);
//             return false;
//         }
//     }
//     sampleUV = ot;
//     return false;
// }


int ComputePowerOfTwoLevel_NoBranch(float smallRadius, float bigRadius)
{
    smallRadius = max(smallRadius, 1e-8);
    bigRadius = saturate(bigRadius);

    float ratio = bigRadius / smallRadius;

    // max(ratio, 1) 防止 log2 负数
    float kf = ceil(log2(max(ratio, 1.0)));

    return 9 - min((int)kf, 9);
}
bool SingleRecursiveTraverseImpostorImageDDA(
    in Impostor impostor,
    uint3 triView,
    float3 rayPosW,
    float3 rayDirW,
    out float hitDepthEnter,
    out float hitDepthExit,
    out float3 compare_pos,
    float tNear,
    float tFar,
    out uint iter_cnt,
    out float4 sampleUV[3],
    float depth_tanh,
)
{
    ImageRay imgRay[3];
    for (int i = 0; i < 3; i++) {
        if (!BuildImageRay2(impostor, triView[i], rayPosW, rayDirW, imgRay[i],0)) {
   
            return false;
        }
    }

    uint level = 9;
    float tCurrent = tNear;

    float tExitStack[10];
    tExitStack[9] = tFar;

    const int MAX_STEPS = 2048;
    float4 ot = float4(0, 0, 0, 1);
    uint endlevel = ComputePowerOfTwoLevel_NoBranch(tCurrent * depth_tanh, 1);
    // iter_cnt = endlevel / 9.0;
    // return true;

    for (int iter = 0; iter < MAX_STEPS; ++iter)
    {
        

        float stepSize = 1e30;

        float res = 512 >> level;
        int sellect_view = -1;
        for (int i = 0; i < 1; ++i) {
            float dt = GetDistToNextBoundary(tCurrent, imgRay[i].uv0, imgRay[i].duv_dt, res);
            if (dt < stepSize) {
                sellect_view = i;
                stepSize = dt;
            }
        }

        stepSize += 1e-5;

        float tWorldEnter = tCurrent;
        float tWorldExit = tCurrent + stepSize;
        if (tWorldExit > tExitStack[level]) {
            tWorldExit = tExitStack[level];
        }
        iter_cnt = endlevel;
        uint currentViewIdx = triView[0];
        uint currentViewIdx2 = triView[1];
        uint currentViewIdx3 = triView[2];
        bool isHitCandidate = SingleCheckImpostorHit(
                                  impostor, 
            triView[0], 
            imgRay[0], 
            rayPosW, 
            rayDirW, 
            tWorldEnter, 
            tWorldExit, 
            level, depth_tanh, sampleUV[0]
            //                   ) || SingleCheckImpostorHit(
            //                       impostor, 
            // triView[1], 
            // imgRay[1], 
            // rayPosW, 
            // rayDirW, 
            // tWorldEnter, 
            // tWorldExit, 
            // level, depth_tanh, sample[1]
            //                   ) || SingleCheckImpostorHit(
            //                       impostor, 
            // triView[2], 
            // imgRay[2], 
            // rayPosW, 
            // rayDirW, 
            // tWorldEnter, 
            // tWorldExit, 
            // level, depth_tanh, sample[2]
                              );
        // sampleUV = sample[0];
        // return true;
        if (isHitCandidate)
        {
            if (level == endlevel) {
                hitDepthEnter = tWorldEnter;
                hitDepthExit = tWorldExit;
                compare_pos = float3(0);
                return true;
            }
            else {
                level--;
                tExitStack[level] = tWorldExit;
                continue;
            }
        }

        tCurrent = tWorldExit;
        endlevel = ComputePowerOfTwoLevel_NoBranch(tCurrent * depth_tanh, 1);

        while ((level < 9 && tCurrent >= tExitStack[level] - 1e-4) || level < endlevel) {
            level = level + 1;
        }
        if (level == 9 && tCurrent >= tExitStack[9]) {
            
            return false;
        }
    }
    return false;
}
float ConeSpanAlongDir(
    float3 axis, // normalized
    float tanTheta,
    float tNear,
    float tFar,
    float3 dir // normalized
) {
    float ad = dot(axis, dir);
    float perp = sqrt(max(0.0f, 1.0f - ad * ad));
    float span =
          abs(tFar - tNear) * abs(ad)
        + tanTheta * perp * (tFar + tNear);
    return span;
}


// 计算锥体投影到射线的 [t_min, t_max]
float2 GetConeProjectionInterval(
    float3 conePos, // 锥体位置 C
    float3 coneDir, // 锥体方向（非单位也行）
    float tanHalf,  // 半角 tan 值
    float tnear,    // 近截面距离
    float tfar,     // 远截面距离
    float3 rayPos,  // 射线起点
    float3 rayDir   // 射线方向（非单位也行）
)
{
    // 归一化方向
    float3 U = coneDir;
    float3 D = rayDir;

    // 基础偏移
    float3 diff = conePos - rayPos;
    float base = dot(diff, D);

    // 投影系数
    float ds = dot(U, D);
    float beta_sq = saturate(1.0 - ds * ds); // 防止浮点误差导致 NaN
    float beta = sqrt(beta_sq);

    float alpha = tanHalf;

    // 线性系数
    float gamma_plus = ds + alpha * beta;
    float gamma_minus = ds - alpha * beta;

    // 计算 t_max：上包络的最大值
    float s_max_plus = (gamma_plus >= 0.0) ? tfar : tnear;
    float t_max = base + gamma_plus * s_max_plus;

    // 计算 t_min：下包络的最小值
    float s_min_minus = (gamma_minus >= 0.0) ? tnear : tfar;
    float t_min = base + gamma_minus * s_min_minus;

    // 如果区间无效（t_min > t_max），可根据需要调整（例如返回 float2(0, 0) 或 INF）
    // 这里直接返回，调用方判断
    // return float2(base + gamma_minus * tnear, base + gamma_plus * tnear);
    //return float2(base - tnear * tanHalf, base + 1e-3 + tnear * tanHalf);
    return float2(t_min, t_max);
}


float3 AccStepRecursiveTraverseImpostorImageDDA(
    in Impostor impostor,
    uint3 triView,
    float3 rayPosW,
    float3 rayDirW,
    out float3 hitDepthEnter,
    out float3 hitDepthExit,
    out int3 hitSampleLevel,
    out float3 compare_pos,
    float tNear,
    float tFar,
    out uint iter_cnt,
    out float4 sampleUV[3],
    float depth_tanh
)
{
    // 1. 初始化所有 Output 参数 (防止未命中时返回垃圾值)
    hitDepthEnter = float3(1000, 1000, 1000);
    hitDepthExit = float3(1000, 1000, 1000);
    hitSampleLevel = int3(0, 0, 0);
    compare_pos = float3(0, 0, 0);
    iter_cnt = 0;
    sampleUV[0] = float4(0, 0, 0, 0);
    sampleUV[1] = float4(0, 0, 0, 0);
    sampleUV[2] = float4(0, 0, 0, 0);

    ImageRay imgRay[3];
    float3 isHit = float3(0, 0, 0);

    // 构建射线
    for (int i = 0; i < 3; i++) {
        if (!BuildImageRay2(impostor, triView[i], rayPosW, rayDirW, imgRay[i], 0)) {
            // 如果构建失败，该 View 保持 false，继续处理其他 View
            isHit[i] = 0;
        }
    }
    
    // 2. 定义步长常量
    // 目标精度 (最小步长)
    const float MIN_STEP_SCALE = 1.0 / 512.0;
    // 初始步长 (例如 16倍 于最小步长，可根据 DepthMap 的复杂度调整，太大会漏掉薄片)
    const float MAX_STEP_SCALE = 1.0 / 8.0;

    // 减少最大迭代次数，因为大步长不需要那么多步
    const int MAX_STEPS = 512;

    uint total_iters = 0;
    uint sampleLevel;
    //return isHit;
    for (int viewid = 0; viewid < 3; viewid++) {
        // 如果射线构建失败，跳过该 View
        // 注意：这里需要根据 BuildImageRay2 的结果判断，假设 imgRay[viewid] 数据无效时不应计算
        // 但由于原逻辑是 return isHit，这里我们假设如果构建失败不应进入循环
        // 这里做一个简单的防御性检查，或者你可以依赖之前的 return 逻辑

        float tCurrent = tNear;

        float3 camR = impostor.cRight[0][triView[viewid]];
        
        float3 camU = impostor.cUp[0][triView[viewid]];
        float3 camP = impostor.cPosition[0][triView[viewid]];
        float rView = impostor.cRadius[0][triView[viewid]];
        float rViewInv = 1.0 / impostor.cRadius[0][triView[viewid]];
        //isHit[viewid] = camR.z + camU.z + camP.z + rView + rViewInv;
        // // 当前视角的动态步长
        uint currentStepScale = 64;

        // 迭代计数
        int loop_iter = 0;
        
        for (loop_iter = 0; loop_iter < MAX_STEPS; ++loop_iter)
        {
            float currentStepSize = rView * MIN_STEP_SCALE * currentStepScale;

            // // 计算 LOD (保持原逻辑)
            float2 uinterval = GetConeProjectionInterval(rayPosW, rayDirW, depth_tanh, tCurrent, tCurrent + currentStepSize, camP, camR);
            float2 vinterval = GetConeProjectionInterval(rayPosW, rayDirW, depth_tanh, tCurrent, tCurrent + currentStepSize, camP, camU);
            
            uinterval = (uinterval * rViewInv + 1) * 0.5;
            vinterval = (vinterval * rViewInv + 1) * 0.5;
            // //debug

            bool isHitCandidate = SphereCheckImpostorHit(
                impostor,
                triView[viewid],
                imgRay[viewid],
                rayPosW,
                rayDirW,
                tCurrent,
                tCurrent + currentStepSize,
                sampleLevel,
                depth_tanh,
                sampleUV[viewid],
                uinterval, vinterval
            );
            uint nextStepScale = currentStepScale;
            if (isHitCandidate)
            {
                // [核心优化逻辑]：命中时的处理
                isHit[viewid] = 1;
                // 如果当前已经是最小精度，或者是起始点直接命中(防止回退越界)
                if (nextStepScale == 1)
                {
                    // 确认命中
                    hitDepthEnter[viewid] = tCurrent;
                    hitDepthExit[viewid] = tCurrent; // 简单 DDA 假设进出同点，如有厚度需另外处理
                    
                    hitSampleLevel[viewid] = sampleLevel;
                    sampleUV[viewid] = float4(uinterval.x, uinterval.y,vinterval.x,vinterval.y);
                    break; // 结束该 View 的循环
                }
                else
                {
                    nextStepScale = max(nextStepScale >> 1, 1);
                }
            }
            else
            {
                isHit[viewid] = 0;
                // 未命中：大胆前进
                tCurrent += currentStepSize;

                // // // 边界检查
                if (tCurrent > tFar) {
                    break;
                }
                if (nextStepScale < 64) { nextStepScale = nextStepScale << 1; }
            }
            currentStepScale = nextStepScale;
        }
        total_iters += loop_iter;
    }

    iter_cnt = total_iters;
    return isHit;
}

bool3 StepRecursiveTraverseImpostorImageDDA(
    in Impostor impostor,
    uint3 triView,
    float3 rayPosW,
    float3 rayDirW,
    out float3 hitDepthEnter,
    out float3 hitDepthExit,
    out int3 hitSampleLevel,
    out float3 compare_pos,
    float tNear,
    float tFar,
    out uint iter_cnt,
    out float4 sampleUV[3],
    float depth_tanh,
)
{
    ImageRay imgRay[3];
    bool3 isHit;
    for (int i = 0; i < 3; i++) {
        isHit[i] = false;
    }
    
    for (int i = 0; i < 3; i++) {
            if (!BuildImageRay2(impostor, triView[i], rayPosW, rayDirW, imgRay[i], 0)) {
                return isHit;
        }
    }

    
    
    const int MAX_STEPS = 1024;
    for (int viewid = 0; viewid < 3; viewid++) {
        float tCurrent = tNear;

        float3 camR = impostor.cRight[0][triView[viewid]];
        float3 camP = impostor.cPosition[0][triView[viewid]];
        float3 camU = impostor.cUp[0][triView[viewid]];
        float rView = impostor.cRadius[0][triView[viewid]];
        float rViewInv = 1.0 / impostor.cRadius[0][triView[viewid]];
        float stepScale = 1.0 / 512 ;
        uint sampleLevel;
        for (int iter = 0; iter < MAX_STEPS; ++iter)
        {
            // float uLenth = ConeSpanAlongDir(rayDirW, depth_tanh, tCurrent, tCurrent, camR);
            // float vLenth = ConeSpanAlongDir(rayDirW, depth_tanh, tCurrent, tCurrent, camU);
                float2 uinterval = GetConeProjectionInterval(rayPosW, rayDirW, depth_tanh, tCurrent, tCurrent , camP, camR);
                float2 vinterval = GetConeProjectionInterval(rayPosW, rayDirW, depth_tanh, tCurrent, tCurrent, camP, camU);
                uinterval = (uinterval * rViewInv + 1) * 0.5;
                vinterval = (vinterval * rViewInv + 1) * 0.5;
            //sampleLevel = ComputePowerOfTwoLevel_NoBranch(max(uLenth, vLenth), rView);

            bool isHitCandidate = SphereCheckImpostorHit(
                impostor, 
                triView[viewid], 
                imgRay[viewid], 
                rayPosW, 
                rayDirW, 
                tCurrent, 
                tCurrent, 
                sampleLevel, depth_tanh, sampleUV[viewid],uinterval,vinterval
            );
            if (isHitCandidate)
            {
                hitDepthEnter[viewid] = tCurrent;
                hitDepthExit[viewid] = tCurrent;

                isHit[viewid] = true;
                hitSampleLevel[viewid] = sampleLevel;
                break;
            }
            else {
                tCurrent = tCurrent + rView  * stepScale;
                if (tCurrent > tFar){
                    break;
                }
            }
        }
    }
    return isHit;
}
bool SphereConeOverlap(
    float3 conePos,      // 圆锥 apex
    float3 coneDir,      // 圆锥轴向（必须 normalized）
    float tanTheta,      // 圆锥半角 tan(theta)
    float3 sphereCenter, // 球心
    float sphereRadius,  // 球半径
    out float tNear,
    out float tFar
) {
    // ------------------------------------------------------------
    // Step 0: 转到圆锥局部
    // ------------------------------------------------------------
    float3 v = sphereCenter - conePos;

    float tc = dot(v, coneDir); // 球心在轴向的投影
    float v2 = dot(v, v);
    float d2 = v2 - tc * tc; // 到轴线的垂直距离^2
    float dPerp = sqrt(max(d2, 0.0));

    // ------------------------------------------------------------
    // Step 1: 快速剔除（球不可能碰到圆锥）
    // 圆锥侧面外扩一个球半径
    // ------------------------------------------------------------
    float expand = sphereRadius * sqrt(1.0 + tanTheta * tanTheta);

    // 完全在圆锥外
    if (dPerp > tc * tanTheta + expand)
        return false;

    // 完全在 apex 后方
    if (tc + sphereRadius <= 0.0)
        return false;

    // ------------------------------------------------------------
    // Step 2: 解球 ∩ 圆锥 的 t 区间
    //
    // (1 + tan^2) t^2 - 2 tc t + (tc^2 - r^2) >= 0
    // ------------------------------------------------------------
    float A = 1.0 + tanTheta * tanTheta;
    float B = -2.0 * tc;
    float C = tc * tc - sphereRadius * sphereRadius;

    float delta = B * B - 4.0 * A * C;

    // ------------------------------------------------------------
    // Step 3: 根据判别式求区间
    // ------------------------------------------------------------
    if (delta < 0.0)
    {
        // 球完全在圆锥内（或被完全包住）
        tNear = max(tc - sphereRadius, 0.0);
        tFar = tc + sphereRadius;
        return tNear <= tFar;
    }

    float sqrtD = sqrt(delta);
    float inv2A = 0.5 / A;

    float t1 = (-B - sqrtD) * inv2A;
    float t2 = (-B + sqrtD) * inv2A;

    // 球自身的轴向范围
    float sNear = tc - sphereRadius;
    float sFar = tc + sphereRadius;

    tNear = max(max(t1, sNear), 0.0);
    tFar = min(t2, sFar);

    return tNear <= tFar;
}

float4 sampleImpostorSingleViewLevel(uint2 pixel, in Impostor impostor, float3 rayDirW, float3 rayPosW, int level, float depth_tanh,int idx)
{
    init(pixel);
    float dirScale = length(rayDirW);
    float3 rayDirW = normalize(rayDirW);
    float tNear, tFar;
    float tNear2, tFar2;
    //return float4(1);
    // if (!RaySphereIntersectNearFar(rayPosW - impostor.centerWS[0], rayDirW, 1, tNear, tFar)) {
    //     return float4(0, 0, 0, 0);
    // }
    if (!SphereConeOverlap(
            rayPosW,
            rayDirW,
            depth_tanh,
            impostor.centerWS[0].xyz,
            1,
            tNear,
            tFar))
        return float4(0, 0, 0, 0);
    //return float4(tNear, 0,0,0);
    float minDepth = 1000;
    
    float4 result = float4(0, 0, 0, 0);
    uint viewIdx;

    float2 hitUV;
    float3 hitDepthEnterList = float3(1000);
    float3 hitDepthExitList = float3(1000);
    float3 compare_pos;
    uint3 triView;
    int3 hitSampleLevel= int3(0);
    triView = getNearestImpostorView(impostor, -normalize(rayDirW), 0);
    uint iter;
    float4 sampleUV[3];
    // return float4(1, 0, 0, 0);
    // if (SingleRecursiveTraverseImpostorImageDDA(
    float3 isHit = AccStepRecursiveTraverseImpostorImageDDA(
        impostor,
        triView,
        rayPosW,
        rayDirW,
        hitDepthEnterList,
        hitDepthExitList,
        hitSampleLevel,
        compare_pos,
        tNear,
        tFar,
        iter, sampleUV, depth_tanh );
    //return float4(isHit, 1);
    float hitDepthEnter = +1e30f; // 或 FLT_MAX
    bool anyHit = false;
    //return float4(hitDepthEnterList,0);
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        if (isHit[i]>0.5)
        {
            hitDepthEnter = min(hitDepthEnter, hitDepthEnterList[i]);
            anyHit = true;
        }
    }

    // return sampleUV;
    // gAlbedo0[0] = float4(hitDepthEnterList[0]);
    // return gAlbedo0[0];
    
    for (int i = 0; i < 3; i++) {
        // if (!isHit[i])
        //     continue;
        viewIdx = triView[i];

        // float hitDepthEnter = hitDepthEnterList[i];
        float hitDepthEnter = hitDepthEnter;
        //float hitDepthExit = hitDepthExitList[i];
        float3 hitposEnter = hitDepthEnter * (1) * rayDirW + rayPosW;
        //float3 hitposExit = hitDepthExit * (1) * rayDirW + rayPosW;
        float3 hitpos = hitDepthEnter * rayDirW + rayPosW;
        float hitRadius = depth_tanh * hitDepthEnter;
        float3 hitSphere = (hitDepthEnter + hitDepthEnter * depth_tanh) * rayDirW + rayPosW;
        
        float3 camPos = impostor.cPosition[0][viewIdx];
        float3 camR = impostor.cRight[0][viewIdx];
        float3 camU = impostor.cUp[0][viewIdx];
        float3 camF = impostor.cForward[0][viewIdx];
        float rView = impostor.cRadius[0][viewIdx];
        float3 delta = hitpos - camPos;
        float x = dot(delta, camR);
        float y = dot(delta, camU);
        float2 uv = float2(x, y) * (0.5 / rView) + 0.5;
        float depth =
                SingleSampleImpostorPoint(impostor.texDepthArray,
                                        impostor.samplerPoint[0],
                                        viewIdx,
                                        uv, hitSampleLevel[i]).r;

        float2 textureDepth2 = SampleImpostorDepth4Texels(
            impostor.texDepthArray[hitSampleLevel[i]],
            viewIdx,
            uv,
            hitSampleLevel[i],
            uint2(512, 512)
        );
        float uvLenth = hitDepthEnter * depth_tanh * (1.0 / rView) * 0.5;
        float2 textureDepth3;
        textureDepth3 = SmartMinMaxSample( impostor, viewIdx, float2(sampleUV[i].x, sampleUV[i].y), float2(sampleUV[i].z,sampleUV[i].w), float2(512, 512), level);

        float3 P_view = camPos +
                    (x * camR) +
                    (y * camU) +
                    (depth * camF);
        float3 P_view2 = camPos +
                    (x * camR) +
                    (y * camU) +
                    (textureDepth2.x * camF);
        float3 P_view3 = camPos +
                    (x * camR) +
                    (y * camU) +
                    (textureDepth3.x * camF);
        float max_interval = max(sampleUV[i].y - sampleUV[i].x, sampleUV[i].w - sampleUV[i].z) /2;
        float radius = max_interval * rView;
        float3 ConeScale = float3(textureDepth3.x, textureDepth3.y, radius);
        // if (length(hitpos - P_view) < 1 / 128 * 1) {
        //     result = float4(normal, 1);
        // }
        float2 midUV = float2(sampleUV[i].x + sampleUV[i].y, sampleUV[i].z + sampleUV[i].w) * 0.5;
        float lenth = max(sampleUV[i].y - sampleUV[i].x, sampleUV[i].w - sampleUV[i].z) / 2;
        float4 trueUV = float4(uv.x - uvLenth, uv.x + uvLenth, uv.y - uvLenth, uv.y + uvLenth);

        if (i == 0 && textureDepth2.x < 9) {
            gAlbedo0[idx][pixel] = trueUV;
            gDepth0[idx][pixel] = float4(ConeScale, hitDepthEnter);
            gDir0[idx][pixel] = float4(dirScale, viewIdx,0, viewIdx);

        }
        if (i == 0 && textureDepth2.x == 10) {
            gAlbedo0[idx][pixel] = float4(textureDepth2.x);
            gDepth0[idx][pixel] = float4(0);
        }
        if (i == 1 && textureDepth2.x < 9) {
            gAlbedo1[idx][pixel] = trueUV;
            gDepth1[idx][pixel] = float4(ConeScale, hitDepthEnter);
            gDir1[idx][pixel] = float4(dirScale, viewIdx,0, viewIdx);

        }
        if (i == 1 && textureDepth2.x == 10) {
            gAlbedo1[idx][pixel] = float4(textureDepth2.x);
            gDepth1[idx][pixel] = float4(0);
        }
        if (i == 2 && textureDepth2.x < 9) {
            gAlbedo2[idx][pixel] = trueUV;
            gDepth2[idx][pixel] = float4(ConeScale, hitDepthEnter);
            gDir2[idx][pixel] = float4(dirScale, viewIdx,0, viewIdx);
        }
        if (i == 2 && textureDepth2.x == 10) {
            gAlbedo2[idx][pixel] = float4(textureDepth2.x);
            gDepth2[idx][pixel] = float4(0);
        }
    }

    // return float4(iter);
    return float4(1);
}


bool RaySphereIntersect(float3 rayPosW, float3 rayDirW, float radius, out float tHit)
{
    float b = dot(rayDirW, rayPosW);
    float c = dot(rayPosW, rayPosW) - radius * radius;
    float discriminant = b * b - c;

    if (discriminant < 0)
    {
        tHit = 0;
        return false; // 没有交点
    }

    float sqrtD = sqrt(discriminant);
    float t0 = -b - sqrtD; // 最近交点
    float t1 = -b + sqrtD;

    // 选择第一个正交点
    tHit = (t0 > 0) ? t0 : ((t1 > 0) ? t1 : 0);

    return tHit > 0;
}

bool RaySphereIntersectNearFar(
    float3 rayPosW, // ray origin in sphere local space
    float3 rayDirW, // normalized ray direction
    float radius,
    out float tMin,
    out float tMax
)
{
    // b = dot(D, O)
    float b = dot(rayDirW, rayPosW);

    // c = |O|^2 - r^2
    float c = dot(rayPosW, rayPosW) - radius * radius;

    float discriminant = b * b - c;

    if (discriminant < 0.0)
    {
        tMin = 0.0;
        tMax = 0.0;
        return false; // no intersection
    }

    float sqrtD = sqrt(discriminant);

    // two roots
    float t0 = -b - sqrtD;
    float t1 = -b + sqrtD;

    // ensure ordering
    tMin = min(t0, t1);
    tMax = max(t0, t1);

    // 如果整个区间都在 ray 起点之后
    if (tMax < 0.0)
        return false;

    // clamp tMin 到 0（ray 从球内部出发的情况）
    if (tMin < 0.0)
        tMin = 0.0;

    return true;
}

void init(uint2 pixel) {
}





    void writeHit(uint2 pixel, float3 rayOrigin, float3 rayDir, const HitInfo hit)
    {
        
        gVBuffer[pixel] = hit.pack();

        VertexData v;
        float depth = 1.f;
        float3 prevPosW = float3(0.f);
        float2 motionVector = float2(0.f);

        if (hit.getType() == HitType::Triangle)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            v = gScene.getVertexData(triangleHit);
            prevPosW = gScene.getPrevPosW(triangleHit);
        }
        else if (hit.getType() == HitType::DisplacedTriangle)
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            v = gScene.getVertexData(displacedTriangleHit, -rayDir);
            prevPosW = gScene.getPrevPosW(displacedTriangleHit);
        }
        else if (hit.getType() == HitType::Curve)
        {
            const CurveHit curveHit = hit.getCurveHit();
            v = gScene.getVertexDataFromCurve(curveHit);
            prevPosW = gScene.getPrevPosWFromCurve(curveHit);
        }
        else if (hit.getType() == HitType::SDFGrid)
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);

            prevPosW = gScene.getPrevPosWFromSDFGrid(sdfGridHit, v.posW);
        }

        if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle || hit.getType() == HitType::Curve ||
            hit.getType() == HitType::SDFGrid)
        {
            // Compute depth similar to raster (NDC).
            float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(v.posW, 1.f));
            depth = curPosH.z / curPosH.w;

            // Compute motion vector.
            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
            // Remove camera jitter from motion vector
            motionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
        }

        float3 position = v.posW;
        if (is_valid(gDepth))
            gDepth[pixel] = depth;
        if (is_valid(gMotionVector))
            gMotionVector[pixel] = motionVector;
        if (is_valid(gMask))
            gMask[pixel] = 1.0f;
        uint3 gridDim = uint3(16, 16, 16);
        uint3 pos = uint3((position + 1) / 2 * 16);                  // 映射到 [-1,1]^3
        float3 color = hash3(pos.x * 16 * 16 + pos.y * 16 + pos.z);  // 自定义 hash 函数

    }
    
    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        gVBuffer[pixel] = {};

        if (is_valid(gDepth))
            gDepth[pixel] = 0.f;
        if (is_valid(gMotionVector))
            gMotionVector[pixel] = {};
        if (is_valid(gMask))
            gMask[pixel] = 0.0f;
        // if (is_valid(gViewW))
        //gViewW[pixel] = float4(-rayDir, 0.f);
    }

    void writeAux(uint2 pixel, const Ray ray)
    {
        // Write view direction.
        gViewW[pixel] = float4(-ray.dir, 0.f);
    }

    void writeAux(uint2 pixel, const float4 result)
    {
        // Write view direction.
        gViewW[pixel] = result;
    }
    void writeDebug(uint2 pixel, const float4 result)
    {
        // Write view direction.ww
    }
    void writeAuxImpostor(uint2 pixel, float4 result)
    {
        // Write view direction.
        gViewW[pixel] = result;
    }
    void beginTime(inout GpuTimer timer)
    {
        if (is_valid(gTime))
            timer.start();
    }

    void endTime(uint2 pixel, inout GpuTimer timer)
    {
        if (is_valid(gTime))
            gTime[pixel] = timer.getElapsed();
    }
};
