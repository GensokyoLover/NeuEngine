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
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float> gDepth;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gMask;
RWTexture2D<float4> gPosition;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float4> gAlbedo0;
RWTexture2D<float4> gAlbedo1;
RWTexture2D<float4> gAlbedo2;
RWTexture2D<float4> gDepth0;
RWTexture2D<float4> gWdepth;
RWTexture2D<float4> gDepth1;
RWTexture2D<float4> gDepth2;
RWTexture2D<float4> gReferenceAlbedo;
RWTexture2D<float4> gReferenceDepth;
RWTexture2D<float4> gDebug;

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
static float half_tan[101] = {0.0,   0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.011, 0.014, 0.017, 0.021, 0.025, 0.029, 0.034, 0.039, 0.045, 0.05, 0.057, 0.063, 0.07,  0.077, 0.085, 0.093, 0.102, 0.111, 0.12,  0.13,  0.141, 0.152, 0.163, 0.175, 0.187, 0.2,   0.214, 0.228, 0.243, 0.259, 0.275, 0.292, 0.31,  0.328, 0.348, 0.368, 0.389, 0.411, 0.435, 0.459, 0.484, 0.51,  0.538, 0.567, 0.597, 0.629, 0.661, 0.695, 0.731, 0.768, 0.807, 0.847, 0.889, 0.932, 0.977, 1.024, 1.072, 1.122, 1.173, 1.227, 1.282, 1.338, 1.397, 1.457, 1.519, 1.582, 1.647, 1.714, 1.783, 1.853, 1.924, 1.998, 2.072, 2.149, 2.227, 2.306, 2.387, 2.47, 2.554, 2.639, 2.726, 2.814, 2.904, 2.995, 3.087, 3.181, 3.276, 3.372, 3.47,  3.569, 3.669, 3.77,  3.873 };
struct VBufferRT
{
    static const bool kComputeDepthOfField = COMPUTE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kRayFlags = RAY_FLAGS;
    static const bool kMultiView = false;

    uint2 frameDim;
    uint frameCount;
    Impostor gImpostor;
    int roughness;
    
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
    if (!RaySphereIntersect(rayPosW - impostor.centerWS[level],
                            rayDirW,
                            impostor.radius[level],
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

void InitViewDDA(
    in Impostor impostor,
    uint viewIdx,
    ImageRay ray,
    float tEnter,
    out ViewDDA s,
    int level
)
{
    const int res = impostor.texDim[level].x;

    s.viewIdx = viewIdx;
    s.valid = true;

    float2 uvEnter = ray.uv0 ;
    s.uvEnter = uvEnter;

    float fx = uvEnter.x * res;
    float fy = uvEnter.y * res;

    s.ix = (int)floor(fx);
    s.iy = (int)floor(fy);

    if (s.ix < 0 || s.ix >= res || s.iy < 0 || s.iy >= res)
    {
            s.valid = false;
            return;
    }

    float du = ray.duv_dt.x;
    float dv = ray.duv_dt.y;

    s.stepX = (du > 0) ? 1 : -1;
    s.stepY = (dv > 0) ? 1 : -1;

    float nextU = (s.stepX > 0) ? ((s.ix + 1) / float(res))
                                : (s.ix / float(res));
    float nextV = (s.stepY > 0) ? ((s.iy + 1) / float(res))
                                : (s.iy / float(res));

    s.tMaxX = (abs(du) > 1e-10) ? (nextU - uvEnter.x) / du : 1e30;
    s.tMaxY = (abs(dv) > 1e-10) ? (nextV - uvEnter.y) / dv : 1e30;

    s.tDeltaX = (abs(du) > 1e-10) ? abs(1.0 / (du * res)) : 1e30;
    s.tDeltaY = (abs(dv) > 1e-10) ? abs(1.0 / (dv * res)) : 1e30;

    s.tMaxX = max(s.tMaxX, 0.0);
    s.tMaxY = max(s.tMaxY, 0.0);
}

bool TraverseImpostorImageDDA(
    in Impostor impostor,
    uint3 triView,
    float3 rayPosW,
    float3 rayDirW,
    out float hitDepth,
    float tNear,
    float tFar,
    int level,
)
{
    ImageRay imgRay[3];
    ViewDDA view[3];
    for (int i = 0; i < 3; i++) {
            if (!BuildImageRay(impostor, triView[i], rayPosW, rayDirW, imgRay[i], tNear, level))
            return false;
        InitViewDDA(impostor, triView[i], imgRay[i], tNear, view[i],level);
    }
 
    const float eps = 1e-6;

    const int MAX_PIXEL_STEPS = 1024 * 3;
    float t = tNear;
    float tEnter = tNear;
    float tExit = tFar;
    for (int iter = 0; iter < MAX_PIXEL_STEPS; ++iter)
    {
        float tNext = 1e30;
        int nextView = -1;
        bool stepX = false;

        for (int i = 0; i < 3; ++i)
        {
            if (!view[i].valid) continue;

            if (view[i].tMaxX < tNext)
            {
                tNext = view[i].tMaxX;
                nextView = i;
                stepX = true;
            }
            if (view[i].tMaxY < tNext)
            {
                tNext = view[i].tMaxY;
                nextView = i;
                stepX = false;
            }
        }


        float tWorldEnter = t;
        float tWorldExit = tEnter + tNext;
        float tMid = 0.5 * (tWorldEnter + tWorldExit);                   
        if (tWorldEnter > tExit) return false;
        ViewDDA s = view[nextView];
        ImageRay ray = imgRay[nextView];
        // ✅ 代表点对应的 uv（严格沿 ray 的投影线）
        float2 uv =
            s.uvEnter + ray.duv_dt * (tMid - tEnter);

        float viewDepth =
        SampleImpostorPoint(
            impostor.texDepthArray,
            impostor.samplerPoint[level],
            s.viewIdx,
            uv, level).r;
        float3 camPos = impostor.cPosition[level][s.viewIdx];
        float3 camR = impostor.cRight[level][s.viewIdx];
        float3 camU = impostor.cUp[level][s.viewIdx];
        float3 camF = impostor.cForward[level][s.viewIdx];
        float rView = impostor.cRadius[level][s.viewIdx];
        float depthRayEnter = dot(
            (rayPosW + tWorldEnter * rayDirW) - camPos,
            camF
        );

        float depthRayExit = dot(
            (rayPosW + tWorldExit * rayDirW) - camPos,
            camF
        );
        float rayMinDepth = min(depthRayEnter, depthRayExit);
        float rayMaxDepth = max(depthRayEnter, depthRayExit);
        if (viewDepth<9 )
        {
            float rView = impostor.cRadius[level][s.viewIdx];

            // 用 uv 重建 view 平面上的 x,y（与你原先的定义一致）
            float x = (uv.x * 2.0 - 1.0) * rView;
            float y = (uv.y * 2.0 - 1.0) * rView;

            float3 Pview = camPos + x * camR + y * camU + viewDepth * camF;

            float3 samplePos = rayPosW + tMid * rayDirW;
            float dist = length(samplePos - Pview);

            // TODO：把这个阈值换成你原来那套更严谨的判定
            if (dist < impostor.radius[level] * pow(2, level) / 128.0)
            {
                hitDepth = tMid;
                //hitDepth = iter;
                return true;
            }
        }

        // ---------------------------------------
        // 4) 前进到下一个 cell：比较 tMaxX 和 tMaxY
        // ---------------------------------------
        if (stepX)
        {
            view[nextView].ix += view[nextView].stepX;
            view[nextView].tMaxX += view[nextView].tDeltaX;
        }
        else
        {
            view[nextView].iy += view[nextView].stepY;
            view[nextView].tMaxY += view[nextView].tDeltaY;
        }

        // 越界则该 view 失效
        if (view[nextView].ix < 0 || view[nextView].ix >= impostor.texDim[level].x ||
        view[nextView].iy < 0 || view[nextView].iy >= impostor.texDim[level].x)
        {
            view[nextView].valid = false;
        }

        t = tWorldExit;

        // 处理极端：如果 du/dv 很小，tDelta 会非常大，导致 tWorld 直接超过 tExit
            // 上面已经用 tWorld>tExit 退出了
    }

    return false;
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
    float2 nextBoundaryPixel = (duvdt > 0.0) ? (iPos + 1.0) : iPos;

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
    float threshold = (impostor.radius[0] / 128.0) * float(1 << level) ;

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
    
    if (textureDepth.x - eps_min < rayDepthMax && rayDepthMin - (impostor.radius[0] / 128.0) < textureDepth.y + eps_max)
    {
        return true;
    }
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
    float inv_scale = (tCurrent * depth_tanh) / impostor.radius[0] ;
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
        inv_scale = (tCurrent * depth_tanh) / impostor.radius[0];
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
float4 sampleImpostorMarching(uint2 pixel, in Impostor impostor, float3 rayDirW, float3 rayPosW, int level, float depth_tanh)
{
    init(pixel);
    float tNear, tFar;
    if (!RaySphereIntersectNearFar(rayPosW - impostor.centerWS[level], rayDirW, impostor.radius[level], tNear, tFar))
        return float4(0, 0, 0, 0);

    float minDepth = 1000;

    float4 result = float4(0, 0, 0, 0);
    uint viewIdx;

    float2 hitUV;
    float hitDepthEnter;
    float hitDepthExit;
    float3 compare_pos;
    uint3 triView;
    for (int l = 0; l < 5; l++) {
        triView = getNearestImpostorView(impostor, -normalize(rayDirW), 0);
    }
    float iter;
    float4 sampleUV;
    //return float4(1, 0, 0, 0);
    if (RecursiveTraverseImpostorImageDDA(
            impostor,
                triView,
                rayPosW,
                rayDirW,
                hitDepthEnter,
                hitDepthExit,
                compare_pos,
                tNear,
                tFar,
            iter, sampleUV, depth_tanh))
    {
        float3 hitposEnter = hitDepthEnter * (1) * rayDirW + rayPosW;
        float3 hitposExit = hitDepthExit * (1) * rayDirW + rayPosW;
        float3 hitpos = (hitDepthEnter + hitDepthExit) / 2 * rayDirW + rayPosW;
        //return float4(hitposEnter, hitDepthEnter);
        //return float4(iter);
        for (int i = 0; i < 3; i++) {

            viewIdx = triView[0][i];
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
                    SampleImpostorPoint(impostor.texDepthArray,
                                         impostor.samplerPoint[0],
                                         viewIdx,
                                         uv, 0).r;
            float3 P_view = camPos +
                        (x * camR) +
                        (y * camU) +
                        (depth * camF);
            if (length(hitpos - P_view) < impostor.radius[0] / 128 * 1) {
                result =
                    SampleImpostorPoint(impostor.texNormalArray,
                                         impostor.samplerPoint[0],
                                         viewIdx,
                                         uv, 0);
            }
        }

        //result = float4(hitposEnter, 1);
        gDepth0[pixel] = float4(hitposEnter, 1);
        gDepth1[pixel] = float4(hitposExit, 1);
        gDepth2[pixel] = float4(hitDepthEnter, hitDepthExit, 0, 1);
        gDebug[pixel] = float4(depth_tanh);
    }
    else {
        gDepth0[pixel] = float4(0);
        gDepth1[pixel] = float4(0);
        gDepth2[pixel] = float4(0);
        gDebug[pixel] = float4(depth_tanh);
    }
    //return float4(iter);
    gDebug[pixel] = float4(depth_tanh);
    return result;
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
    gAlbedo0[pixel] = float4(0);
    gAlbedo1[pixel] = float4(0);
    gAlbedo2[pixel] = float4(0);
    gDepth0[pixel] = float4(0);
    gDepth1[pixel] = float4(0);
    gDepth2[pixel] = float4(0);
}

float4 sampleImpostorAccurate(uint2 pixel, in Impostor impostor, float3 rayDirW, float3 rayPosW, int level)
{
    init(pixel);
    float tNear, tFar;
    if (!RaySphereIntersectNearFar(rayPosW - impostor.centerWS[level], rayDirW, impostor.radius[level], tNear, tFar))
        return float4(0, 0, 0, 0);

    float minDepth = 1000;

    float4 result = float4(0, 0, 0, 0);
    uint viewIdx;

    float2 hitUV;
    float hitDepth;
    uint3 triView = getNearestImpostorView(impostor, -normalize(rayDirW), level);
    if (TraverseImpostorImageDDA(
                impostor,
                triView,
                rayPosW,
                rayDirW,
                hitDepth,
                tNear,
                tFar,
            level))
    {
        //return float4(1, 1, 1, 1);
        float3 hitpos = hitDepth * rayDirW + rayPosW;
        for (int i = 0; i < 3; i++) {
            viewIdx = triView[i];
            float3 camPos = impostor.cPosition[level][viewIdx];
            float3 camR = impostor.cRight[level][viewIdx];
            float3 camU = impostor.cUp[level][viewIdx];
            float3 camF = impostor.cForward[level][viewIdx];
            float rView = impostor.cRadius[level][viewIdx];
            float3 delta = hitpos - camPos;
            float x = dot(delta, camR);
            float y = dot(delta, camU);
            float2 uv = float2(x, y) * (0.5 / rView) + 0.5;
            float depth =
                    SampleImpostorPointAtlas(impostor.texDepthAtlas[0],
                                         impostor.samplerPoint[0],
                                         viewIdx,
                                         uv, 0).r;
            float3 P_view = camPos +
                        (x * camR) +
                        (y * camU) +
                        (depth * camF);
            if (length(hitpos - P_view) < impostor.radius[level] * pow(2.828,level) / 128) {
                result =
                    SampleImpostorPoint(impostor.texNormalArray,
                                         impostor.samplerPoint[0],
                                         viewIdx,
                                         uv, 0);
            }
            
            
        }
    }
    return result;
}


// float4 sampleImpostorMarching( uint2 pixel, in Impostor impostor, in Impostor impostor2, in Impostor impostor3, in Impostor impostor4, float3 rayDirW, float3 rayPosW)
// {
//     init(pixel);
//     float tNear, tFar;
//     if (!RaySphereIntersectNearFar(rayPosW - impostor.centerWS, rayDirW, impostor.radius, tNear, tFar))
//         return float4(0, 0, 0, 0);

//     float minDepth = 1000;

//     float4 result = float4(0, 0, 0, 0);
//     uint viewIdx;

//     // return float4(imgRay.tEnter);
//     float2 hitUV;
//     float hitDepth;
//     uint3 triView[4];
    
//     triView[0]= getNearestImpostorView(impostor, -normalize(rayDirW));
//     triView[1]= getNearestImpostorView(impostor2, -normalize(rayDirW));
//     triView[2]= getNearestImpostorView(impostor3, -normalize(rayDirW));
//     triView[3]= getNearestImpostorView(impostor4, -normalize(rayDirW));
//     if (RecursiveTraverseImpostorImageDDA(
//                 impostor,
//                 impostor2,
//                 impostor3,
//                 impostor4,
//                 triView,
//                 rayPosW,
//                 rayDirW,
//                 hitDepth,
//                 tNear,
//                 tFar,3))
//     {
//         // return float4(hitDepth, hitDepth, hitDepth, 1);
//         float3 hitpos = hitDepth * rayDirW + rayPosW;
//         for (int i = 0; i < 3; i++) {
//             viewIdx = triView[0][i];
//             float3 camPos = impostor[0].cPosition[viewIdx];
//             float3 camR = impostor[0].cRight[viewIdx];
//             float3 camU = impostor[0].cUp[viewIdx];
//             float3 camF = impostor[0].cForward[viewIdx];
//             float rView = impostor[0].cRadius[viewIdx];
//             float3 delta = hitpos - camPos;
//             float x = dot(delta, camR);
//             float y = dot(delta, camU);
//             float2 uv = float2(x, y) * (0.5 / rView) + 0.5;
//             float depth =
//                     SampleImpostorPoint(impostor[0].texDepth,
//                                          impostor[0].samplerPoint,
//                                          viewIdx,
//                                          uv).r;
//             float3 P_view = camPos +
//                         (x * camR) +
//                         (y * camU) +
//                         (depth * camF);
//             if (length(hitpos - P_view) < impostor[0].radius / 128) {
//                 result =
//                     SampleImpostorPoint(impostor[0].texNormal,
//                                          impostor[0].samplerPoint,
//                                          viewIdx,
//                                          uv);
//             }
//         }
//     }
//     return result;
// }


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
        gPosition[pixel] = float4(position, 1.f);
        uint3 gridDim = uint3(16, 16, 16);
        uint3 pos = uint3((position + 1) / 2 * 16);                  // 映射到 [-1,1]^3
        float3 color = hash3(pos.x * 16 * 16 + pos.y * 16 + pos.z);  // 自定义 hash 函数
        gAlbedo[pixel] = float4(color, 1.0);
        gWdepth[pixel].r = length(position - rayOrigin);

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
        gPosition[pixel] = float4(0);
        gAlbedo[pixel] = float4(0);
        gWdepth[pixel].r = 0;
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
        // Write view direction.
        gDebug[pixel] = result;
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
