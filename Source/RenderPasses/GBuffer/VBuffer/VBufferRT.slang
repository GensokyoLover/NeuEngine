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

struct VBufferRT
{
    static const bool kComputeDepthOfField = COMPUTE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kRayFlags = RAY_FLAGS;
    static const bool kMultiView = false;

    uint2 frameDim;
    uint frameCount;
    Impostor gImpostor;
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


    float2 OctEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
    return n.xy * 0.5 + 0.5;
}

uint3 getNearestImpostorView(in Impostor impostor,float3 dir)
{
    float2 uv = OctEncode(dir); // [-1,1] → [0,1]
    uint viewIdx = impostor.texFaceIndex.Load(int3(uv * impostor.baseCameraResolution, 0));

    // 然后直接采样对应层的 impostor 纹理
    return impostor.cFace[viewIdx];
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

float4 sampleImpostor(in Impostor impostor, float3 rayDirW, float3 rayPosW)
{
    
    float3 planePoint = impostor.centerWS;
    float3 planeNormal = normalize(planePoint - rayPosW);

    float denom = dot(rayDirW, planeNormal);
    if (abs(denom) < 1e-5) return float4(0, 0, 0, 0); // 平行，无交点

    float t = dot(planePoint - rayPosW, planeNormal) / denom;


    //return float4(denom,0,0,1); 
    float3 hitPosW = rayPosW + t * rayDirW;
    uint3 triCameraIndex = getNearestImpostorView(impostor, -planeNormal); 

    float invRadius2 = 0.5f / impostor.radius; // 用于将 [-radius, radius] 映射到 [-0.5, 0.5]

    // 用于存储每个视图的采样结果（Albedo 和 Depth）
    float4 samples[3];
    float weights[3] = { 0, 0, 0 }; // 混合权重
    float maxWeight = -1.0;
    uint bestIdx = 0;
    float sigma = 0.2;
    // ----------------------------------------------------
    // 【遍历三个相机，计算纹理坐标、深度和权重】
    // ----------------------------------------------------
    for (int i = 0; i < 3; ++i)
    {
        uint viewIdx = triCameraIndex[i];

        // 获取当前相机的位置、右向量和上向量
        float3 camPos = impostor.cPosition[viewIdx];
        float3 camR = impostor.cRight[viewIdx];
        float3 camU = impostor.cUp[viewIdx];
        float3 camF = impostor.cForward[viewIdx]; // 相机前向

        // 1. 将世界空间命中点投影到相机平面，计算相对位移
        float3 delta = hitPosW - camPos;
        
        // 2. 计算局部坐标 (x, y)
        // x = 沿右方向的距离 (世界空间)
        // y = 沿上方向的距离 (世界空间)
        float x = dot(delta, camR);
        float y = dot(delta, camU);

        // 3. 计算二维纹理坐标 UV [0, 1]
        // 假设投影区域是 [-radius, radius]
        float2 uv = float2(x, y) * invRadius2 + 0.5; // [-radius, radius] -> [0, 1]
        //return float4(uv.x, uv.y, 0, 1);
        // 4. 边界检查：如果 UV 超出 [0, 1] 范围，则此视图无效
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        {
            samples[i] = float4(0, 0, 0, 0);
            weights[i] = 0.0;
            continue;
        }

        // 5. 采样深度纹理（获取视图空间深度）
        // 这里的 texDepth.w 应该存储视图空间深度（例如：从相机位置到顶点的距离）
        float4 depthSample = impostor.texDepth.SampleLevel(impostor.samplerLinear, float3(uv, viewIdx), 0);
        float viewDepth = depthSample.r;
        // 深度检查：如果采样到的深度小于相机到命中点的距离，说明命中点在物体后面，此采样无效

        // 注意：Impostor 的深度通常是从相机到物体的距离，
        // 如果 distToHit > viewDepth (加上一个小的容差)，说明 rayDirW 命中了平面，但深度信息显示物体在这个位置更近。
        // 但这里我们只关心深度是否有效，一般用一个简单的深度容差来判断。
        // 一个更鲁棒的检查：检查命中点是否落在物体的表面上。
        if (viewDepth == 0 )
        {
            // 命中点在 impostor 物体之外或后面
            samples[i] = float4(0, 0, 0, 0);
            weights[i] = 0.0;
            continue;
        }
        float distToHit = dot(delta, camF); // 沿相机前向的距离（视图空间 Z）
        //return float4(distToHit, 0, 0, 1);
        float3 P_view = camPos + 
                    (x * camR) + 
                    (y * camU) + 
                    (viewDepth * camF);
        float3 vec_ray = P_view - rayPosW;
        float corrected_depth = dot(vec_ray, rayDirW);
        // 6. 采样 Albedo 纹理
        samples[i] = impostor.texAlbedo.SampleLevel(impostor.samplerLinear, float3(uv, viewIdx), 0);
        samples[i].w = corrected_depth;

        samples[i].rgb = P_view;
        float3 camera_to_p = P_view - camPos;
        float theta =  acos(saturate(dot(normalize(camera_to_p), rayDirW)));
        // 7. 计算权重：基于视线方向和相机前向的夹角
        // 夹角越小，权重越大。使用 dot(A, B) 衡量相似度。
        float camera2camera_distance = length(impostor.cPosition[viewIdx] - rayPosW);
        float invD0 = 1.0 / (camera2camera_distance + 1e-5); 
        invD0 = saturate(invD0);               // 确保在 [0, 1] 范围内

        // 可以使用 pow(NdotV, power) 来锐化权重
        weights[i] = exp(-theta * theta / (2.0 * sigma * sigma));; // 原始权重
        //weights[i] = invD0; // 原始权重
        //return float4(NdotV,0,0,1);
        // 记录权重最大的视图
        if (weights[i] > maxWeight)
        {
            maxWeight = weights[i];
            bestIdx = viewIdx;
        }
    }
    // /return float4(weights[0], weights[1], weights[2],1);
    // ----------------------------------------------------
    // 【功能一：选择最邻近的视图进行采样】
    // ----------------------------------------------------
    // 如果只需要最近的视图，并且找到了一个有效采样
    if (maxWeight > 0.0 && false) // 禁用此分支以运行混合功能
    {
        // 找到权重最大的有效采样的索引 i (0, 1, or 2)
        // 注意：bestIdx 存储的是全局 viewIdx，需要找到它在 triCameraIndex 中的位置
        int bestLocalIdx = (triCameraIndex[0] == bestIdx) ? 0 : 
                            (triCameraIndex[1] == bestIdx) ? 1 : 2;

        return samples[bestLocalIdx]; // 返回权重最大的采样结果
    }

    // ----------------------------------------------------
    // 【功能二：对三个视图进行权重混合】
    // ----------------------------------------------------
    float totalWeight = weights[0] + weights[1] + weights[2];

    if (totalWeight < 1e-5)
    {
        // 所有视图采样都无效（例如：超出边界或深度失败）
        return float4(0, 0, 0, 0);
    }

    // 归一化权重
    float invTotalWeight = 1.0 / totalWeight;
    weights[0] *= invTotalWeight;
    weights[1] *= invTotalWeight;
    weights[2] *= invTotalWeight;

    // 混合颜色（Albedo）
    float4 blendedColor = samples[0] * weights[0] +
                          samples[1] * weights[1] +
                          samples[2] * weights[2];

    // TODO: 如果需要，也混合法线 (需要先从 Normal 纹理采样，并将其转换为世界空间)

    return blendedColor;
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

void init(uint2 pixel) {
    gAlbedo0[pixel] = float4(0);
    gAlbedo1[pixel] = float4(0);
    gAlbedo2[pixel] = float4(0);
    gDepth0[pixel] = float4(0);
    gDepth1[pixel] = float4(0);
    gDepth2[pixel] = float4(0);
}

float4 sampleImpostorAccurate(uint2 pixel, in Impostor impostor, float3 rayDirW, float3 rayPosW)
{
    init(pixel);
    bool debug = true;
    
    
    float3 planePoint = impostor.centerWS;
    float3 planeNormal = normalize(planePoint - rayPosW);

    float denom = dot(rayDirW, planeNormal);
    if (abs(denom) < 1e-5) return float4(0, 0, 0, 0); // 平行，无交点

    float t = dot(planePoint - rayPosW, planeNormal) / denom;

    // return float4(denom,0,0,1);
    float3 hitPosW = rayPosW + t * rayDirW;
    uint3 triCameraIndex = getNearestImpostorView(impostor, -normalize(rayDirW));

    float4 samples[3] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
    float weights[3] = { 0, 0, 0 };
    float sigma = 0.5;
    int view_resolution = gImpostor.texDim.x;
    const int N = 512;                                    // 光线上采样步数
    const float step = 2.0 * impostor.radius / float(N);  // 步长 [-radius, radius]
    int reference_view = triCameraIndex[pixel.x / view_resolution];
    if (gScene.camera.data.debugPixel.x == 0 && gScene.camera.data.debugPixel.y == 0) {
        debug = false;
    }
    else {
        const Ray debugRay = generateRay(gScene.camera.data.debugPixel);
        uint3 debugTriCameraIdx = getNearestImpostorView(impostor, -debugRay.dir);
        int debugReference = debugTriCameraIdx[pixel.x / view_resolution];
        int rx = pixel.x % view_resolution;
        int ry = pixel.y % view_resolution; 
        uint2 reference_pixel = uint2(rx, ry);

        if (pixel.y / view_resolution == 1 && pixel.x / view_resolution < 3) {
            float4 depthSample = impostor.texDepth.SampleLevel( impostor.samplerLinear, float3((float2(rx, view_resolution - ry) + 0.5) * impostor.invTexDim, debugReference), 0 );
            gReferenceDepth[pixel] = depthSample;
            gReferenceDepth[pixel].w = 1;
            gReferenceAlbedo[pixel] = impostor.texAlbedo.SampleLevel( impostor.samplerLinear, float3((float2(rx, view_resolution - ry) + 0.5) * impostor.invTexDim, debugReference), 0 );
            gReferenceAlbedo[pixel].w = 1;
        }
    }
    // 遍历三个视图
    float minDepth = 100;
    uint final_idx = 0;
    float4 theta_final = 0;
    for (int i = 0; i < 3; ++i)
    {
        uint viewIdx = triCameraIndex[i];
        float3 camPos = impostor.cPosition[viewIdx];
        float3 camR = impostor.cRight[viewIdx];
        float3 camU = impostor.cUp[viewIdx];
        float3 camF = impostor.cForward[viewIdx];
        float viewRadius = impostor.cRadius[viewIdx];
        float bestDepthDiff = 1e10;
        float3 bestPosW = float3(0, 0, 0);
        float2 bestUV = float2(0, 0);
        bool notFound = false;
        bool hitFound = false;
        int final_step;
        float tStart;
        if (!RaySphereIntersect(rayPosW - impostor.centerWS, rayDirW, impostor.radius, tStart))
        {
            return float4(0, 0, 0, 0); // 光线未穿过 impostor 球
        }
        // 沿光线线性搜索
        float3 searchStart = rayPosW + tStart * rayDirW;
        float viewDepth;
        float camera_depth;
        float basetheta = 100;
        for (int s = 0; s < N; ++s)
        {
            float t = step * (s + 0.5);
            float3 samplePos = searchStart + t * rayDirW;
            float3 delta = samplePos - camPos;
            camera_depth = length(delta);
            float x = dot(delta, camR);
            float y = dot(delta, camU);
            float2 uv = float2(x, y) * (0.5 / viewRadius) + 0.5;

            if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
                continue;

            viewDepth = impostor.texDepth.SampleLevel(impostor.samplerLinear, float3(uv, viewIdx), 0).r;
            if (debug && pixel.x == gScene.camera.data.debugPixel.x && pixel.y == gScene.camera.data.debugPixel.y) {
                int2 viewWritePixel = int2(uv * view_resolution);
                //viewWritePixel = int2(viewWritePixel.y, viewWritePixel.x);
                viewWritePixel.x = viewWritePixel.x + view_resolution * i;
                viewWritePixel.y = view_resolution-viewWritePixel.y + view_resolution;
                gDebug[viewWritePixel] = float4(1);
                int2 writePixel = int2(s, i);

                gDebug[writePixel].r = viewDepth;
                gDebug[writePixel].g = camera_depth;
                gDebug[writePixel].b = int(uv.x * view_resolution);
                gDebug[writePixel].w = int(view_resolution - uv.y * view_resolution);

            }
            if (viewDepth == 0 || (viewDepth >= (camera_depth -1e-4))) continue;
            float3 P_view = camPos + 
                    (x * camR) + 
                    (y * camU) + 
                    (viewDepth * camF);
            float3 camera_to_p = bestPosW - samplePos;
            float theta = length(camera_to_p);
            if (basetheta>theta) {
                basetheta = theta;
                hitFound = true;
                bestUV = uv;
                bestPosW = P_view;
                final_step = s;
            }


        }

        if (!hitFound) {
            float4 result = float4(0, 0, 0, 0);
            return result;
        }

        // 采样颜色
        float4 albedo = impostor.texAlbedo.SampleLevel(impostor.samplerLinear, float3(bestUV, viewIdx), 0);

        // samples[i].w = length(bestPosW - rayPosW); // 存储光线方向距离
        // samples[i].rgb = bestPosW;
        float3 samplePos = searchStart + step * (final_step + 0.5) * rayDirW;  

            weights[i] = exp(-basetheta * basetheta / (2.0 * sigma * sigma));
        samples[i].rgb = albedo.rgb;

        if (basetheta < minDepth) {
            minDepth = basetheta;
            final_idx = i;
        }
        theta_final[i] = basetheta;
        if (i == 0) {
            gAlbedo0[pixel] = albedo;
            gDepth0[pixel].r = basetheta;
        }
        if (i == 1) {
            gAlbedo1[pixel] = albedo;
            gDepth1[pixel].r = basetheta;
        }
        if (i == 2) {
            gAlbedo2[pixel] = albedo;
            gDepth2[pixel].r = basetheta;
        }
    }
    if (minDepth > 0.01) {
        return float4(0, 0, 0, 0);
    }
    // theta_final.w = 1;
    // theta_final.r = min(min(theta_final.r, theta_final.g), theta_final.b);
    // theta_final.g = 0;
    // theta_final.b = 0;
    // return theta_final;
    // 权重归一化与混合
    float totalWeight = weights[0] + weights[1] + weights[2];
    if (totalWeight < 1e-5) return float4(0, 0, 0, 0);

    float invTotal = 1.0 / totalWeight;
    weights[0] *= invTotal;
    weights[1] *= invTotal;
    weights[2] *= invTotal;

    float4 blended = samples[0] * weights[0] +
                     samples[1] * weights[1] +
                     samples[2] * weights[2];
    blended = samples[final_idx];
    return blended;
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
        gPosition[pixel] = float4(position, 1.f);
        uint3 gridDim = uint3(16, 16, 16);
        uint3 pos = uint3((position + 1) / 2 * 16);                  // 映射到 [-1,1]^3
        float3 color = hash3(pos.x * 16 * 16 + pos.y * 16 + pos.z);  // 自定义 hash 函数
        gAlbedo[pixel] = float4(color, 1.0);
        gWdepth[pixel].r = length(position - rayOrigin);
        gViewW[pixel] = float4(-rayDir, 0.f);
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
        gViewW[pixel] = float4(-rayDir, 0.f);
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
