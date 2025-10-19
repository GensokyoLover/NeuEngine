/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Camera.CameraData;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;
struct Camera
{
    CameraData data;

    float3 getPosition() { return data.posW; }
    float4x4 getViewProj() { return data.viewProjMat; }

    /** Computes a camera ray for a given pixel assuming a pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter true if jitter should be applied else false.
        \return Returns the camera ray.
    */
    Ray computeRayPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        Ray ray;

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.origin = data.posW;
        ray.dir = normalize(computeNonNormalizedRayDirPinhole(pixel, frameDim, applyJitter));

        float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
        ray.tMin = data.nearZ * invCos;
        ray.tMax = data.farZ * invCos;

        return ray;
    }

    Ray computeRayOrthographic(uint2 pixel, uint2 frameDim, bool applyJitter = false)
    {
        Ray ray;

        // 1. 相机基向量
        float3 right = data.cameraU;          // 水平基向量
        float3 up = data.cameraV;             // 垂直基向量
        float3 fwd = normalize(data.cameraW); // 相机朝向

        // 2. 像素坐标归一化到 [-1,1]
        float2 ndc = (float2(pixel) + 0.5f) / float2(frameDim);
        ndc = ndc * 2.0f - 1.0f; // 左下 -1,-1 右上 +1,+1

                // 4. 映射到正交平面上的世界坐标
        // 假设相机的正交视口宽高由 data.orthoWidth / data.orthoHeight 决定
        float3 worldPos = data.posW + ndc.x * right * (data.frameWidth * 0.5f)
                                 + ndc.y * up * (data.frameHeight * 0.5f);

        // 5. 光线起点 = 平面上该点，方向恒为相机前向
        ray.origin = worldPos;
        ray.dir = fwd;

        // 6. near/far 直接为固定值，不乘角度校正
        ray.tMin = data.nearZ;
        ray.tMax = data.farZ;

        return ray;
    }
    /** Computes the primary ray's direction, non-normalized assuming pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter True if jitter should be applied else false.
        \return Returns the non-normalized ray direction
    */
    float3 computeNonNormalizedRayDirPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim;
        if (applyJitter) p += float2(-data.jitterX, data.jitterY);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the non-normalized ray direction assuming a pinhole camera.
        return ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
    }


    /** Computes a camera ray for a given pixel assuming a thin-lens camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] u Uniform 2D sample.
        \return Returns the camera ray.
    */
    Ray computeRayThinlens(uint2 pixel, uint2 frameDim, float2 u)
    {
        Ray ray;

        // Sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim + float2(-data.jitterX, data.jitterY);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the normalized ray direction assuming a thin-lens camera.
        ray.origin = data.posW;
        ray.dir = ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
        float2 apertureSample = sample_disk(u); // Sample lies in the unit disk [-1,1]^2
        float3 rayTarget = ray.origin + ray.dir;
        ray.origin += data.apertureRadius * (apertureSample.x * normalize(data.cameraU) + apertureSample.y * normalize(data.cameraV));
        ray.dir = normalize(rayTarget - ray.origin);

        float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
        ray.tMin = data.nearZ * invCos;
        ray.tMax = data.farZ * invCos;

        return ray;
    }

    float3 GeneratePositionByUV(float2 uv) {
        // 将 [0, 1] 区间的 UV 坐标映射到 [-1, 1] 区间
        float u = 2.0 * uv.x - 1.0;
        float v = 2.0 * uv.y - 1.0;

        // 计算绝对值
        float up = abs(u);
        float vp = abs(v);

        // 计算距离
        float signedDistance = 1.0 - (up + vp);
        float d = abs(signedDistance);
        float r = 1.0 - d;

        // 计算角度 phi
        float phi;
        if (r == 0.0) {
            phi = 1.0 * 3.14159265359 / 4.0;
        } else {
            phi = ((vp - up) / r + 1.0) * 3.14159265359 / 4.0;
        }

        // 计算 z 值
        float z = abs(1.0 - r * r);
        if (signedDistance < 0.0) {
            z = -z;
        }

        // 计算 cos(phi) 和 sin(phi)
        float cosPhi = abs(cos(phi));
        if (u < 0.0) {
            cosPhi = -cosPhi;
        }

        float sinPhi = abs(sin(phi));
        if (v < 0.0) {
            sinPhi = -sinPhi;
        }

        // 返回结果
        return float3(cosPhi * r * sqrt(2.0 - r * r), sinPhi * r * sqrt(2.0 - r * r), z);
    }

    float4x4 GenerateUvwTransform(float3 m_Forward, float3 m_Position) {
        // 声明 right 和 up 向量
        float3 right, up;
        up = float3(0, 0, 1);
        right = normalize(cross(m_Forward, up)); // 计算右向量
        up = normalize(cross(m_Forward, right)); // 重新计算上向量
        // 计算纵向和横向的切线
        float vTan = tan(radians(120/ 2.0)); // 纵向切线（tan(fov/2)）
        float hTan = vTan;                         // 横向切线（根据纵向和宽高比计算）
        up *= vTan;                                // 否则调整 up 向量
        right *= hTan;                             // 调整 right 向量

        // 创建UvwTransform矩阵（注意：GLSL中的矩阵是列主序列）
        float4x4 UvwTransform = float4x4(
            float4(right, 0.0),     // 第一列（右向量）
            float4(up, 0.0),        // 第二列（上向量）
            float4(m_Forward, 0.0), // 第三列（前向量，负号表示视角方向）
            float4(m_Position, 1.0) // 第四列（位置向量）
        );
        return UvwTransform;
    }

    float3 GenerateUvwTransform3(float3 m_Forward, float3 m_Position) {
        // 声明 right 和 up 向量
        float3 right, up;
        up = float3(0, 1, 0);
        right = normalize(cross(m_Forward, up)); // 计算右向量
        up = normalize(cross(m_Forward, right)); // 重新计算上向量
        // 计算纵向和横向的切线
        float vTan = tan(radians(90 / 2.0)); // 纵向切线（tan(fov/2)）
        float hTan = vTan;                   // 横向切线（根据纵向和宽高比计算）
        up *= vTan;                          // 否则调整 up 向量
        right *= hTan;                       // 调整 right 向量

        // 创建UvwTransform矩阵（注意：GLSL中的矩阵是列主序列）
        float4x4 UvwTransform = float4x4(
            float4(right, 0.0),     // 第一列（右向量）
            float4(up, 0.0),        // 第二列（上向量）
            float4(m_Forward, 0.0), // 第三列（前向量，负号表示视角方向）
            float4(m_Position, 1.0) // 第四列（位置向量）
        );
        return m_Forward;
    }

    float3 CameraPerspectiveDir(const float2 traceUV, const float3x3 cameraUVW) {
        float3 transDir = float3(traceUV * 2.0 - 1.0, 1.0);
        transDir = normalize(transDir);
        return mul(cameraUVW,transDir);
    }

    float3 hemiOctahedralDir(float2 uv01)
    {
        // 把 [0,1]² 挤压到 [-1,1]²
        float2 p = uv01 * 2.0 - 1.0;

        // 投影到八面体面:  x+y+z = 1  或  -x-y+z = 1 （折叠前）
        float3 v = float3(p, 1.0 - abs(p.x) - abs(p.y));

        // 如果落在下半球，沿对角线折叠回上半球
        if (v.z < 0.0)
        {
            v.xy = (1.0 - abs(v.yx)) * sign(v.xy);
            v.z = -v.z; // ★关键：翻正 z，确保 z ≥ 0
        }
        return normalize(v);
    }

    float3 concentricMappingHemisphere2DTo3D(float2 direction)
    {
        // 把 [0,1]² 映射到 [-1,1]²
        direction = direction * 2.0 - 1.0;

        float u = max(abs(direction.x), abs(direction.y));
        float v = min(abs(direction.x), abs(direction.y));

        float r = u;
        float phi = (3.14159265358979323846 * 0.25) * v / max(u, 1e-6); // 避免除 0

        float x = cos(phi) * r * sqrt(max(0.0, 2.0 - r * r));
        float y = sin(phi) * r * sqrt(max(0.0, 2.0 - r * r));
        float z = 1.0 - r * r;

        // 交换 x/y（与 Shirley–Chiu 同步）
        if (abs(direction.x) < abs(direction.y))
        {
            float tmp = x;
            x = y;
            y = tmp;
        }

        x *= sign(direction.x);
        y *= sign(direction.y);

        return normalize(float3(x, y, z)); // z ≥ 0 的单位向量
    }

    float2 dirToHemiOctahedral(float3 dirW)
    {
        float3 v = dirW;
        v /= abs(v.x) + abs(v.y) + abs(v.z); // 归一化到八面体面
        if (v.z < 0.0)
        {
            v.xy = (1.0 - abs(v.yx)) * sign(v.xy); // 再折叠一次
        }
        return v.xy * 0.5 + 0.5;
    }

    Ray computeRayMultiView(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        Ray ray;
        // Compute the normalized ray direction assuming a pinhole camera.
        int2 camera_idx = pixel / data.perViewResolution;
        int2 view_idx = pixel - camera_idx * data.perViewResolution;
        float3 dir = normalize(GeneratePositionByUV((float2(camera_idx) + 0.5)/float2(data.cameraResolution)));
        ray.origin = dir * 0.8;
        ray.origin.y = ray.origin.y;
        float2 view_uv = (float2(view_idx) + 0.5) / data.perViewResolution;
        float4x4 uvw = GenerateUvwTransform(-dir, ray.origin);
        float3x3 uvw3 = transpose(float3x3(uvw));
        //float3 localDir = concentricMappingHemisphere2DTo3D(view_uv);   // +z ≈ “前半球”
        ray.dir = CameraPerspectiveDir(view_uv, uvw3);
        //ray.dir = normalize(mul(uvw3, localDir));
        //ray.dir = localDir;
        ray.tMin = 0.001;
        ray.tMax = 99999;
        printSetPixel(pixel);
        print("dir", ray.dir);
        return ray;
    }
};
