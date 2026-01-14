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
import Scene.RaytracingInline;
import VBufferRT;
import Scene.Impostor.Impostor;
ConstantBuffer<VBufferRT> gVBufferRT;
float AngleBetweenDeg(float3 a, float3 b)
{
    float3 na = normalize(a);
    float3 nb = normalize(b);

    float cosTheta = dot(na, nb);
    cosTheta = clamp(cosTheta, -1.0f, 1.0f);

    return acos(cosTheta) * (180.0f / 3.1415926);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gVBufferRT.frameDim))
        return;
    const Ray ray = gVBufferRT.generateRay(pixel);
    uint r = int(gPreRoughness[pixel].r * 100);
    float depth_tanh = half_tan[r];
    float half_deg = deg[r];
    

    for (int i = 0; i < gVBufferRT.impostorCnt; i++) {
        uint idx = NonUniformResourceIndex(i);
        gWDepth[idx][pixel] = float4(0);
        gSphere[idx][pixel] = float4(0);
        

        if (idx == 0) {
            float4x4 invM = gVBufferRT.gImpostor0.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local,1);
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor0, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 1) {
            float4x4 invM = gVBufferRT.gImpostor1.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor1, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 2) {
            float4x4 invM = gVBufferRT.gImpostor2.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor2, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 3) {
            float4x4 invM = gVBufferRT.gImpostor3.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor3, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 4) {
            float4x4 invM = gVBufferRT.gImpostor4.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor4, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 5) {
            float4x4 invM = gVBufferRT.gImpostor5.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor5, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 6) {
            float4x4 invM = gVBufferRT.gImpostor6.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor6, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 7) {
            float4x4 invM = gVBufferRT.gImpostor7.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor7, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 8) {
            float4x4 invM = gVBufferRT.gImpostor8.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor8, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 9) {
            float4x4 invM = gVBufferRT.gImpostor9.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor9, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
        else if (idx == 10) {
            float4x4 invM = gVBufferRT.gImpostor10.inverseWorldTransform;
            float3 rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
            float3 rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);
            float3 reflect_pos_local = mul(invM, float4(gReflectPos[pixel].xyz, 1.0)).xyz;
            gDebug[idx][pixel] = float4(reflect_pos_local, 1);
            float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel(
        pixel, gVBufferRT.gImpostor10, rayDirL_raw, rayPos_local, 0, depth_tanh, idx);
        }
    }
    // for (int idx = 0; idx < gVBufferRT.impostorCnt; idx++) {
    //     float4x4 invM;
    //     float3 rayPos_local;
    //     float3 rayDirL_raw;
    //     float4 result4;

    //     // 根据 idx 的值选择相应的 impostor
    //     invM = (idx == 0)   ? gVBufferRT.gImpostor0.inverseWorldTransform :
    //       (idx == 1) ? gVBufferRT.gImpostor1.inverseWorldTransform :
    //       (idx == 2) ? gVBufferRT.gImpostor2.inverseWorldTransform :
    //       (idx == 3) ? gVBufferRT.gImpostor3.inverseWorldTransform :
    //       (idx == 4) ? gVBufferRT.gImpostor4.inverseWorldTransform :
    //       (idx == 5) ? gVBufferRT.gImpostor5.inverseWorldTransform :
    //       (idx == 6) ? gVBufferRT.gImpostor6.inverseWorldTransform :
    //       (idx == 7) ? gVBufferRT.gImpostor7.inverseWorldTransform :
    //       (idx == 8) ? gVBufferRT.gImpostor8.inverseWorldTransform :
    //       (idx == 9) ? gVBufferRT.gImpostor9.inverseWorldTransform :
    //       gVBufferRT.gImpostor10.inverseWorldTransform;

    //     rayPos_local = mul(invM, float4(gPrePosition[pixel].xyz, 1.0)).xyz;
    //     rayDirL_raw = mul((float3x3)invM, gPreDirection[pixel].xyz);

    //     // 使用对应的 gImpostor 进行采样
    //     result4 = gVBufferRT.sampleImpostorSingleViewLevel( pixel,
    //         (idx == 0)   ? gVBufferRT.gImpostor0 : 
    //           (idx == 1) ? gVBufferRT.gImpostor1 : 
    //           (idx == 2) ? gVBufferRT.gImpostor2 : 
    //           (idx == 3) ? gVBufferRT.gImpostor3 : 
    //           (idx == 4) ? gVBufferRT.gImpostor4 : 
    //           (idx == 5) ? gVBufferRT.gImpostor5 : 
    //           (idx == 6) ? gVBufferRT.gImpostor6 : 
    //           (idx == 7) ? gVBufferRT.gImpostor7 : 
    //           (idx == 8) ? gVBufferRT.gImpostor8 : 
    //           (idx == 9) ? gVBufferRT.gImpostor9 : 
    //           gVBufferRT.gImpostor10,
    //         rayDirL_raw, rayPos_local, 0, depth_tanh, idx );
    // }
}
