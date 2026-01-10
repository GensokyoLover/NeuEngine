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
    gDepth0[pixel] = float4(0);
    gDepth1[pixel] = float4(0);
    gDepth2[pixel] = float4(0);
    gDir0[pixel] = float4(0);
    gDir1[pixel] = float4(0);
    gDir2[pixel] = float4(0);
    gAlbedo0[pixel] = float4(0);
    gAlbedo1[pixel] = float4(0);
    gAlbedo2[pixel] = float4(0);
    float miDepth = 1000;
    if (any(gPrePosition[pixel].xyz != float3(0, 0, 0)) || any(gPreDirection[pixel].xyz != float3(0, 0, 0))) {
        float4 result4 = gVBufferRT.sampleImpostorSingleViewLevel( pixel, gVBufferRT.gImpostor, gPreDirection[pixel].xyz, gPrePosition[pixel].xyz, 0, depth_tanh );
        //float4 result4 =float4(4);
        gVBufferRT.writeAux(pixel, result4);
    }
    else {
 
        gVBufferRT.writeAux(pixel, float4(0,0,0,0));
    }
    //gVBufferRT.writeDebug(pixel, float4(float(endlevel)/3));
    //gVBufferRT.writeAux(pixel, depth_tanh/100.0);

    
    // SceneRayQuery<VBufferRT::kUseAlphaTest> sceneRayQuery;
    // float hitT;
    // const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, VBufferRT::kRayFlags);
    // if (hit.isValid())
    // {
    //     gVBufferRT.writeHit(pixel, ray.origin, ray.dir, hit);
    // }
    // else
    // {
    //     gVBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    // }


    //gVBufferRT.endTime(pixel, timer);
}
