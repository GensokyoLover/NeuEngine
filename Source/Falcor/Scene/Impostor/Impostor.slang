
struct Impostor
{
    // impostor 纹理（多视角合并）
    Texture2DArray<float4> texDepth;
    Texture2DArray<float4> texAlbedo;
    Texture2DArray<float4> texNormal;
    Texture2D<uint> texFaceIndex;
    SamplerState samplerLinear;

    // 每个视角的相机方向和up方向
    StructuredBuffer<float3> cForward;
    StructuredBuffer<float3> cUp;
    StructuredBuffer<float3> cRight;
    StructuredBuffer<float3> cPosition;
    StructuredBuffer<int3> cFace;
    StructuredBuffer<float> cRadius;

    // impostor 元数据

    uint level;     // subdivision level
    float3 centerWS; // impostor中心在世界空间的位置
    float radius;
    uint2 texDim;
    float2 invTexDim;
    uint baseCameraResolution;
};  

    ;


