static const uint IMPOSTOR_CHUNK_SIZE = 256;

void DecodeSlice(
    uint sliceId,
    out uint arrayId,
    out uint layerId
)
{
    arrayId = sliceId >> 11;     // /256
    layerId = sliceId & 2047;    // %256
}
// 定义一个 ParameterBlock

static const float4 gTilesX = float4(7, 13, 26, 51);                // 对应 Python 的 tiles_x
static const float4 gInvTilesX = 1/float4(7, 13, 26, 51);      // 对应 Python 的 tiles_x
static const float4 gHalfTexelSize = 0.5/float4(512, 256, 128, 64); // 对应 Python 的 tiles_x
static const float4 gTileScaleCol = 1 / float4(7,13,26,51); // 对应 Python 的 tiles_x
static const float4 gTileScaleRow = 1 / float4(6, 13, 25, 51); // 对应 Python 的 tiles_x
static const float2 gTileSizeNorm[4] = {
    float2(1.0f / 7.0f, 1.0f / 6.0f),   // Level 0: 7 Cols, 6 Rows
    float2(1.0f / 13.0f, 1.0f / 13.0f), // Level 1: 13 Cols, 13 Rows
    float2(1.0f / 26.0f, 1.0f / 25.0f), // Level 2: 26 Cols, 25 Rows
    float2(1.0f / 51.0f, 1.0f / 51.0f)  // Level 3: 51 Cols, 51 Rows
};

struct Impostor
{
    // ... 纹理和 Buffer 资源部分保持不变 ...
    Texture2DArray<float4> texDepthArray[10];
    Texture2D<uint> texFaceIndex[5];
    SamplerState samplerLinear[5];
    SamplerState samplerPoint[5];

    StructuredBuffer<float3> cForward[5];
    StructuredBuffer<float3> cUp[5];
    StructuredBuffer<float3> cRight[5];
    StructuredBuffer<float3> cPosition[5];
    StructuredBuffer<int3> cFace[5];
    StructuredBuffer<float> cRadius[5];

    // --- ⬇️ 这里必须改 (使用向量类型) ⬇️ ---


    uint4 levels; // ✅ 安全：占用 16 字节，用 levels.x, levels.y 访问


    float4 centerWS[4]; // ✅ 安全：xyz存位置，w可存其他或留空

    uint4 baseCameraResolution; // ✅ 安全

    float4x4 worldTransform;
    float4x4 inverseWorldTransform;
};

// struct Impostor
// {
//     // impostor 纹理（多视角合并）
    


//     Texture2DArray<float4> texDepthArray[10];
//     Texture2D<uint> texFaceIndex[5];
//     SamplerState samplerLinear[5];
//     SamplerState samplerPoint[5];

//     // 每个视角的相机方向和up方向
//     StructuredBuffer<float3> cForward[5];
//     StructuredBuffer<float3> cUp[5];
//     StructuredBuffer<float3> cRight[5];
//     StructuredBuffer<float3> cPosition[5];
//     StructuredBuffer<int3> cFace[5];
//     StructuredBuffer<float> cRadius[5];
//     uint level[4];     // subdivision level
//     float3 centerWS[4]; // impostor中心在世界空间的位置
//     float radius[4];
//     uint2 texDim[4];

//     uint baseCameraResolution[4];
//     float4x4 worldTransform;
//     float4x4 inverseWorldTransform;
    

// };
float4 SampleImpostorPoint(
    Texture2DArray<float4> texArray[],
    SamplerState samplerPoint,
    uint sliceId,
    float2 uv,
    int level
)
{
    uint arrayId, layerId;
    DecodeSlice(sliceId, arrayId, layerId);
    if (level == 4)
        arrayId = arrayId + 1;
    // 强烈建议显式 SampleLevel，避免隐式 LOD
    return texArray[arrayId + level].SampleLevel(
        samplerPoint,
        float3(uv, layerId ),
        0.0f
    );
}

float4 SingleSampleImpostorPoint(
    Texture2DArray<float4> texArray[],
    SamplerState samplerPoint,
    uint sliceId,
    float2 uv,
    int level
)
{
    uint arrayId, layerId;
    DecodeSlice(sliceId, arrayId, layerId);

    // 强烈建议显式 SampleLevel，避免隐式 LOD
    return texArray[arrayId + level].SampleLevel(
        samplerPoint,
        float3(uv, layerId),
        0.0f
    );
}

float4 SampleImpostorPointSingle(
    Texture2DArray<float4> texArray[],
    SamplerState samplerPoint,
    uint sliceId,
    float2 uv
)
{
    uint arrayId, layerId;
    DecodeSlice(sliceId, arrayId, layerId);

    return texArray[arrayId].SampleLevel(
        samplerPoint,
        float3(uv, layerId),
        0.0f
    );
}
// 建议在 C++ 端计算好 reciprocalNumTiles (1.0 / N) 传入
// 避免在 Shader 里做除法
float4 SampleImpostorPointAtlas_be(
    Texture2D<float4> atlasTex, // 改为单张大图
    SamplerState samplerPoint,
    uint sliceId,
    float2 uv,
    int level
)
{
    uint col = sliceId % gTilesX[level];
    uint row = sliceId / gTilesX[level];

    float2 safeUV = clamp(uv, gHalfTexelSize[level], 1.0f - gHalfTexelSize[level]);
    float2 globalUV = (float2(col, row) + safeUV) * float2(gTileScaleCol[level], gTileScaleRow[level]);

    // 4. 强制采样 Level 0
    return atlasTex.SampleLevel(samplerPoint, globalUV, 0);
}

float4 SampleImpostorPointAtlas(
    Texture2D<float4> atlasTex,
    SamplerState samplerPoint,
    uint sliceId,
    float2 uv,
    int level
    // 移除了 int level，假设直接读取对应 Level 的 CB 变量
)
{
    // --------------------------------------------------------------------
    // 优化 1: 用浮点运算替代整数 / 和 %
    // --------------------------------------------------------------------
    // 原逻辑: row = id / w; col = id % w;
    // 即使在 2026 年，整数除法依然慢。
    // 使用 FMA (Fused Multiply-Add) 指令优化

    float fSlice = (float)sliceId;

    // Row = floor(id * (1/Width))
    float row = floor(fSlice * gInvTilesX[level]);

    // Col = id - row * Width
    // 这种写法会被编译器优化为一条 MAD 指令
    float col = fSlice - row * gTilesX[level];

    float2 tileGridPos = float2(col, row);

    float2 safeUV = uv;

    float2 globalUV = (tileGridPos + safeUV) * gTileSizeNorm[level];
  // --------------------------------------------------------------------
    // 采样
    // --------------------------------------------------------------------
    return atlasTex.SampleLevel(samplerPoint, globalUV, 0);
}
;


