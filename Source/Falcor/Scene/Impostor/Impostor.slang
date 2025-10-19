
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

    // impostor 元数据
    uint viewCount;
    uint texWidth;
    uint texHeight;
    float margin;    // 可选：外扩比例
    float level;     // subdivision level
    float3 centerWS; // impostor中心在世界空间的位置
    float radius;
    float2 TexDim;
    float2 inv_TexDim; 
};  

    ;
float2 OctEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
    return n.xy * 0.5 + 0.5;
}

uint getNearestImpostorView(Impostor impostor, float3 rayPosW)
{
    float3 dir = normalize(rayPosW);
    float2 uv = OctEncode(dir); // [-1,1] → [0,1]
    uint viewIdx = impostor.texFaceIndex.Load(int3(uv * impostor.TexDim, 0)).r;

    // 然后直接采样对应层的 impostor 纹理
    float4 color = impostor.texAlbedo.Sample(impostor.samplerLinear, float3(uv, viewIdx));
}

// ============================================================
// 最基础 impostor 渲染函数：选择最近视角并采样
// ============================================================
float4 sampleImpostor(Impostor impostor, float3 rayDirW, float3 rayPosW)
{
    uint viewIdx = getNearestImpostorView(impostor, rayPosW);

    // === Step 1: 获取该视角对应的摄像机空间 ===
    float3 forward = impostor.cForward[viewIdx];
    float3 up = impostor.cUp[viewIdx];
    float3 right = impostor.cRight[viewIdx];
    float3 camPos = impostor.cPosition[viewIdx];

    // === Step 2: 将射线坐标转换到该视角的局部空间 ===
    // 世界坐标 → 相机局部
    float3 localDir = float3(dot(rayPosW, right), dot(rayPosW, up), dot(rayPosW, forward));
    float3 localPos = float3(dot(rayDirW - camPos, right),
                             dot(rayDirW - camPos, up),
                             dot(rayDirW - camPos, forward));

    // === Step 3: 计算投影到该 impostor 贴图的 UV ===
    // 假设 impostor 在 [-radius, +radius] 范围内
    float2 uv = (localPos.xy / (impostor.radius * 2.0)) + 0.5;

    // UV 反转或裁剪（按需要）
    uv = saturate(uv);

    // === Step 4: 采样对应层的颜色 / 深度 / 法线 ===
    float4 albedo = impostor.texAlbedo.Sample(impostor.samplerLinear, float3(uv, viewIdx));
    float depth = impostor.texDepth.Sample(impostor.samplerLinear, float3(uv, viewIdx)).r;
    float3 normal = impostor.texNormal.Sample(impostor.samplerLinear, float3(uv, viewIdx)).xyz * 2.0 - 1.0;

    // === Step 5: 输出组合结果（这里只输出 albedo，可扩展）===
    return float4(albedo.rgb, 1.0);
}