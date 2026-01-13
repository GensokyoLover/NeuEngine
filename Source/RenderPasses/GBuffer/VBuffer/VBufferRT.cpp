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
#include "VBufferRT.h"
#include "Scene/HitInfo.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const std::string kProgramRaytraceFile = "RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang";
//const std::string kProgramComputeFile = "RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang";
const std::string kProgramComputeFile = "RenderPasses/GBuffer/VBuffer/ReflectionImpostorRT.cs.slang";
//const std::string kProgramComputeFile = "RenderPasses/GBuffer/VBuffer/ImpostorRT.cs.slang";

// Scripting options.
const char kUseTraceRayInline[] = "useTraceRayInline";
const char kUseDOF[] = "useDOF";

// Ray tracing settings that affect the traversal stack size. Set as small as possible.
// TODO: The shader doesn't need a payload, set this to zero if it's possible to pass a null payload to TraceRay()
const uint32_t kMaxPayloadSizeBytes = 4;
const uint32_t kMaxRecursionDepth = 1;

const std::string kVBufferName = "vbuffer";
const std::string kVBufferDesc = "V-buffer in packed format (indices + barycentrics)";
const ChannelList kInputChannels = {
    // clang-format off
    { "prePosition",        "gPrePosition",     "Visibility buffer in packed format", true },
    { "preDirection",    "gPreDirection",       "World-space view direction (xyz float format)", true},
    { "preRoughness",    "gPreRoughness",       "World-space view direction (xyz float format)", true},
    // clang-format on
};
// Additional output channels.
const ChannelList kVBufferExtraChannels = {
    // clang-format off
    { "depth",          "gDepth",           "Depth buffer (NDC)",               true /* optional */, ResourceFormat::R32Float    },
    { "mvec",           "gMotionVector",    "Motion vector",                    true /* optional */, ResourceFormat::RG32Float   },
    { "viewW",          "gViewW",           "View direction in world space",    true /* optional */, ResourceFormat::RGBA32Float }, // TODO: Switch to packed 2x16-bit snorm format.
    { "time",           "gTime",            "Per-pixel execution time",         true /* optional */, ResourceFormat::R32Uint     },
    { "mask",           "gMask",            "Mask",                             true /* optional */, ResourceFormat::R32Float    },
    

};
}; //


VBufferRT::VBufferRT(ref<Device> pDevice, const Properties& props) : GBufferBase(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("VBufferRT requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("VBufferRT requires Raytracing Tier 1.1 support.");

    parseProperties(props);

    // Create sample generator
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
    mpRoughness = 0;
}

RenderPassReflection VBufferRT::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    if(mUseTraceRayInline)
        addRenderPassInputs(reflector, kInputChannels);
    // Add the required output. This always exists.
    reflector.addOutput(kVBufferName, kVBufferDesc)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(mVBufferFormat)
        .texture2D(sz.x, sz.y);

    // Add all the other outputs.
    addRenderPassOutputs(reflector, kVBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);
    for (int i = 0; i < 11; i++) {
        
        reflector.addOutput("uv0_"+ std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("uv1_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("uv2_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("depth0_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("depth1_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("depth2_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("direction0_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("direction1_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
        reflector.addOutput("direction2_" + std::to_string(i), "uv output")
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(sz.x, sz.y);
    }
    return reflector;
}

void VBufferRT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    GBufferBase::execute(pRenderContext, renderData);

    // Update frame dimension based on render pass output.
    auto pOutput = renderData.getTexture(kVBufferName);
    FALCOR_ASSERT(pOutput);
    updateFrameDim(uint2(pOutput->getWidth(), pOutput->getHeight()));

    if (mpScene)
    {
        // Check for scene changes.
        if (is_set(mUpdateFlags, IScene::UpdateFlags::RecompileNeeded) || is_set(mUpdateFlags, IScene::UpdateFlags::GeometryChanged) ||
            is_set(mUpdateFlags, IScene::UpdateFlags::SDFGridConfigChanged))
        {
            recreatePrograms();
        }

        // Configure depth-of-field.
        // When DOF is enabled, two PRNG dimensions are used. Pass this info to subsequent passes via the dictionary.
        mComputeDOF = mUseDOF && mpScene->getCamera()->getApertureRadius() > 0.f;
        if (mUseDOF)
        {
            renderData.getDictionary()[Falcor::kRenderPassPRNGDimension] = mComputeDOF ? 2u : 0u;
        }

        mUseTraceRayInline ? executeCompute(pRenderContext, renderData) : executeRaytrace(pRenderContext, renderData);
        //executeCompute(pRenderContext, renderData);
        mUpdateFlags = IScene::UpdateFlags::None;
        mFrameCount++;
    }
    else // If there is no scene, clear the output and return.
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), uint4(0));
        clearRenderPassChannels(pRenderContext, kVBufferExtraChannels, renderData);
        return;
    }
}

void VBufferRT::renderUI(Gui::Widgets& widget)
{
    GBufferBase::renderUI(widget);

    if (widget.checkbox("Use TraceRayInline", mUseTraceRayInline))
    {
        mOptionsChanged = true;
    }

    if (widget.checkbox("Use depth-of-field", mUseDOF))
    {
        mOptionsChanged = true;
    }
    
    widget.var("mpRoughness ", mpRoughness, 0,100);
    //for (int i = 0; i < mpScene->mImpostors.size(); i++) {
    //    widget.var(std::to_string(i).c_str()+"impostor ", mpRoughness, 0, 100);
    //}
    widget.tooltip(
        "This option enables stochastic depth-of-field when the camera's aperture radius is nonzero. "
        "Disable it to force the use of a pinhole camera.",
        true
    );
}

Properties VBufferRT::getProperties() const
{
    Properties props = GBufferBase::getProperties();
    props[kUseTraceRayInline] = mUseTraceRayInline;
    props[kUseDOF] = mUseDOF;

    return props;
}


void VBufferRT::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    GBufferBase::setScene(pRenderContext, pScene);

    recreatePrograms();
}

void VBufferRT::parseProperties(const Properties& props)
{
    GBufferBase::parseProperties(props);

    for (const auto& [key, value] : props)
    {
        if (key == kUseTraceRayInline)
            mUseTraceRayInline = value;
        else if (key == kUseDOF)
            mUseDOF = value;
        // TODO: Check for unparsed fields, including those parsed in base classes.
    }
}

void VBufferRT::recreatePrograms()
{
    mRaytrace.pProgram = nullptr;
    mRaytrace.pVars = nullptr;
    mpComputePass = nullptr;
}

void VBufferRT::executeRaytrace(RenderContext* pRenderContext, const RenderData& renderData)
{
    //std::cout << "rt" << std::endl;
    if (!mRaytrace.pProgram || !mRaytrace.pVars)
    {
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getShaderDefines(renderData));

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramRaytraceFile);
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        ref<RtBindingTable> sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));

        // Add hit group with intersection shader for triangle meshes with displacement maps.
        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("displacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
        }

        // Add hit group with intersection shader for curves (represented as linear swept spheres).
        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("curveClosestHit", "", "curveIntersection")
            );
        }

        // Add hit group with intersection shader for SDF grids.
        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("sdfGridClosestHit", "", "sdfGridIntersection")
            );
        }

        mRaytrace.pProgram = Program::create(mpDevice, desc, defines);
        mRaytrace.pVars = RtProgramVars::create(mpDevice, mRaytrace.pProgram, sbt);

        // Bind static resources.
        ShaderVar var = mRaytrace.pVars->getRootVar();
        mpSampleGenerator->bindShaderData(var);
    }

    mRaytrace.pProgram->addDefines(getShaderDefines(renderData));

    ShaderVar var = mRaytrace.pVars->getRootVar();
    bindShaderData(var, renderData);

    // Dispatch the rays.
    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uint3(mFrameDim, 1));
}

void VBufferRT::executeCompute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Create compute pass.
    //std::cout << "compute" << std::endl;
    FALCOR_PROFILE(pRenderContext,"cs");
    if (pRenderContext->gframeCount >= 3) {
        
        return;
    }
  
    if (!mpComputePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramComputeFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.addCompilerArguments({ "-O0", "-g" });
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getShaderDefines(renderData));

        mpComputePass = ComputePass::create(mpDevice, desc, defines, true);

        
    }

    // Bind static resources
    
    /*if (impostorInit == false) {
        ShaderVar var = mpComputePass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
        mpSampleGenerator->bindShaderData(var);
        auto impostor = mpScene->mImpostors[0];
        impostor->bindShaderData(var["gVBufferRT"]["gImpostor"]);
        auto impostor2 = mpScene->mImpostors[1];
        impostor2->bindShaderData(var["gVBufferRT"]["gImpostor2"]);
        auto impostor3 = mpScene->mImpostors[2];
        impostor3->bindShaderData(var["gVBufferRT"]["gImpostor3"]);
        auto impostor4 = mpScene->mImpostors[3];
        impostor4->bindShaderData(var["gVBufferRT"]["gImpostor4"]);
        mpComputePass->getProgram()->addDefines(getShaderDefines(renderData));
        impostorInit = true;
        bindShaderData(var, renderData);
    }*/
    
    ShaderVar rootVar = mpComputePass->getRootVar();

    mpScene->bindShaderDataForRaytracing(pRenderContext, rootVar["gScene"]);
    mpSampleGenerator->bindShaderData(rootVar);
    for (int i = 0; i < mpScene->mImpostors.size();i++) {
        mpScene->mImpostors[i]->bindShaderData(rootVar["gVBufferRT"]["gImpostor" + std::to_string(i)]);
    }
    
    /* mpScene->mImpostors[1]->bindShaderData(rootVar["gVBufferRT"]["gImpostor"]);
    mpScene->mImpostors[2]->bindShaderData(rootVar["gVBufferRT"]["gImpostor"]);
    mpScene->mImpostors[3]->bindShaderData(rootVar["gVBufferRT"]["gImpostor"]);
    mpScene->mImpostors[4]->bindShaderData(rootVar["gVBufferRT"]["gImpostor"]);*/

    impostorInit = true;
   
    // ✅ 每帧重新取 rootVar（轻量）
   // ShaderVar rootVar = mpComputePass->getRootVar();
    rootVar["gVBufferRT"]["roughness"] = mpRoughness;
    rootVar["gVBufferRT"]["impostorCnt"] = int(mpScene->mImpostors.size());
    rootVar["gPrePosition"] = renderData.getTexture("prePosition");
    rootVar["gPreDirection"] = renderData.getTexture("preDirection");
    rootVar["gPreRoughness"] = renderData.getTexture("preRoughness");
    // ✅ 只绑定 per-frame 数据
    bindShaderData(rootVar, renderData);
    mpComputePass->execute(pRenderContext, uint3(mFrameDim, 1));
}


DefineList VBufferRT::getShaderDefines(const RenderData& renderData) const
{
    DefineList defines;
    defines.add("COMPUTE_DEPTH_OF_FIELD", mComputeDOF ? "1" : "0");
    defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

    // Setup ray flags.
    RayFlags rayFlags = RayFlags::None;
    if (mForceCullMode && mCullMode == RasterizerState::CullMode::Front)
        rayFlags = RayFlags::CullFrontFacingTriangles;
    else if (mForceCullMode && mCullMode == RasterizerState::CullMode::Back)
        rayFlags = RayFlags::CullBackFacingTriangles;
    defines.add("RAY_FLAGS", std::to_string((uint32_t)rayFlags));

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    defines.add(getValidResourceDefines(kVBufferExtraChannels, renderData));
    return defines;
}

void VBufferRT::bindShaderData(const ShaderVar& var, const RenderData& renderData)
{
    var["gVBufferRT"]["frameDim"] = mFrameDim;
    var["gVBufferRT"]["frameCount"] = mFrameCount;

    // Bind resources.
    var["gVBuffer"] = getOutput(renderData, kVBufferName);

    // Bind output channels as UAV buffers.
    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = getOutput(renderData, channel.name);
        var[channel.texname] = pTex;
    };
    for (const auto& channel : kVBufferExtraChannels)
        bind(channel);
    for (int i = 0; i < mpScene->mImpostors.size(); i++) {
        var["gAlbedo0"][i] = getOutput(renderData, "uv0_" + std::to_string(i));
        var["gAlbedo1"][i] = getOutput(renderData, "uv1_" + std::to_string(i));
        var["gAlbedo2"][i] = getOutput(renderData, "uv2_" + std::to_string(i));
        var["gDepth0"][i] = getOutput(renderData, "depth0_" + std::to_string(i));
        var["gDepth1"][i] = getOutput(renderData, "depth1_" + std::to_string(i));
        var["gDepth2"][i] = getOutput(renderData, "depth2_" + std::to_string(i));
        var["gDir0"][i] = getOutput(renderData, "direction0_" + std::to_string(i));
        var["gDir1"][i] = getOutput(renderData, "direction1_" + std::to_string(i));
        var["gDir2"][i] = getOutput(renderData, "direction2_" + std::to_string(i));
       
    }
}
