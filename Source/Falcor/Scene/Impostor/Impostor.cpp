/***************************************************************************
 # Copyright (c) 2015-25, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 # MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 **************************************************************************/
#include "Impostor.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/Gui.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ScriptWriter.h"
#include <math.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
namespace Falcor
{
    using namespace std;
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    vector<string> vec = {"2depth","2albedo","2normal"};
    Impostor::Impostor(const std::string& name)
        : mName(name)
    {
    }

    bool Impostor::loadFromFolder(ref<Device> pDevice,const std::string& folderPath)
    {
        if (!fs::exists(folderPath))
        {
            logWarning("Impostor::loadFromFolder() - folder does not exist: {}", folderPath);
            return false;
        }

        std::ifstream ifs(folderPath + "//basic_info.json");
        if (!ifs.is_open()) {
            std::cerr << "Failed to open JSON file!" << std::endl;
            return false;
        }


        json j;
        ifs >> j;
        radius = j["radius"].get<float>();
        centorWS = float3(j["centorWS"][0].get<float>(), j["centorWS"][1].get<float>(), j["centorWS"][2].get<float>());
        level = j["level"].get<int>() - 1;
        texDim = uint2(j["texDim"][0].get<uint32_t>()/ std::pow(2, level), j["texDim"][1].get<uint32_t>()/ std::pow(2, level)) ;
        invTexDim = 1.0f / float2(texDim);
        baseCameraResolution = j["baseCameraResolution"].get<int>();
        if (level == 0) {
            mpNormalAtlas = Texture::createFromFolder(pDevice, folderPath , false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None,"2normal");
            mpAlbedoAtlas = Texture::createFromFile(pDevice, folderPath + "\\albedo_atlas.exr", false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None);
        }
        
        mpDepthArray = Texture::createFromFolder(pDevice, folderPath, false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None,"2depth");
        mpDepthAtlas = Texture::createFromFile(pDevice, folderPath + "\\depth_atlas.exr", false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None);
        Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(folderPath + "\\lookup_uint16.png",true, Bitmap::ImportFlags::None);
        mpFaceIndex = pDevice->createTexture2D(
            pBitmap->getWidth(),
            pBitmap->getHeight(),
            ResourceFormat::R16Uint,
            1,
            1,
            pBitmap->getData(),
            ResourceBindFlags::ShaderResource
        );;
     
       
        createCameraDirectionBuffersFromFolder(pDevice, folderPath);
        createSampler(pDevice);

        const uint32_t n = mpForwardDirs->getElementCount();
  

        mViewCount = n;
        mDirty = true;

        
        return true;
    }

    void Impostor::createSampler(ref<Device> pDevice) {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
        samplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpPointSampler = pDevice->createSampler(samplerDesc);
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        samplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpLinearSampler = pDevice->createSampler(samplerDesc);
    }

    template<typename Vec3T>
    std::vector<Vec3T> loadVec3ArrayTyped(const std::filesystem::path& jsonPath)
    {
        std::vector<Vec3T> data;

        if (!std::filesystem::exists(jsonPath))
        {
            logWarning("JSON file '{}' not found.", jsonPath.string());
            return data;
        }

        std::ifstream ifs(jsonPath);
        if (!ifs.is_open())
        {
            logWarning("Failed to open JSON file '{}'.", jsonPath.string());
            return data;
        }

        nlohmann::json j;
        try { ifs >> j; }
        catch (const std::exception& e)
        {
            logWarning("Failed to parse JSON '{}': {}", jsonPath.string(), e.what());
            return data;
        }

        if (!j.is_array())
        {
            logWarning("JSON file '{}' is not an array.", jsonPath.string());
            return data;
        }

        data.reserve(j.size());
        for (auto& elem : j)
        {
            if (elem.is_array() && elem.size() >= 3)
            {
                Vec3T v;
                v.x = (typename Vec3T::value_type)elem[0].get<double>();
                v.y = (typename Vec3T::value_type)elem[1].get<double>();
                v.z = (typename Vec3T::value_type)elem[2].get<double>();
                data.push_back(v);
            }
        }
        return data;
    }

    void Impostor::createCameraDirectionBuffersFromFolder(ref<Device> pDevice, const std::filesystem::path& folder)
    {

      
        // 1️⃣ 读取 JSON
        auto fPath = folder / "forward.json";
        auto uPath = folder / "up.json";
        auto rPath = folder / "right.json";
        auto pPath = folder / "position.json";
        auto facePath = folder / "faces.json";
        auto radiusPath = folder / "radius.json";
     

        std::vector<float3> forward = loadVec3ArrayTyped<float3>(fPath);
        std::vector<float3> up = loadVec3ArrayTyped<float3>(uPath);
        std::vector<float3> right = loadVec3ArrayTyped<float3>(rPath);
        std::vector<float3> position = loadVec3ArrayTyped<float3>(pPath);
        std::vector<int3> face = loadVec3ArrayTyped<int3>(facePath);
        std::ifstream file(radiusPath);
        json r;
        file >> r;
        std::vector<float> viewRadius = r.get<std::vector<float>>();
        
        // 2️⃣ 校验数量一致
        size_t n = std::max({ forward.size(), up.size(), right.size() });
        if (n == 0)
        {
            logWarning("No valid vectors loaded from '{}'.", folder.string());
        }
        if (!(forward.size() == up.size() && up.size() == right.size()))
        {
            logWarning("Warning: forward/up/right array sizes differ (f={}, u={}, r={}).", forward.size(), up.size(), right.size());
        }

        mpForwardDirs = make_ref<Buffer>(
            pDevice,
            sizeof(float3),                            // structSize
            (uint32_t)forward.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            forward.data(),                     // 初始数据
            false                                      // 不需要 counter
        );
        mpUpDirs = make_ref<Buffer>(
            pDevice,
            sizeof(float3),                            // structSize
            (uint32_t)up.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            up.data(),                     // 初始数据
            false                                      // 不需要 counter
        );
        mpRightDirs = make_ref<Buffer>(
            pDevice,
            sizeof(float3),                            // structSize
            (uint32_t)right.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            right.data(),                     // 初始数据
            false                                      // 不需要 counter
        );
        mpPosition = make_ref<Buffer>(
            pDevice,
            sizeof(float3),                            // structSize
            (uint32_t)position.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            position.data(),                     // 初始数据
            false                                      // 不需要 counter
        );
        mpFaces = make_ref<Buffer>(
            pDevice,
            sizeof(int3),                            // structSize
            (uint32_t)face.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            face.data(),                     // 初始数据
            false                                      // 不需要 counter
        );
        mpRadius = make_ref<Buffer>(
            pDevice,
            sizeof(float),                            // structSize
            (uint32_t)viewRadius.size(),           // elementCount
            ResourceBindFlags::ShaderResource,         // 绑定给 shader
            MemoryType::DeviceLocal,           // GPU 专用
            viewRadius.data(),                     // 初始数据
            false                                      // 不需要 counter
        );

    }

    void Impostor::bindShaderData(const ShaderVar& var) const
    {
        if (!mpLinearSampler || !mpPointSampler)
        {
            logWarning("Impostor::bindShaderData() called without valid texture or sampler.");
            return;
        }

        //var["texDepth"] = mpDepthArray;                                                                                   
        //var["texAlbedo"] = mpAlbedoArray;
        //var["texNormal"] = mpNormalArray;

        if (level == 0) {
            var["texNormalArray"][0] = mpNormalAtlas[0];
            var["texAlbedoAtlas"] = mpAlbedoAtlas;
        }
        var["texDepthAtlas"][level] = mpDepthAtlas;
        if (level < 3) 
            var["texDepthArray"][level] = mpDepthArray[0];
        else if (level == 3) {
            var["texDepthArray"][level] = mpDepthArray[0];
            var["texDepthArray"][level + 1] = mpDepthArray[1];
        }
        else if (level == 4) {
            var["texDepthArray"][level + 1] = mpDepthArray[0];
            var["texDepthArray"][level + 2] = mpDepthArray[1];
        }
        var["texFaceIndex"][level] = mpFaceIndex;
        var["samplerLinear"][level] = mpLinearSampler;
        var["samplerPoint"][level] = mpPointSampler;

        if (mpForwardDirs) var["cForward"][level] = mpForwardDirs;
        if (mpUpDirs) var["cUp"][level] = mpUpDirs;
        if (mpRightDirs) var["cRight"][level] = mpRightDirs;
        if (mpPosition) var["cPosition"][level] = mpPosition;
        if (mpFaces) var["cFace"][level] = mpFaces;
        if (mpRadius) var["cRadius"][level] = mpRadius;
        var["level"][level] = level;
        var["centerWS"][level] = centorWS;
        var["radius"][level] = radius;
        var["invTexDim"][level] = invTexDim;
        var["texDim"][level] = texDim;
        var["baseCameraResolution"][level] = baseCameraResolution;
    }

    void Impostor::reload(RenderContext* pRenderContext)
    {
        if (!mName.empty())
        {
            logInfo("Reloading impostor '{}'", mName);
            mDirty = false;
        }
    }

    void Impostor::renderUI(Gui::Widgets& widget)
    {
        widget.text("Impostor: " + mName);
        widget.text(fmt::format("{}x{} views: {}", mTexWidth, mTexHeight, mViewCount));



     /*   if (mpTexArray)
        {
            Gui::DropdownList formats;
            formats.push_back({ 0, to_string(mpTexArray->getFormat()) });
            widget.dropdown("Format", formats, 0);
        }*/

        if (mpLinearSampler)
        {
            widget.text("Sampler: Linear Clamp");
        }

        widget.separator();

    }

    FALCOR_SCRIPT_BINDING(Impostor)
    {
        using namespace pybind11::literals;

        pybind11::class_<Impostor, ref<Impostor>> impostor(m, "Impostor");
        impostor.def(pybind11::init(&Impostor::create), "name"_a = "");
        impostor.def("loadFromFolder", &Impostor::loadFromFolder,"pDevice"_a, "folderPath"_a);
    }
} // namespace Falcor
