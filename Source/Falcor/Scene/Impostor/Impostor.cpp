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

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
namespace Falcor
{
    using namespace std;
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    vector<string> vec = {"depth","albedo","normal"};
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

        std::ifstream ifs(folderPath + "//base.json");
        if (!ifs.is_open()) {
            std::cerr << "Failed to open JSON file!" << std::endl;
            return -1;
        }

        // 2️⃣ 解析 JSON
        json j;
        ifs >> j;

        mpDepthArray = Texture::createFromFolder(pDevice, folderPath, false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None, vec[0]);
        mpAlbedoArray= Texture::createFromFolder(pDevice, folderPath, false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None, vec[1]);
        mpNormalArray= Texture::createFromFolder(pDevice, folderPath, false, false, ResourceBindFlags::ShaderResource, Bitmap::ImportFlags::None, vec[2]);
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
     
       
        const uint32_t w = mpDepthArray->getWidth();
        const uint32_t h = mpDepthArray->getHeight();
        const uint32_t n = (uint32_t)mpDepthArray->getArraySize();
        const auto fmt = mpDepthArray->getFormat();
        mTexWidth = w;
        mTexHeight = h;
        mViewCount = n;
        mDirty = true;

        logInfo("Loaded impostor from folder '{}', {} views ({}x{})", folderPath, n, w, h);
        createCameraDirectionBuffersFromFolder(pDevice, folderPath);
        createSampler(pDevice);
        return true;
    }

    void Impostor::createSampler(ref<Device> pDevice) {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        samplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpSampler = pDevice->createSampler(samplerDesc);
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
     

        std::vector<float3> forward = loadVec3ArrayTyped<float3>(fPath);
        std::vector<float3> up = loadVec3ArrayTyped<float3>(uPath);
        std::vector<float3> right = loadVec3ArrayTyped<float3>(rPath);
        std::vector<float3> position = loadVec3ArrayTyped<float3>(pPath);
        std::vector<int3> face = loadVec3ArrayTyped<int3>(facePath);
        
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

    }

    void Impostor::bindShaderData(const ShaderVar& var) const
    {
        if (!mpDepthArray || !mpSampler)
        {
            logWarning("Impostor::bindShaderData() called without valid texture or sampler.");
            return;
        }

        var["texDepth"] = mpDepthArray;
        var["texAlbedo"] = mpAlbedoArray;
        var["texNormal"] = mpNormalArray;
        var["texFaceIndex"] = mpFaceIndex;
        var["samplerLinear"] = mpSampler;
        if (mpForwardDirs) var["cForward"] = mpForwardDirs;
        if (mpUpDirs) var["cUp"] = mpUpDirs;
        if (mpRightDirs) var["cRight"] = mpRightDirs;
        if (mpPosition) var["cPosition"] = mpPosition;
        if (mpFaces) var["cFace"] = mpFaces;
        var["viewCount"] = mViewCount;
        var["texWidth"] = mTexWidth;
        var["texHeight"] = mTexHeight;
        var["margin"] = mMargin;
        var["level"] = mLevel;
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

        float margin = mMargin;
        if (widget.var("Margin", margin, 0.f, 2.f, 0.001f)) setMargin(margin);

        float level = mLevel;
        if (widget.var("Subdiv Level", level, 0.f, 8.f, 1.f)) setLevel(level);

     /*   if (mpTexArray)
        {
            Gui::DropdownList formats;
            formats.push_back({ 0, to_string(mpTexArray->getFormat()) });
            widget.dropdown("Format", formats, 0);
        }*/

        if (mpSampler)
        {
            widget.text("Sampler: Linear Clamp");
        }

        widget.separator();
        if (widget.button("Dump Info"))
        {
            fmt::print("Impostor '{}' — {}x{} / {} views / level {:.1f}\n",
                mName, mTexWidth, mTexHeight, mViewCount, mLevel);
        }
    }

    FALCOR_SCRIPT_BINDING(Impostor)
    {
        using namespace pybind11::literals;

        pybind11::class_<Impostor, ref<Impostor>> impostor(m, "Impostor");
        impostor.def(pybind11::init(&Impostor::create), "name"_a = "");
        impostor.def("loadFromFolder", &Impostor::loadFromFolder,"pDevice"_a, "folderPath"_a);
    }
} // namespace Falcor
