#pragma once
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Math/AABB.h"
#include "Utils/UI/Gui.h"
#include "Utils/Timing/CpuTimer.h"
#include "Core/API/Texture.h"
#include "Core/API/Buffer.h"
#include "Core/API/Sampler.h"

#include <string>

namespace Falcor
{
    struct ShaderVar;

    /** Impostor resource class.
        This class manages a multi-view impostor representation built from geodesic camera captures.
        It stores view textures, per-view orientation data, and metadata for sampling and rendering.
    */
    class FALCOR_API Impostor : public Object
    {
        FALCOR_OBJECT(Impostor)
    public:
        using SharedPtr = ref<Impostor>;

        /** Create an empty impostor. */
        static SharedPtr create(const std::string& name = "") { return make_ref<Impostor>(name); }

        Impostor(const std::string& name);
        ~Impostor() = default;

        /** Get/Set impostor name. */
        void setName(const std::string& name) { mName = name; }
        const std::string& getName() const { return mName; }

        /** Load a folder of multi-view textures and create the impostor texture array.
            The directory should contain EXR/PNG files for each view.
        */
        bool loadFromFolder(ref<Device> pDevice, const std::string& folderPath);

        /** Set impostor texture array directly. */

        void createCameraDirectionBuffersFromFolder(ref<Device> pDevice, const std::filesystem::path& folder);
        void createSampler(ref<Device> pDevice);
        /** Set per-view direction/up buffers. */
        void setViewDirs(const ref<Buffer>& buf) { mpForwardDirs = buf; mDirty = true; }
        void setUpDirs(const ref<Buffer>& buf) { mpUpDirs = buf; mDirty = true; }

        const ref<Buffer>& getViewDirs() const { return mpForwardDirs; }
        const ref<Buffer>& getUpDirs() const { return mpUpDirs; }

        /** Set and get sampler state. */
        void setSampler(const ref<Sampler>& sampler) { mpSampler = sampler; }
        const ref<Sampler>& getSampler() const { return mpSampler; }

        /** Set impostor metadata. */
        void setViewCount(uint32_t count) { mViewCount = count; mDirty = true; }
        uint32_t getViewCount() const { return mViewCount; }

        void setResolution(uint32_t w, uint32_t h) { mTexWidth = w; mTexHeight = h; mDirty = true; }
        uint32_t getWidth() const { return mTexWidth; }
        uint32_t getHeight() const { return mTexHeight; }




        /** Bind all resources into a shader variable. */
        void bindShaderData(const ShaderVar& var) const;

        /** Render UI for debug/inspection. */
        void renderUI(Gui::Widgets& widget);

        /** Reload impostor texture from disk (for dev hot-reload). */
        void reload(RenderContext* pRenderContext);

        /** Get changes flag. */
        bool isDirty() const { return mDirty; }

    private:
        mutable bool mDirty = false;
        std::string mName;

        ref<Texture> mpDepthArray;
        ref<Texture> mpAlbedoArray;
        ref<Texture> mpNormalArray;
        ref<Texture> mpFaceIndex;
        ref<Buffer> mpForwardDirs;
        ref<Buffer> mpUpDirs;
        ref<Buffer> mpRightDirs;
        ref<Buffer> mpPosition;
        ref<Buffer> mpFaces;
        ref<Buffer> mpRadius;
        ref<Sampler> mpSampler;

        uint32_t mViewCount = 0;
        uint32_t mTexWidth = 0;
        uint32_t mTexHeight = 0;
    

        float radius;
        float3 centorWS;
        uint32_t level;
        uint2 texDim;
        float2 invTexDim;
        uint32_t baseCameraResolution;
    };

} // namespace Falcor
