add_plugin(ImpostorTracer)

target_sources(ImpostorTracer PRIVATE
    ImpostorTracer.cpp
    ImpostorTracer.h
    ImpostorTracer.rt.slang
)

target_copy_shaders(ImpostorTracer RenderPasses/ImpostorTracer)

target_source_group(ImpostorTracer "RenderPasses")
