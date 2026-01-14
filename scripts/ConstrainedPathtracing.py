from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    AccumulatePass2 = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass2, "AccumulatePass2")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    MinimalPathTracer = createPass("MinimalPathTracer", {'maxBounces': 6})
    MinimalPathTracer2 = createPass("MinimalPathTracer", {'maxBounces': 6})
    g.addPass(MinimalPathTracer, "MinimalPathTracer")
    g.addPass(MinimalPathTracer2, "MinimalPathTracer2")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":False})
    VBufferRT2 = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":True})
    ##VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addPass(VBufferRT2, "VBufferRT2")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer2.vbuffer")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer2.viewW")

    g.addEdge("MinimalPathTracer2.position", "VBufferRT2.prePosition")
    g.addEdge("MinimalPathTracer2.reflect", "VBufferRT2.preDirection")
    g.addEdge("MinimalPathTracer2.roughness", "VBufferRT2.preRoughness")
    g.addEdge("MinimalPathTracer.mind", "VBufferRT2.preReflectPos")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.addEdge("MinimalPathTracer.type", "AccumulatePass2.input")
    g.markOutput("MinimalPathTracer.position")
    g.markOutput("MinimalPathTracer.albedo")
    g.markOutput("MinimalPathTracer.specular")
    g.markOutput("MinimalPathTracer.normal")
    g.markOutput("MinimalPathTracer.roughness")
    g.markOutput("MinimalPathTracer.depth")
    g.markOutput("MinimalPathTracer.emission")
    g.markOutput("AccumulatePass.output")
    g.markOutput("MinimalPathTracer.view")
    g.markOutput("MinimalPathTracer.raypos")
    # g.markOutput("MinimalPathTracer.intersect0")
    # g.markOutput("MinimalPathTracer.intersect1")
    # g.markOutput("MinimalPathTracer.intersect2")
    for i in range(9):

        g.markOutput("VBufferRT2.sphere_{}".format(i))
        g.markOutput("VBufferRT2.wdepth_{}".format(i))
        g.markOutput("VBufferRT2.debug_{}".format(i))

    
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
