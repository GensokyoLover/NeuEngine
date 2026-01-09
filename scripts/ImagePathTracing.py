from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 1,"useTraceRayInline":True,})
    g.addPass(VBufferRT, "VBufferRT")
    g.markOutput("VBufferRT.viewW")
    g.markOutput("VBufferRT.debug")
    g.markOutput("VBufferRT.depth0")
    g.markOutput("VBufferRT.depth1")
    g.markOutput("VBufferRT.depth2")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
