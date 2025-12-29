from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("GBufferVisualize")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.markOutput("VBufferRT.viewW")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
