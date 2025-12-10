from falcor import *

def render_graph_VBufferRT():
    g = RenderGraph("VBufferRT")
    g.addPass(createPass("VBufferRT"), "VBufferRT")

    g.markOutput("VBufferRT.vbuffer")
    g.markOutput("VBufferRT.vbuffer", TextureChannelFlags.Alpha)
    g.markOutput("VBufferRT.depth")
    g.markOutput("VBufferRT.mvec")
    g.markOutput("VBufferRT.viewW")
    g.markOutput("VBufferRT.mask")
    g.markOutput("VBufferRT.albedo0")
    g.markOutput("VBufferRT.albedo")
    g.markOutput("VBufferRT.albedo1")
    g.markOutput("VBufferRT.albedo2")
    g.markOutput("VBufferRT.wdepth")
    g.markOutput("VBufferRT.depth0")
    g.markOutput("VBufferRT.depth1")
    g.markOutput("VBufferRT.depth2")
    g.markOutput("VBufferRT.referencealbedo")
    g.markOutput("VBufferRT.referencedepth")
    g.markOutput("VBufferRT.gdebug")
    

    return g

VBufferRT = render_graph_VBufferRT()
try: m.addGraph(VBufferRT)
except NameError: None
