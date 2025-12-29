from falcor import *

def render_graph_ImpostorTracer():
    g = RenderGraph("ImpostorTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    ImpostorTracer = createPass("ImpostorTracer", {'maxBounces': 3})
    g.addPass(ImpostorTracer, "ImpostorTracer")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("ImpostorTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

ImpostorTracer = render_graph_ImpostorTracer()
try: m.addGraph(ImpostorTracer)
except NameError: None
