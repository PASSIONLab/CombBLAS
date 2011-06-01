import sys
import kdt
import pygraphviz as pgv

layouts = ['neato','dot','twopi','circo','fdp','nop']
layout = 'neato'
#layout = 'dot'

#parse arguments
if (len(sys.argv) > 1):
	outfile = sys.argv[1]
	for i in range(1, len(sys.argv)):
		a = sys.argv[i]
		if a == '-l':
			layout = sys.argv[i+1]
else:
	print "Usage: python %s graph_image_filename [-l layout]"%(sys.argv[0])
	print "   -l layout  : layout style. default is '%s'. Available layouts: %s. neato and fdp seem to work best."%(layout, str(layouts))
	sys.exit()
	
	
	
DG = kdt.DiGraph()
DG.genGraph500Edges(8)
#DG._spm.load("bfwb62.mtx")
DG._spm.Apply(kdt.pyCombBLAS.set(1))
DG.removeSelfLoops()

[iv, jv, vv] = DG.toParVec()

n = DG.nvert()
m = len(iv)

G = pgv.AGraph(directed=False)
G.graph_attr["outputorder"] = "edgesfirst"

# add vertices
for i in range(0, n):
	G.add_node(i, label="", color="blue3", width=0.1, height=0.1);
	
# add edges
for i in range(0, m):
	G.add_edge(int(iv[i]), int(jv[i]), color="orange")
	
print "Graph created. %d nodes, %d edges. Doing layout..."%(G.number_of_nodes(), G.number_of_edges())

#draw graph
G.layout(prog=layout)
#f = "l_" + layout + "_" + outfile
print "Writing to %s..."%(outfile)
G.draw(outfile)

#########################################################################
print "Clustering.."
C = DG._markov(addSelfLoops=True, expansion=3, inflation=2, prunelimit=0.00001)
C.removeSelfLoops()

[iv, jv, vv] = C.toParVec()
m = len(iv)

G2 = pgv.AGraph(directed=False)
G2.graph_attr["outputorder"] = "edgesfirst"

# add vertices
for i in range(0, n):
	G2.add_node(i, label="", color="blue3", width=0.1, height=0.1, pin=True, pos=G.get_node(i).attr["pos"]);
	
# add edges
for i in range(0, m):
	G2.add_edge(int(iv[i]), int(jv[i]), color="orange")
	
print "Graph created. %d nodes, %d edges. Doing layout..."%(G.number_of_nodes(), G.number_of_edges())

#draw graph
G2.layout(prog="neato", args="-n")
#f = "l_" + layout + "_" + outfile

outfile = outfile.replace(".", "-clustered.")
print "Writing to %s..."%(outfile)
G2.draw(outfile)
