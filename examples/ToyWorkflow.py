import sys
import kdt
#import pygraphviz as pgv


#parse arguments
if (len(sys.argv) < 2):
	print "Usage: python %s graph.mtx graph_image_filename"%(sys.argv[0])
	sys.exit()
	
inmatrixfile = sys.argv[1]
outfile = sys.argv[2]

def draw(G, copyLocationFrom, outfile, directed = False):
	[iv, jv, vv] = G.toParVec()
	m = len(iv)
	
	DG = pgv.AGraph(directed=directed)
	DG.graph_attr["outputorder"] = "edgesfirst"
	
	# add vertices
	for i in range(0, n):
		if copyLocationFrom != None:
			pos = copyLocationFrom.get_node(i).attr["pos"]
		else:
			pos = None

		DG.add_node(i, label="", color="blue3", width=0.1, height=0.1, pin=True, pos=pos);
		
	# add edges
	for i in range(0, m):
		DG.add_edge(int(iv[i]), int(jv[i]), color="orange")
		
	print "Graph created. %d nodes, %d edges. Doing layout..."%(DG.number_of_nodes(), DG.number_of_edges())
	
	#draw graph
	if copyLocationFrom == None:
		DG.layout(prog="neato")
	else:
		DG.layout(prog="neato", args="-n")
	
	print "Writing to %s..."%(outfile)
	DG.draw(outfile)
	return DG


	
G = kdt.DiGraph()
#G.genGraph500Edges(8)

G._spm.load(inmatrixfile)
G._spm.Apply(kdt.pyCombBLAS.set(1))
#G.removeSelfLoops()

print "drawing the original graph:"
#OrigVertLocSource = draw(G, None, outfile.replace(".", "-1-original."), directed=True)

print "Finding the largest component:"
Comp = G._getLargestComponent()
#OrigVertLocSource = draw(Comp, None, outfile.replace(".", "-2-largestcomp."), directed=True)
G = Comp

print "Clustering:"
C = G._markov(addSelfLoops=True, expansion=2, inflation=2, prunelimit=0.00001)
C.removeSelfLoops()
#draw(C, OrigVertLocSource, outfile.replace(".", "-3-clusters."), directed=False)

print "Collapsing:"
#draw(C, OrigVertLocSource, outfile.replace(".", "-4-collapsed."), directed=True)
