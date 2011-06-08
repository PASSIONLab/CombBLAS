import sys
import kdt
import pygraphviz as pgv


#parse arguments
if (len(sys.argv) < 2):
	print "Usage: python %s graph.mtx graph_image_filename"%(sys.argv[0])
	sys.exit()
	
inmatrixfile = sys.argv[1]
outfile = sys.argv[2]

def draw(G, outfile, copyLocationFrom = None, directed = False):
	"""
	Draws the graph G using pyGraphViz and saves the result to outfile.
	If copyLocationFrom is a pyGraphViz graph. If it is not None, then the
	position of each vertex in G is copied from its counterpart in
	copyLocationFrom. This is so that the same vertex occupies the same location
	on multiple graphs (vertecies line up). Directed specifies if the graph is 
	directed (show arrows and self loops) or not.
	"""
	[iv, jv, vv] = G.toParVec()
	n = G.nvert()
	m = len(iv)
	
	DG = pgv.AGraph(directed=directed)
	DG.graph_attr["outputorder"] = "edgesfirst"
	
	# add vertices
	for i in range(0, n):
		if copyLocationFrom != None:
			DG.add_node(i, label="", color="blue3", width=0.1, height=0.1, pin=True, pos=copyLocationFrom.get_node(i).attr["pos"]);
		else:
			DG.add_node(i, label="", color="blue3", width=0.1, height=0.1);
		
	# add edges
	for i in range(0, m):
		DG.add_edge(int(iv[i]), int(jv[i]), color="orange")
		
	print "Graph created. %d nodes, %d edges. Doing layout..."%(DG.number_of_nodes(), DG.number_of_edges())
	
	#draw graph
	if copyLocationFrom != None:
		DG.layout(prog="neato", args="-n")
	else:
		DG.layout(prog="neato")
	
	print "Writing to %s..."%(outfile)
	DG.draw(outfile)
	return DG


	
bigG = kdt.DiGraph.load(inmatrixfile)
bigG._spm.Apply(kdt.pyCombBLAS.set(1))
#G.removeSelfLoops()

print "drawing the original graph:"
OrigVertLocSource = draw(bigG, outfile.replace(".", "-1-original."), None, directed=True)

print "Finding the largest component:"
comp = bigG.getLargestComponent()
OrigVertLocSource = draw(comp, outfile.replace(".", "-2-largestcomp."), None, directed=True)
G = comp

print "Clustering:"
markovG = G._markov(addSelfLoops=True, expansion=3, inflation=3, prunelimit=0.00001)
markovG.removeSelfLoops()
draw(markovG, outfile.replace(".", "-3-clusters."), OrigVertLocSource, directed=False)
clus = markovG.connComp()

print "Contracting:"
print "clusters:",clus
smallG = G.contract(groups=clus, collapseInto=clus)
draw(smallG, outfile.replace(".", "-4-contracted."), None, directed=True)
