import sys
import kdt
import pygraphviz as pgv

directed = False

#parse arguments
if (len(sys.argv) < 2):
	print "Usage: python %s graph.mtx graph_image_filename"%(sys.argv[0])
	sys.exit()
	
inmatrixfile = sys.argv[1]
outfile = sys.argv[2]

def draw(G, outfile, copyLocationFrom = None, copyFromIndexLookup = None, directed = False, selfLoopsOK = False):
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
	
	if copyFromIndexLookup is None:
		copyFromIndexLookup = range(0,n)
	
	DG = pgv.AGraph(directed=directed)
	DG.graph_attr["outputorder"] = "edgesfirst"
	
	# add vertices
	for i in range(0, n):
		if copyLocationFrom != None:
			DG.add_node(i, label="", color="blue3", width=0.1, height=0.1, pin=True, pos=copyLocationFrom.get_node(int(copyFromIndexLookup[i])).attr["pos"]);
		else:
			DG.add_node(i, label="", color="blue3", width=0.1, height=0.1);
		
	# add edges
	for i in range(0, m):
		if selfLoopsOK or (not selfLoopsOK and int(iv[i]) != int(jv[i])):
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


# load
bigG = kdt.DiGraph.load(inmatrixfile)
bigG.ones()

print "drawing the original graph:"
OrigVertLocSource = draw(bigG, outfile.replace(".", "-1-original."), None, directed=directed)

print "Finding the largest component:"
comp = bigG.connComp(sym=False)
giantComp = comp.hist().argmax()
G = bigG.subgraph(mask=(comp==giantComp))

# Get the indices of the vertices by getting the indicies of the nonzeros in the mask.
# These indices are used to make sure the vertices in the subgraph appear at the same x,y positions
# as they did in the original plot.
compInds = kdt.DiGraph.convMaskToIndices(comp==giantComp)
OrigVertLocSource = draw(G, outfile.replace(".", "-2-largestcomp."), OrigVertLocSource, copyFromIndexLookup=compInds, directed=directed)

print "Clustering:"
clus, markovG = G.cluster('Markov', addSelfLoops=True, expansion=3, inflation=3, prunelimit=0.00001)
draw(markovG, outfile.replace(".", "-3-clusters."), OrigVertLocSource, directed=False)

print "Contracting:"
smallG = G.contract(clusterParents=clus)

# Make a lookup table to convert a contracted vertex number into its old cluster parent so it can
# use the same position in the graph.
clusterGroup, perm = kdt.DiGraph.convClusterParentToGroup(clus, retInvPerm=True)

draw(smallG, outfile.replace(".", "-4-contracted."), OrigVertLocSource, copyFromIndexLookup=perm, directed=directed)
