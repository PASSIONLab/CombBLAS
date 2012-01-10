import sys
import kdt
import pygraphviz as pgv
import kdt.pyCombBLAS as pcb

directed = False

#parse arguments
#if (len(sys.argv) < 2):
#	print "Usage: python %s graph.mtx graph_image_filename"%(sys.argv[0])
#	sys.exit()

shouldGenerate = 0
if len(sys.argv) >= 2:
	shouldGenerate = sys.argv[1]
shouldDraw = True
inmatrixfile = "FilteredBFSnetwork.mtx" #sys.argv[1]
outfile = "FilteredBFS.png" #sys.argv[2]

def draw(G, outfile, copyLocationFrom = None, copyFromIndexLookup = None, directed = False, selfLoopsOK = False, sizes=None):
	"""
	Draws the graph G using pyGraphViz and saves the result to outfile.
	If copyLocationFrom is a pyGraphViz graph. If it is not None, then the
	position of each vertex in G is copied from its counterpart in
	copyLocationFrom. This is so that the same vertex occupies the same location
	on multiple graphs (vertecies line up). Directed specifies if the graph is 
	directed (show arrows and self loops) or not.
	"""
	[iv, jv, vv] = G.e.toVec()
	n = G.nvert()
	m = len(iv)
	
	if copyFromIndexLookup is None:
		copyFromIndexLookup = range(0,n)
	
	DG = pgv.AGraph(directed=directed)
	DG.graph_attr["outputorder"] = "edgesfirst"
	
	# add vertices
	for i in range(0, n):
		if sizes is None:
			size = 0.1
		else:
			size = sizes[i] + 0.05
			
		if copyLocationFrom != None:
			DG.add_node(i, label="", color="blue3", width=size, height=size, pin=True, pos=copyLocationFrom.get_node(int(copyFromIndexLookup[i])).attr["pos"]);
		else:
			DG.add_node(i, label="", color="blue3", width=size, height=size);
		
	# add edges
	for i in range(0, m):
		if vv[i] == 1:
			color="orange"
		elif vv[i] == 2:
			color="grey"
		else:
			color="orange3"
		if selfLoopsOK or (not selfLoopsOK and int(iv[i]) != int(jv[i])):
			DG.add_edge(int(iv[i]), int(jv[i]), color=color)
		
	print "Graph created. %d nodes, %d edges. Doing layout..."%(DG.number_of_nodes(), DG.number_of_edges())
	
	#draw graph
	if copyLocationFrom != None:
		DG.layout(prog="neato", args="-n")
	else:
		DG.layout(prog="neato")
	
	print "Writing to %s..."%(outfile)
	DG.draw(outfile)
	return DG

# generate
def generate(scale=5):
	# type 1 is twitter: small number of heavy nodes
	type1 = kdt.DiGraph.generateRMAT(scale, element=1.0, edgeFactor=7, delIsolated=False, initiator=[0.60, 0.19, 0.16, 0.05])
	type1.e.apply(lambda x: 1)
	# type 2 is email: fairly even
	type2 = kdt.DiGraph.generateRMAT(scale, element=1.0, edgeFactor=4, delIsolated=False, initiator=[0.20, 0.25, 0.25, 0.30])
	type2.e.apply(lambda x: 2)
	
	G = type1
	G.e = type1.e + type2.e
	print "Writing generated network"
	G.save(inmatrixfile)

if shouldGenerate:
	generate(int(shouldGenerate))

# load
print "Reading network"
G = kdt.DiGraph.load(inmatrixfile)
Gorig = G

if shouldDraw:
	print "drawing the original graph:"
	OrigVertLocSource = draw(G, outfile.replace(".", "-1-original."), None, directed=directed)

print "Doing BFS on original"
start = 0
parents = G.bfsTree(start)
if shouldDraw:
	parents.apply(lambda x: 1/(x+2))
	draw(G, outfile.replace(".", "-2-origBC."), OrigVertLocSource, directed=False, sizes=parents)


print "Doing BFS on edge type 1 and 3"
def f1(x):
	return (x != 2)
G.e.addFilter(f1)
G.e.materializeFilter()
start = 0
parents = G.bfsTree(start)
if shouldDraw:
	parents.apply(lambda x: 1/(x+2))
	draw(G, outfile.replace(".", "-3-sub1BC."), OrigVertLocSource, directed=False, sizes=parents)
G.e.delFilter(f1)

print "Doing BFS on edge type 2 and 3"
def f2(x):
	return (x != 1)
G.e.addFilter(f2)
G.e.materializeFilter()
start = 0
parents = G.bfsTree(start)
G.e.delFilter(f2)
G.e.apply(lambda x: 2) # to make the draw function use the right color
if shouldDraw:
	parents.apply(lambda x: 1/(x+2))
	draw(G, outfile.replace(".", "-4-sub2BC."), OrigVertLocSource, directed=False, sizes=parents)

