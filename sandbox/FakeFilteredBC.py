import sys
import kdt
import pygraphviz as pgv
import kdt.pyCombBLAS as pcb

directed = False

#parse arguments
#if (len(sys.argv) < 2):
#	print "Usage: python %s graph.mtx graph_image_filename"%(sys.argv[0])
#	sys.exit()
	
inmatrixfile = "network.mtx" #sys.argv[1]
outfile = "FilteredBC.eps" #sys.argv[2]

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
	type1 = kdt.DiGraph()
	type1._spm.GenGraph500Edges(scale, None, 7, False, 0.60, 0.19, 0.16, 0.05)
	type1._spm.Apply(pcb.set(1))
	# type 2 is email: fairly even
	type2 = kdt.DiGraph()
	type2._spm.GenGraph500Edges(scale, None, 4, False, 0.20, 0.25, 0.25, 0.30)
	type2._spm.Apply(pcb.set(2))
	
	G = type1 + type2
	print "Writing generated network"
	G.save(inmatrixfile)

#generate()

# load
print "Reading network"
G = kdt.DiGraph.load(inmatrixfile)
Gorig = G
#G.spOnes()

#G.save("test.mtx")

print "drawing the original graph:"
OrigVertLocSource = draw(G, outfile.replace(".", "-1-original."), None, directed=directed)

print "Finding BC of original"
bc = G.centrality('exactBC',normalize=True)
G = kdt.DiGraph.load(inmatrixfile) # BC changes the values for some reason
draw(G, outfile.replace(".", "-2-origBC."), OrigVertLocSource, directed=False, sizes=bc)


print "Finding BC of edge type 1 and 3"
Gsub = kdt.DiGraph.load(inmatrixfile)
def f(x):
	return (x == 2)
Gsub.e._prune(pcb.unary(f))
#Gsub.addFilter(lambda x: x != 2)
bc = Gsub.centrality('exactBC',normalize=True)
draw(Gsub, outfile.replace(".", "-3-sub1BC."), OrigVertLocSource, directed=False, sizes=bc)


print "Finding BC of edge type 2 and 3"
Gsub = kdt.DiGraph.load(inmatrixfile)
def f(x):
	return (x == 1)
Gsub.e._prune(pcb.unary(f))
#Gsub.addFilter(lambda x: x != 1)
bc = Gsub.centrality('exactBC',normalize=True)
Gsub.e.apply(pcb.set(2)) # to make the draw function use the right color
draw(Gsub, outfile.replace(".", "-4-sub2BC."), OrigVertLocSource, directed=False, sizes=bc)

