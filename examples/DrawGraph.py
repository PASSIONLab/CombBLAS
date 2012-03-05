import sys
import kdt
import pygraphviz as pgv
import kdt.pyCombBLAS as pcb

directed = True

outfile = "DrawGraph.png"


# graph configuration
i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
v = 1
nverts = 0 # gets increased to match largest edge endpoints listed in i,j
edgeFilter = None
vertexFilter = None



if len(i) != len(j):
	raise ValueError,"bad indices!"

iVec = kdt.Vec(len(i), sparse=False)
jVec = kdt.Vec(len(i), sparse=False)
vVec = kdt.Vec(len(i), sparse=False)
for ind in range(len(i)):
	iVec[ind] = i[ind] 
	jVec[ind] = j[ind]
	if isinstance(v, (int, float, long, kdt.Obj1, kdt.Obj2)):
		vVec[ind] = v
	else:
		vVec[ind] = v[ind]
	nverts = max(nverts, i[ind]+1)
	nverts = max(nverts, j[ind]+1)

print "graph has",nverts,"vertices"

def draw(G, outfile, directed = False, selfLoopsOK = False, sizes=None, labelInd=True):
	"""
	Draws the graph G using pyGraphViz and saves the result to outfile.
	Directed specifies if the graph is 
	directed (show arrows and self loops) or not.
	"""
	[jv, iv, vv] = G.e.toVec() # intentionally transposed
	n = G.nvert()
	m = len(iv)
	
	DG = pgv.AGraph(directed=directed)
	DG.graph_attr["outputorder"] = "edgesfirst"
	
	# add vertices
	for i in range(0, n):
		if sizes is None:
			size = 0.1
		else:
			size = sizes[i] + 0.05
		
		if labelInd:
			label = str(i)
		else:
			label=""
		DG.add_node(i, label=label, color="blue3", width=size, height=size);
		
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
	DG.layout(prog="neato")
	
	print "Writing to %s..."%(outfile)
	DG.draw(outfile)
	return DG

G = kdt.DiGraph(iVec, jVec, vVec, nverts)
if edgeFilter is not None:
	G.addEFilter(edgeFilter)
if vertexFilter is not None:
	G.addVFilter(vertexFilter)

draw(G, outfile, directed=directed)
