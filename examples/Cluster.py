import sys
import time
import math
import kdt
# pyGraphViz is a graph drawing library. Putting import in a catch statement so the rest of the
# script is usable on machines that don't have it.
try:
	import pygraphviz as pgv
except ImportError:
	pass


directed = False

#parse arguments
if (len(sys.argv) < 4):
	print "Usage: python %s graph.mtx algorithm k [graph_image_filename] [-exp=2] [-inf=2]"%(sys.argv[0])
	sys.exit()
	
inmatrixfile = sys.argv[1]
algorithm = sys.argv[2]
k = int(sys.argv[3])
outfile = None
markovExpansion = 2
markovInflation = 2

# parse the optional arguments
for i in range(4, len(sys.argv)):
	a = sys.argv[i]
	if a.startswith("-exp="):
		markovExpansion = int(a.lstrip("-exp="))
	elif a.startswith("-inf="):
		markovInflation = float(a.lstrip("-inf="))
	else:
		outfile = a

colors = ["red","blue2","green","orange","purple","yellow","cyan", "brown"]

def getClusterColor(c):
	try:
		return colors[int(c)]
	except IndexError:
		return "grey4"

def draw(G, outfile, copyLocationFrom = None, copyFromIndexLookup = None, directed = False, selfLoopsOK = False, cluster=None):
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
		color = getClusterColor(cluster[i])
		if copyLocationFrom != None:
			DG.add_node(i, label="", color=color, width=0.1, height=0.1, pin=True, pos=copyLocationFrom.get_node(int(copyFromIndexLookup[i])).attr["pos"]);
		else:
			DG.add_node(i, label="", color=color, width=0.1, height=0.1);
		
	# add edges
	for i in range(0, m):
		if selfLoopsOK or (not selfLoopsOK and int(iv[i]) != int(jv[i])):
			if cluster[int(iv[i])] == cluster[int(jv[i])]:
				# both edge endpoints are in the same cluster, so draw the edge in the cluster color
				color = getClusterColor(cluster[int(iv[i])])
			else:
				# endpoints in different clusters, so draw them in a non-cluster color
				color = "grey70"
			DG.add_edge(int(iv[i]), int(jv[i]), color=color)
		
	if kdt.master():
		print "Graph created. %d nodes, %d edges. Doing layout..."%(DG.number_of_nodes(), DG.number_of_edges())
	
	#draw graph
	if copyLocationFrom != None:
		DG.layout(prog="neato", args="-n")
	else:
		DG.layout(prog="neato")
	
	if kdt.master():
		print "Writing to %s..."%(outfile)
		DG.draw(outfile)
	return DG

def calcModularity(G, group):
	# using modularity definition from page 4 of dimacs10-rules.pdf
	# modularity(Clustering) =
	#   (Sum of weights of edges belonging to a cluster)/(Sum of all edge weights)
	#   - ( (sum over all clusters ((sum of all vertex weights in Cluster)**2))
	#     / (4 * (sum of all edge weights)**2) )
	
	sumAllEdgeWeights = G.e.reduce(kdt.Mat.Column, kdt.op_add).reduce(kdt.op_add)

	EC = G.copy()
	EC.e.scale(group, op=(lambda x, y: y), dir=kdt.Mat.Row)
	EC.e.scale(group, op=(lambda x, y: x==y), dir=kdt.Mat.Column)
	# EC now contains 1 for edges in a cluster, 0 otherwise
	ECW = G.e.eWiseApply(EC.e, op=(lambda x,y: x*y))
	# ECW has all edges in a cluster with their original weight, 0 for inter-cluster edges
	sumAllClusterEdgeWeights = ECW.reduce(kdt.Mat.Column, kdt.op_add).reduce(kdt.op_add)
	
	# find (sum over all clusters ((sum of all vertex weights in Cluster)**2)
	V = kdt.Mat(group, kdt.Vec.range(G.nvert()), G.v, G.nvert())
	intraClusterVSums = V.reduce(kdt.Mat.Row, kdt.op_add)
	# square the sums
	intraClusterVSums.apply(lambda x: x*x)
	clusterV2Sum = intraClusterVSums.reduce(kdt.op_add)
	
	return sumAllClusterEdgeWeights/sumAllEdgeWeights - (clusterV2Sum / (4*sumAllEdgeWeights*sumAllEdgeWeights))
	
def getNumClusters(group):
	"""
	returns the number of individual clusters actually present in the group.
	"""
	V = kdt.Mat(group, kdt.Vec.range(len(group)), 1, len(group))
	sum = V.reduce(kdt.Mat.Row, kdt.op_add, init=0)
	return sum.count(lambda x: x != 0)

# load
G = kdt.DiGraph.load(inmatrixfile)
G.v = kdt.Vec.ones(G.nvert())
#print "loaded:",G

# time the clustering
startTime = time.time()
if algorithm == "markov":
	if kdt.master():
		print "running Markov clustering (expansion=%d, inflation=%.1f)..."%(markovExpansion,markovInflation)
	clus, markovG = G.cluster('Markov', addSelfLoops=True, expansion=markovExpansion, inflation=markovInflation, prunelimit=0.00001)
	algorithm = "markov(exp=%d,inf=%.1f)"%(markovExpansion,markovInflation)

elif algorithm == "spectral":
	if kdt.master():
		print "spectral clustering not implemented yet."
	sys.exit()
elif algorithm == "agglomerative":
	# use pagerank to find cluster roots
	if kdt.master():
		print "running PageRank..."
	pg = G.pageRank()
	if pg.count(lambda x: math.isnan(x)) > 0:
		# pagerank didn't converge (BUG in PageRank!)
		# select roots randomly
		if kdt.master():
			print "PageRank failed. selecting roots randomly..."
		r = kdt.Vec.range(G.nvert())
		r.randPerm()
		roots = kdt.Vec(k, sparse=False)
		for i in range(k):
			roots[i] = r[i]
	else:
		if kdt.master():
			print "picking",k,"with largest PageRank scores as roots..."
		pg.apply(kdt.op_negate)
		perm = pg.sort()
		roots = kdt.Vec(k, sparse=False)
		for i in range(k):
			roots[i] = perm.findInds(lambda x: round(x) == i)[0]

	if kdt.master():
		print "agglomerating..."
	clus = G.cluster('agglomerative', roots=roots)
else:
	if kdt.master():
		print "unrecognized algorithm:",algorithm
	sys.exit()
elapsedTime = time.time() - startTime
	
# `clus` now holds cluster roots.
# convert roots to cluster numbers (i.e. in the range [0,k) )
groups = kdt.DiGraph.convClusterParentToGroup(clus)

# compute the modularity using the DIMACS formula
modularity = calcModularity(G, groups)
numClusters = getNumClusters(groups)
nproc = kdt._nproc()

# print results
if kdt.master():
	print "Results: %s\t p=\t%d\t (k,clusters)=\t%d,%d\t alg=\t%s\t time(s)=\t%f\t modularity=\t%f"%(inmatrixfile, nproc, k, numClusters, algorithm, elapsedTime,modularity)

if outfile is not None:
	if kdt.master():
		print "drawing..."
	draw(G, outfile, directed=directed, cluster=groups)
