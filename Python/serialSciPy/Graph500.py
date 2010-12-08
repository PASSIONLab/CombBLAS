import time
import scipy as sc
import DiGraph as kdtd
#from DiGraph import DiGraph

def k2Validate(G, start, parents, levels):
	good = True;
	[[Gi,Gj],Gv] = G._toArrays(G.spmat)
	root = (levels==0).nonzero()[0][0];
	nverts = sc.shape(parents)[0];

	
	# Spec test #1:
	# confirm that the tree is a tree;  i.e., that it does not have any 
	# cycles and that every vertex with a parent is in the tree
	visited = sc.zeros(sc.shape(parents)[0]);
	visited[root] = 1;
	fringe = (root, );
	cycle = False;
	firstTime = True;
	while len(fringe) <> 0 and not cycle:		#ToDo:  n^2 algorithm here
		newfringe = [];
		for i in range(len(fringe)):
			newvisits = (parents==fringe[i])
			if firstTime:		# handle parents[root] == root cycle
				firstTime = False;
				newvisits[root] = False;
			newvisits = newvisits.nonzero();
			if sc.any(visited[newvisits]):
				cycle = True;
				break;
			visited[newvisits] = 1;
			newfringe += newvisits[0].tolist();
		fringe = newfringe;
	if cycle:
		print "Cycle detected"; 
		good = False;
	
	# Spec test #2:  
	# every tree edge connects vertices whose BFS levels differ by 1
	treeEdges = (parents <> -1) & (parents <> sc.arange(nverts))
	treeI = parents[treeEdges].astype(int);
	treeJ = sc.arange(sc.shape(treeEdges)[0])[treeEdges];
	if sc.any(levels[treeI]-levels[treeJ] <> -1):
		print "The levels of some tree edges' vertices differ by other than 1"


	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	if sc.any((parents <> -1) & (visited <> 1)):
		print "The levels of some input edges' vertices differ by more than 1"
		good = False;

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges either
	# have both endpoints in the tree or not in the tree)
	li = levels[Gi]; 
	lj = levels[Gj];
	neither_in = (li == -1) & (lj == -1);
	both_in = (li > -1) & (lj > -1);
	if sc.any(sc.logical_not(neither_in | both_in)):
		print "The levels of some input edges' vertices differ by more than 1"
		good = False;

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph
	respects = abs(li-lj) <= 1
	if sc.any(sc.logical_not(neither_in | respects)):
		print "At least one vertex and its parent are not joined by an original edge"
		good = False;

	return good;



scale = 6;
edgefactor = 16;
edges = kdtd.Graph500Edges(scale, edgefactor);

before = time.clock()
edges.Graph500CleanEdges();
G = kdtd.DiGraph(edges,(2**scale,2**scale));
K1elapsed = time.clock() - before;

deg3verts = sc.array(sc.nonzero(G.degree().flatten() > 2)).flatten();	#indices of vertices with degree > 2
nstarts = 4;
starts = sc.unique((sc.floor(sc.rand(nstarts*2)*sc.shape(deg3verts)[0])).astype(int))
K2elapsed = 0;
K2edges = 0;
for start in starts[:nstarts]:
	before = time.clock();
	[parents, levels] = kdtd.bfsTree(G, deg3verts[start]);
	K2elapsed += time.clock() - before;
	if not k2Validate(G, deg3verts[start], parents, levels):
		print "Invalid BFS tree generated";
		break;
	K2edges += sc.shape((parents[edges.verts()[0]] <> -1).nonzero())[1];


print 'Graph500 benchmark run for scale = %2i' % scale
print 'Kernel 1 time = %8.4f seconds' % K1elapsed
print 'Kernel 2 time = %8.4f seconds' % K2elapsed
print '                    %8.4f seconds for each of %i starts' % (K2elapsed/nstarts, nstarts)
print 'Kernel 2 TEPS = %7.4e' % (K2edges/K2elapsed)
