import time
import scipy as sc
import DiGraph as kdtd
#from DiGraph import DiGraph

def k2Validate(G, start, parents, levels):
	good = True;
	[[Gi,Gj],Gv] = G._toArrays(G.spmat)
	root = (levels==0).nonzero()[0][0];

	
	# Spec test #1:
	# confirm that the tree is a tree;  i.e., that it does not have any 
	# cycles and that every vertex with a parent is in the tree
	visited = sc.zeros(sc.shape(parents)[0]);
	visited[root] = 1;
	fringe = (root, );
	cycle = False;
	while len(fringe) <> 0 and not cycle:		#ToDo:  n^2 algorithm here
		newfringe = [];
		for i in range(len(fringe)):
			newvisits = (parents==fringe[i]).nonzero();
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
	treeEdges = ((parents <> -2) & (parents <> -1))
	treeI = parents[treeEdges].astype(int);
	treeJ = sc.arange(sc.shape(treeEdges)[0])[treeEdges];
	if any(levels[treeI]-levels[treeJ] <> -1):
		print "The levels of some tree edges' vertices differ by other than 1"


	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	if any((parents <> -2) & (visited <> 1)):
		print "The levels of some input edges' vertices differ by more than 1"
		good = False;


	return good;



scale = 4;
edges = kdtd.Graph500Edges(scale);

before = time.clock()
G = kdtd.DiGraph(edges,(2**scale,2**scale));
K1elapsed = time.clock() - before;

deg3verts = sc.array(sc.nonzero(G.degree().flatten() > 2)).flatten();	#indices of vertices with degree > 2
nstarts = 4;
starts = sc.unique((sc.floor(sc.rand(nstarts*2)*sc.shape(deg3verts)[0])).astype(int))
K2elapsed = 0;
for start in starts[:nstarts]:
	before = time.clock();
	[parents, levels] = kdtd.bfsTree(G, deg3verts[start]);
	K2elapsed += time.clock() - before;
	if not k2Validate(G, deg3verts[start], parents, levels):
		print "Invalid BFS tree generated blah blah";
		break;


print 'Graph500 benchmark run for scale = %2i' % scale
print 'Time for kernel 1 = %8.4f seconds' % K1elapsed
print 'Time for kernel 2 = %8.4f seconds' % K2elapsed
print '                    %8.4f seconds for each of %i starts' % (K2elapsed/nstarts, nstarts)
#ToDo:  calculate TEPS and print it out