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
	visited = SPV.zeros(G.nverts());		# NEW:  hmmm;  could G.nverts return a funny object that 
									# causes zeros() to create a DPV without the "DPV."?
									# Or use zeros_like(parents) to do the same?
	visited[root] = 1;
	fringe = SPV.SpParVec(G.nverts());
	fringe[root] = 1;		
	cycle = False;
	#ToDo:  n^2 algorithm here
	while len(fringe.nonzero()) <> 0 and not cycle:		
		newfringe = SPV.zeros(G.nverts());
		for i in range(len(fringe.nonzero())):
			newvisits = (parents==fringe[i]).nonzero();	# NEW: overload of "=="
			if (visited[newvisits]).any():	# NEW:  SPV.any() needs to return a scalar
				cycle = True;
				break;
			visited[newvisits] = 1;
			newfringe[newvisits] = 1;
		fringe = newfringe;
	if cycle:
		print "Cycle detected"; 
		good = False;
	
	# Spec test #2:  
	# every tree edge connects vertices whose BFS levels differ by 1
	treeEdges = ((parents <> -2) & (parents <> -1))	# NEW:  ne() and and()
	#old treeI = parents[treeEdges].astype(int);
	treeI = int(parents[treeEdges]);
	treeJ = SPV.range(len(treeEdges)[0])[treeEdges];	
	if any(levels[treeI]-levels[treeJ] <> -1):
		print "The levels of some tree edges' vertices differ by other than 1"


	# Spec test #3:
	# every input edge has vertices whose levels differ by no more than 1
	if any((parents <> -2) & (visited <> 1)):
		print "The levels of some input edges' vertices differ by more than 1"
		good = False;

	# Spec test #4:
	# the BFS tree spans a connected component's vertices (== all edges either
	# have both endpoints in the tree or not in the tree)
	li = levels[Gi]; 
	lj = levels[Gj];
	neither_in = (li == -2) & (lj == -2);
	both_in = (li > -2) & (lj > -2);
	#old if any(sc.logical_not(neither_in | both_in)):	
	if any(SPV.logical_not(neither_in | both_in)):	# NEW:  not()
		print "The levels of some input edges' vertices differ by more than 1"
		good = False;

	# Spec test #5:
	# a vertex and its parent are joined by an edge of the original graph
	respects = abs(li-lj) <= 1					# NEW:  abs() (and binary "-"?)
	#old if any(sc.logical_not(neither_in | respects)):
	if any(SPV.logical_not(neither_in | respects)):
		print "At least one vertex and its parent are not joined by an original edge"
		good = False;

	return good;



scale = 4;
edges = kdtd.Graph500Edges(scale);

before = time.clock()
G = kdtd.DiGraph(edges,(2**scale,2**scale));
K1elapsed = time.clock() - before;

#old deg3verts = sc.array(sc.nonzero(G.degree().flatten() > 2)).flatten();	#indices of vertices with degree > 2
deg3verts = (G.degree() > 2).nonzero().toClient;	# NEW:  gt() and toClient()
								# moving deg3verts now makes code below simple SciPy code
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