import numpy as np
import scipy as sc
import scipy.sparse as sp
import Graph as gr

class DiGraph(gr.Graph):


	def __init__(self, edgev, size):
		#  Keeping EdgeV independent of the number of vertices touched by an edge 
		#  creates some complications with creating sparse matrices, as 
		#  scipy.sparse.csr_matrix()expects each of i/j/v to have shape (N,1) or 
		#  (1,N), requiring each of i/j/v to be flatten()ed before calling csr_matrix.
		#print "in DiGraph/__init__"
		# note: make graph symmetric
		fv = sc.hstack((edgev[1].flatten(),edgev[1].flatten()));
		fi = sc.hstack((edgev[0][0].flatten(),edgev[0][1].flatten()));
		fj = sc.hstack((edgev[0][1].flatten(),edgev[0][0].flatten()));
		# note:  for an edge, From is the col and To is the row
		self.spmat = sp.csr_matrix((fv, (fj,fi)), shape=size);

	def degree(self):
		return self.indegree() + self.outdegree();

	def indegree(self):
		tmp = self._spones(self.spmat);
		return np.asarray(tmp.sum(0))

	def outdegree(self):
		tmp = self._spones(self.spmat);
		return np.asarray(tmp.sum(1).reshape(1,np.shape(tmp)[0]));

		

class DiEdgeV(gr.EdgeV):
	pass;		#print "in DiEdgeV"

		

#	No VertexV class for now

#class VertexV():
#	print "in VertexV"
#
#	def __init__(self, ndces):


def torusEdges(n):
	N = n*n;
	nvec = sc.tile(sc.arange(n),(n,1)).T.flatten();	# [0,0,0,...., n-1,n-1,n-1]
	nvecil = sc.tile(sc.arange(n),n)			# [0,1,...,n-1,0,1,...,n-2,n-1]
	north = gr.Graph._sub2ind((n,n),sc.mod(nvecil-1,n),nvec);	
	south = gr.Graph._sub2ind((n,n),sc.mod(nvecil+1,n),nvec);
	west = gr.Graph._sub2ind((n,n),nvecil, sc.mod(nvec-1,n));
	east = gr.Graph._sub2ind((n,n),nvecil, sc.mod(nvec+1,n));
	Nvec = sc.arange(N);
	rowcol = sc.append((Nvec, north), (Nvec, west), axis=1);
	rowcol = sc.append(rowcol,        (Nvec, south), axis=1);
	rowcol = sc.append(rowcol,        (Nvec, east), axis=1);
	rowcol = rowcol.T;
	rowcol = (rowcol[:,0], rowcol[:,1]);
	return gr.EdgeV(rowcol, sc.tile(1,(N*4,)))

def _rmat_gen(scale, edgefactor, probA, probB, probC, probD):
	scale2 = 2**scale;
	nedge = scale2 * edgefactor;
	ii = sc.ones((nedge,1),dtype='int64');
	jj = sc.ones((nedge,1),dtype='int64');

	probAB = probA+probB;
	C_norm = probC/(probC+probD);
	A_norm = probA/(probA+probB);

	for ib in range(scale):
		ii_bit = sc.random.rand(nedge,1) > probAB;
		jj_bit = sc.random.rand(nedge,1) > ( C_norm * ii_bit + A_norm * sc.logical_not(ii_bit));

		ii = ii + 2**ib * ii_bit;
		jj = jj + 2**ib * jj_bit;

	ii -= 1;	# convert to zero-based indexing
	jj -= 1;

	return (ii.flatten(),jj.flatten());

def Graph500Edges(scale, edgefactor):
	rowcol = _rmat_gen(scale, edgefactor, 0.57, 0.19, 0.19, 0.05);
	
 	return gr.EdgeV(rowcol, sc.tile(1,(edgefactor*2**scale,)))


#	creates a breadth-first search tree of a Graph from a starting set of vertices
#	returns a 1D array with the parent vertex of each vertex in the tree; unreached vertices have parent == -Inf
#        and a 1D array with the level at which each vertex was first discovered (-1 if not in the tree)

def bfsTree(G, starts):
	parents = -1*sc.ones(G.spmat.shape[0]).astype(int);
	levels = np.copy(parents);
	newverts = np.copy(starts);
	parents[newverts] = newverts;	# parents[root] == root
	levels[newverts] = 0;
	fringe = np.array([newverts]);
	level = 1;
	while len(fringe) > 0:
		colvec = sc.zeros((G.nverts(),));
		# +1 to deal with 0 being a valid vertex ID
		colvec[fringe] = fringe+1; 
		cand = gr.Graph._SpMV_sel_max(G.spmat, colvec)
		newverts = np.array(((cand.toarray().flatten() <> -np.Inf) & (parents == -1)).nonzero()).flatten();
		if len(newverts) > 0:
			parents[newverts] = cand[newverts].todense().astype(int) - 1;
			levels[newverts] = level;
		level += 1;
		fringe = newverts;
	parents = parents.astype(int);
	return (parents, levels);



#ToDo:  move bc() here from KDT.py
