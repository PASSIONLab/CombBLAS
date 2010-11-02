import numpy as np
import scipy as sc
import scipy.sparse as sp
import Graph as gr
import SpParVec as spv
import SpParMat as spm

class DiGraph(gr.Graph):

	print "in DiGraph"

	def __init__(self, edgev, size):
		self.spmat = spm.SpParMat(edgev, shape=size);

	def degree(self):
		return self.indegree() + self.outdegree();

	def indegree(self):
		tmp = spm.reduce(self._spones(self.spmat), 0, +);
		return tmp;

	def outdegree(self):
		tmp = spm.reduce(self._spones(self.spmat), 1, +);
		return tmp;

		

class DiEdgeV(gr.EdgeV):
	print "in DiEdgeV"

		

#	No VertexV class for now

#class VertexV():
#	print "in VertexV"
#
#	def __init__(self, ndces):


def torusEdges(n):
	N = n*n;
	#old nvec = sc.tile(sc.arange(n),(n,1)).T.flatten();	# [0,0,0,...., n-1,n-1,n-1]
	#old nvecil = sc.tile(sc.arange(n),n)			# [0,1,...,n-1,0,1,...,n-2,n-1]
	nvec = DPV.tile(DPV.range(n), n, interleave=False)	# NEW:  range() as in Py
											# NEW:  tile() as in SciPy (but just 1D), with
	nvecil = DPV.tile(DPV.range(n), n, interleave=True)	#    interleave arg
	north = gr.Graph._sub2ind((n,n),DPV.mod(nvecil-1,n),nvec);	
	south = gr.Graph._sub2ind((n,n),DPV.mod(nvecil+1,n),nvec);
	west = gr.Graph._sub2ind((n,n),nvecil, DPV.mod(nvec-1,n));
	east = gr.Graph._sub2ind((n,n),nvecil, DPV.mod(nvec+1,n));
	Nvec = DPV.range(N);
	row = DPV.append(Nvec, Nvec);
	row = DPV.append(row, row);
	col = DPV.append(north, west);					# NEW:  append() in just 1D
	col = DPV.append(col, south);
	col = DPV.append(rowcol, east);
	return gr.EdgeV((row, col), DPV.ones(N*4))

def Graph500Edges(n):						# NOTE: Not changed yet from SciPy version
	print "NOTE:  Graph500Edges producing torusEdges currently"
	N = n*n;
	nvec = sc.tile(sc.arange(n),(n,1)).T.flatten();	# [0,0,0,...., n-1,n-1,n-1]
	nvecil = sc.tile(sc.arange(n),n)			# [0,1,...,n-1,0,1,...,n-2,n-1]
	north = gr.Graph._sub2ind((n,n),sc.mod(nvecil-1,n),nvec);
	south = gr.Graph._sub2ind((n,n),sc.mod(nvecil+1,n),nvec);
	west = gr.Graph._sub2ind((n,n),nvecil, sc.mod(nvec-1,n));
	east = gr.Graph._sub2ind((n,n),nvecil, sc.mod(nvec+1,n));
	Nvec = sc.arange(N);
	rowcol = sc.append((Nvec, north), (Nvec, west), axis=1)
	rowcol = sc.append(rowcol,        (Nvec, south), axis=1)
	rowcol = sc.append(rowcol,        (Nvec, east), axis=1)
	rowcol = rowcol.T
	rowcol = (rowcol[:,0], rowcol[:,1]);
 	return gr.EdgeV(rowcol, sc.tile(1,(N*4,)))


#	creates a breadth-first search tree of a Graph from a starting set of vertices
#	returns a 1D array with the parent vertex of each vertex in the tree; unreached vertices have parent == -Inf
#        and a 1D array with the level at which each vertex was first discovered (-2 if not in the tree)

def bfsTree(G, starts):
	#old parents = -2*sc.ones(G.spmat.shape[0]).astype(int);
	parents = spv.zeros(G.nverts()) - 2;				# constructor and -=
	#old levels = np.copy(parents);
	levels = parents;					# copy() (or "=") DPV->DPV (NEW) 
	#old newverts = np.copy(starts);
	newverts = spv.SpParVec(starts);				# need constructor (or "=") from scalar (NEW)
	parents[newverts] = -1;						# overload __setitem__ (NEW)
	levels[newverts] = 0;
	#fringe = np.array([newverts]);
	fringe = newverts;
	level = 1;
	#old while len(fringe) > 0:
	while len(fringe) > 0							# 'len' overloaded for SPVs
		#old colvec = sc.zeros((G.nverts(),));
		colvec = spv.zeros(G.nverts());
		# +1 to deal with 0 being a valid vertex ID
		#old colvec[fringe] = fringe+1; 
		colvec[fringe] = fringe+1
		#old cand = gr.Graph._SpMV_times_max(G.spmat, colvec)
		cand = gr.Graph._SpMV_SelMax(G.spmat, colvec);	# 
		#old newverts = np.array(((cand.toarray().flatten() <> 0) & (parents == -2)).nonzero()).flatten();
		tmp1 = ~(spv.bool(cand)) &  spv.bool(parents+2) );		
		newverts = tmp1.nonzero();				
		#old if len(newverts) > 0:
		if spv.reduce(~(spv.bool(newverts)), +)
			#old parents[newverts] = cand[newverts].todense().astype(int) - 1;
			parents[newverts] = cand[newverts] - 1;
			levels[newverts] = level;
		level += 1;
		fringe = newverts;
	#old parents = parents.astype(int);
	parents = int(parents);
	return (parents, levels);



#ToDo:  move bc() here from KDT.py
