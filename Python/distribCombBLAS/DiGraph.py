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


#	creates a breadth-first search tree of a Graph from a starting
#	set of vertices.  Returns a 1D array with the parent vertex of 
#	each vertex in the tree; unreached vertices have parent == -Inf.
#
def bfsTree(G, start):
	parents = pcb.DenseParVec(G.nverts()) - 1;
	# NOTE:  values in fringe go from 1:n instead of 0:(n-1) so can
	# distinguish vertex0 from empty element
	fringe = pcb.SpParVec(G.nverts());
	parents[start] = start;
	fringe[start] = start+1;
	while fringe.getnnz() > 0
		#FIX:  following line needed?
		fringe = pcb.range(fringe);	# or fringe.setNumToInd();
		G.SpMV_SelMax_inplace(fringe);	
		pcb.EWiseMult_inplacefirst(fringe, parents, True, -1);
		parents[fringe-1] = 0
		parents += fringe-1;
	parents = int(parents);		#FIX: needed?  how done?
	return parents;


	# returns tuples with elements
	# 0:  True/False of whether it is a BFS tree or not
	# 1:  levels of each vertex in the tree (-1 if not reached)
def isBfsTree(G, root, parents):

	ret = 1;	# assume valid
	nrowG = G.getnrow();

	# calculate level in the tree for each vertex; root is at level 0
	# about the same calculation as bfsTree, but tracks levels too
	parents2 = pcb.pyDenseParVec(nrowG, -1);
	fringe = pcb.pySpParVec(nrowG);
	parents2.SetElement(root,root);
	fringe.SetElement(root,root);
	levels = pcb.pyDenseParVec(nrowG, -1);
	levels.SetElement(root,0);

	level = 1;
	while fringe.getnnz() > 0:
		fringe.setNumToInd();
		G.SpMV_SelMax_inplace(fringe);
		pcb.EWiseMult_inplacefirst(fringe, parents2, True, -1);
		#fringe.printall();
		parents2.ApplyMasked_SetTo(fringe,0);
		parents2.add(fringe);
		levels.ApplyMasked_SetTo(fringe,level);
		level += 1;
	
	# spec test #1
	#	Not implemented
	

	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
#FIX: need nonzero() of next stmt?
	treeEdges = ((parents <> -1) & (parents <> root));  
	treeI = parents[treeEdges]
	#treeJ = 1..nrowG[treeEdges]
	treeJ = pcb.pyDenseParVec.range(nrowG,0)[treeEdges];
	if any(levels[treeI]-levels[treeJ] <> -1):
		ret = -1;

	

	del G, parents, fringe, levels, treeEdges, treeI, treeJ
	
	return (ret, levels)

#ToDo:  move bc() here from KDT.py
