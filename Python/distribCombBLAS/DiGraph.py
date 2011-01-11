import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import PyCombBLAS as PCB
import Graph as gr

class DiGraph(gr.Graph):

	print "in DiGraph"

	#FIX:  just building a Graph500 graph by default for now
	def __init__(self):
		self.spmat = pcb.pySpParMat();

	def degree(self):
		return self.indegree() + self.outdegree();

	def indegree(self):
		#tmp = spm.reduce(self._spones(self.spmat), 0, +);
		print "\n	DiGraph.indegree() not fully implemented!\n"
		tmp = 1;
		return tmp;

	def outdegree(self):
		#tmp = spm.reduce(self._spones(self.spmat), 1, +);
		print "\n	DiGraph.outdegree() not fully implemented!\n"
		tmp = 1;
		return tmp;

#def DiGraphGraph500():
#	self = DiGraph();
#	self.spmat = GenGraph500Edges(sc.log2(nvert));

		
class DiEdgeV(gr.EdgeV):
	print "in DiEdgeV"

		
class VertexV:
	print "in VertexV"

	def __init__(self, length):
		self.dpv = pcb.pyDenseParVec(length,0);

	def __getitem__(self, key):
		if type(key) == int:
			if key > self.dpv.len()-1:
				raise IndexError;
			ret = self.dpv.GetElement(key);
		else:
			print "__getitem__ only supports scalar subscript"
		return ret;

class SpVertexV:
	print "in SpVertexV"

	def __init__(self, length):
		self.spv = pcb.pySpParVec(length);

def genGraph500Candidates(G, howmany):
	tmpDpv = G.spmat.GenGraph500Candidates(howmany);
	tmpVV = VertexV(0);
	del tmpVV.dpv		# legal?  GC issue with doing this?
	tmpVV.dpv = tmpDpv;
	return tmpVV


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

def genGraph500Edges(self, scale):
	elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spmat, scale);
 	return elapsedTime;


#	creates a breadth-first search tree of a Graph from a starting
#	set of vertices.  Returns a 1D array with the parent vertex of 
#	each vertex in the tree; unreached vertices have parent == -Inf.
#
def bfsTree(G, start):
	parents = PCB.PyDenseParVec(G.nvert(), -1);
	# NOTE:  values in fringe go from 1:n instead of 0:(n-1) so can
	# distinguish vertex0 from empty element
	fringe = PCB.PySpParVec(G.nvert());
	parents[start] = start;
	fringe[start] = start+1;
	while fringe.getnnz() > 0:
		#FIX:  following line needed?
		fringe = PCB.PySpParVec.range(fringe);
		G.spmat.SpMV_SelMax_inplace(fringe.pySPV);	
		pcb.EWiseMult_inplacefirst(fringe.pySPV, parents.pyDPV, True, -1);
		parents[fringe] = 0
		parents += fringe;
	return parents;


	# returns tuples with elements
	# 0:  True/False of whether it is a BFS tree or not
	# 1:  levels of each vertex in the tree (root is 0, -1 if not reached)
def isBfsTree(G, root, parents):

	ret = 1;	# assume valid
	nvertG = G.nvert();

	# calculate level in the tree for each vertex; root is at level 0
	# about the same calculation as bfsTree, but tracks levels too
	parents2 = PCB.PyDenseParVec(nvertG, -1);
	fringe = PCB.PySpParVec(nvertG);
	parents2[root] = root;
	fringe[root] = root;
	levels = PCB.PyDenseParVec(nvertG, -1);
	levels[root] = 0;

	level = 1;
	#old while fringe.getnnz() > 0:
	while fringe.getnee() > 0:
		fringe = fringe.range();	#note: sparse range()
		#FIX:  create PCB graph-level op
		G.spmat.SpMV_SelMax_inplace(fringe.pySPV);
		#FIX:  create PCB graph-level op
		pcb.EWiseMult_inplacefirst(fringe.pySPV, parents2.pyDPV, True, -1);
		parents2[fringe] = fringe;
		levels[fringe] = level;
		level += 1;
	
	# spec test #1
	#	Not implemented
	

	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
	tmp2 = parents <> PCB.PyDenseParVec(nvertG,0).range(nvertG);
	treeEdges = (parents <> -1) & tmp2;  
	treeI = parents[treeEdges.find()]
	treeJ = PCB.PyDenseParVec(0,0).range(nvertG,0)[treeEdges.find()];
	if (levels[treeI]-levels[treeJ] <> -1).any():
		ret = -1;

	return (ret, levels)

#ToDo:  move bc() here from KDT.py
