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
		self.spm = PCB.PySpParMat();

	def degree(self):
		return self.indegree() + self.outdegree();

	def indegree(self):
		#tmp = spm.reduce(self._spones(self.spm), 0, +);
		print "\n	DiGraph.indegree() not fully implemented!\n"
		tmp = 1;
		return tmp;

	def load(self, fname):
		self.spm.load(fname);

	def outdegree(self):
		#tmp = spm.reduce(self._spones(self.spm), 1, +);
		print "\n	DiGraph.outdegree() not fully implemented!\n"
		tmp = 1;
		return tmp;

#def DiGraphGraph500():
#	self = DiGraph();
#	self.spm = GenGraph500Edges(sc.log2(nvert));

		
class DiEdgeV(gr.EdgeV):
	print "in DiEdgeV"

		
class ParVec:
	print "in ParVec"

	def __init__(self, length):
		if length>0:
			self.dpv = PCB.PyDenseParVec(length,0);

	def __abs__(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.abs()
		return ret;

	def __add__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv + other;
		else:	#elif  instance(other,ParVec):
			ret.dpv = self.dpv + other.dpv;
		return ret;

	def __and__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv & other;
		else: 	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv & other.dpv;
		return ret;

	def __copy__(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.copy()
		return ret;

	def __getitem__(self, key):
		if type(key) == int:
			if key > self.dpv.len()-1:
				raise IndexError;
			ret = self.dpv[key];
		else:	#elif isinstance(other,ParVec):
			ret = self.dpv[key.dpv];
		return ret;

	def __ge__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv >= other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv >= other.dpv;
		return ret;

	def __gt__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv > other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv > other.dpv;
		return ret;

	def __iadd__(self, other):
		if type(other) == int:
			self.dpv += other;
		else:	#elif isinstance(other,ParVec):
			self.dpv += other.dpv;
		return self;

	def __isub__(self, other):
		if type(other) == int:
			self.dpv -= other;
		else:	#elif isinstance(other,ParVec):
			self.dpv -= other.dpv;
		return self;

	def __len__(self):
		return self.dpv.len();

	def __mod__(self, other):
		raise AttributeError, "ParVec:__mod__ not implemented yet"
		ret = self.copy()
		while (ret >= other).any():
			tmp = (ret.dpv >= other.dpv).findInds();
			tmp2 = (ret.dpv[tmp] - other.dpv[tmp]).sparse();
			tmp2[0] = 0
			ret.dpv[tmp.sparse()] = tmp2;
		return ret;

	def __mul__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv * other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = (self.dpv.sparse() * other.dpv).dense();
		return ret;

	def __ne__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv <> other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv <> other.dpv;
		return ret;

	def __repr__(self):
		return self.dpv.printall();

	def __setitem__(self, key, value):
		if type(key) == int:
			self.dpv[key] = value;
		else:
			if type(value) == int:
				self.dpv[key.dpv] = value;
			else:
				self.dpv[key.dpv] = value.dpv;

	def __sub__(self, other):
		ret = ParVec(-1);
		ret.dpv = self.dpv - other.dpv;
		if type(other) == int:
			ret.dpv = self.dpv - other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv - other.dpv;
		return ret;

	def any(self):
		ret = ParVec(-1);
		ret = self.dpv.any();
		return ret;

	def copy(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.copy();
		return ret;

	def find(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.find();
		return ret;

	def printall(self):
		return self.dpv.printall();

	def randPerm(self):
		self.dpv.randPerm()
		return self;

	def sum(self):
		return self.dpv.sum();

class VertexV(ParVec):
	print "in VertexV"

def toVertexV(DPV):
	ret = VertexV(-1);
	ret.dpv = DPV;
	return ret;

def ones(sz):
	ret = VertexV(-1);
	ret.dpv = PCB.ones(sz);
	return ret;

def range(stop):
	ret = VertexV(-1);
	ret.dpv = PCB.PyDenseParVec(0,0).range(stop);
	return ret;

def zeros(sz):
	ret = VertexV(-1);
	ret.dpv = PCB.zeros(sz);
	return ret;

class DiEdgeV(ParVec):
	print "in DiEdgeV"


class SpVertexV:
	print "in SpVertexV"

	def __init__(self, length):
		self.spv = pcb.pySpParVec(length);

def genGraph500Candidates(G, howmany):
	pyDPV = G.spm.pySPM.GenGraph500Candidates(howmany);
	ret = toVertexV(PCB.toPyDenseParVec(pyDPV));
	return ret

sendFeedback = gr.sendFeedback;


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
	elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spm.pySPM, scale);
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
		G.spm.pySPM.SpMV_SelMax_inplace(fringe.pySPV);	
		pcb.EWiseMult_inplacefirst(fringe.pySPV, parents.pyDPV, True, -1);
		parents[fringe] = 0
		parents += fringe;
	return toVertexV(parents);


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
	fringe[root] = root+1;
	levels = PCB.PyDenseParVec(nvertG, -1);
	levels[root] = 0;

	level = 1;
	#old while fringe.getnnz() > 0:
	while fringe.getnee() > 0:
		fringe = fringe.range();	#note: sparse range()
		#FIX:  create PCB graph-level op
		G.spm.pySPM.SpMV_SelMax_inplace(fringe.pySPV);
		#FIX:  create PCB graph-level op
		pcb.EWiseMult_inplacefirst(fringe.pySPV, parents2.pyDPV, True, -1);
		parents2[fringe] = fringe;
		levels[fringe] = level;
		level += 1;
	
	# spec test #1
	#	Not implemented
	

	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
	tmp2 = parents <> range(nvertG);
	treeEdges = (parents <> -1) & tmp2;  
	treeI = parents[treeEdges.find()]
	treeJ = range(nvertG)[treeEdges.find()];
	if (levels[treeI]-levels[treeJ] <> -1).any():
		ret = -2;

	return (ret, toVertexV(levels))

#ToDo:  move bc() here from KDT.py
