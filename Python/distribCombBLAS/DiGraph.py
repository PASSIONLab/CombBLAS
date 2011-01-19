import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import PyCombBLAS as PCB
import Graph as gr

class DiGraph(gr.Graph):

	#print "in DiGraph"

	#FIX:  just building a Graph500 graph by default for now
	def __init__(self,*args):
		if len(args) == 0:
			self.spm = PCB.PySpParMat();
		elif len(args) == 4:
			#create a DiGraph from i/j/v ParVecs and nv nverts
			[i,j,v,nv] = args;
			pass;

	def __getitem__(self, key):
		if type(key)==tuple:
			if len(key)==2:
				[key1, key2] = key;
			else:
				raise KeyError, 'Too many indices'
		else:
			key1 = key;  key2 = key;
		#ToDo:  check for isBool, or does lower level handle it?
		ret = self.copy();	#FIX: do actual indexing here
		return ret;

	def copy(self):
		ret = DiGraph();
		ret.spm = self.spm.copy();
		return ret;
		
	def degree(self):
		return self.indegree() + self.outdegree();

	def genGraph500Edges(self, scale):
		elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spm.pySPM, scale);
	 	return elapsedTime;

	def genGraph500Candidates(self, howmany):
		pyDPV = self.spm.pySPM.GenGraph500Candidates(howmany);
		ret = ParVec.toParVec(PCB.toPyDenseParVec(pyDPV));
		return ret

	def indegree(self):
		ret = self.spm.pySPM.Reduce(pcb.pySpParMat.Row(),pcb.plus());
		return ParVec.toParVec(PCB.toPyDenseParVec(ret));

	def load(self, fname):
		self.spm.load(fname);

	def outdegree(self):
		ret = self.spm.pySPM.Reduce(pcb.pySpParMat.Column(),pcb.plus());
		return ParVec.toParVec(PCB.toPyDenseParVec(ret));

#def DiGraphGraph500():
#	self = DiGraph();
#	self.spm = GenGraph500Edges(sc.log2(nvert));

		
		
class ParVec:
	#print "in ParVec"

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

	def __div__(self, other):
		selfcopy = self.copy();
		ret = ParVec(len(self));
		while (selfcopy >= other).any():
			tmp = selfcopy >= other;
			selfcopy[tmp] = selfcopy - other;
			ret[tmp] = ret+1;
		return ret;

	def __getitem__(self, key):
		if type(key) == int:
			if key > self.dpv.len()-1:
				raise IndexError;
			ret = self.dpv[key];
		else:	#elif isinstance(other,ParVec):
			#tmp1 = len((key<0).findInds())==0;
			#tmp2 = len((key>1).findInds())==0;
			#keybool = tmp1 & tmp2;
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

	def __le__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv <= other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv <= other.dpv;
		return ret;

	def __len__(self):
		return self.dpv.len();

	def __lt__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = self.dpv < other;
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv < other.dpv;
		return ret;

	def __mod__(self, other):
		ret = self.copy();
		while (ret >= other).any():
			tmp = ret >= other;
			ret[tmp] = ret - other;
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
		ret.dpv = self.dpv.copy()
		return ret;

#	NOTE:  no ParVec.find() yet because no SpParVec yet
#	def find(self):
#		ret = ParVec(-1);
#		ret.dpv = self.dpv.find();
#		return ret;
#
	def findInds(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.findInds();
		return ret;

	def isBool(self):
		tmp1 = len((self<0).findInds())==0;
		tmp2 = len((self>1).findInds())==0;
		return tmp1 & tmp2;

	def logical_not(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.logical_not();
		return ret;

	@staticmethod
	def ones(sz):
		ret = ParVec(-1);
		ret.dpv = PCB.ones(sz);
		return ret;
	
	def printall(self):
		return self.dpv.printall();

	def randPerm(self):
		self.dpv.randPerm()
		return self;

	@staticmethod
	def range(arg1, *args):
		if len(args) == 0:
			start = 0;
			stop = arg1;
		elif len(args) == 1:	
			start = arg1;
			stop = args[0];
		else:
			raise NotImplementedError, "No 3-argument range()"
		ret = ParVec(0);
		ret.dpv = PCB.PyDenseParVec.range(start,stop);
		return ret;
	
	def sum(self):
		return self.dpv.sum();

	#TODO:  check for class being PyDenseParVec?
	@staticmethod
	def toParVec(DPV):
		ret = ParVec(-1);
		ret.dpv = DPV;
		return ret;
	
	@staticmethod
	def zeros(sz):
		ret = ParVec(-1);
		ret.dpv = PCB.zeros(sz);
		return ret;
	

#class SpParVec:
#	#print "in SpVertexV"
#
#	def __init__(self, length):
#		self.spv = pcb.pySpParVec(length);

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
	return ParVec.toParVec(parents);


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
	
	tmp2 = parents <> ParVec.range(nvertG);
	treeEdges = (parents <> -1) & tmp2;  
	treeI = parents[treeEdges.findInds()]
	treeJ = ParVec.range(nvertG)[treeEdges.findInds()];
	if (levels[treeI]-levels[treeJ] <> -1).any():
		ret = -2;

	return (ret, ParVec.toParVec(levels))

def centrality(alg, G, **kwargs):
#		ToDo:  Normalize option?
	if alg=='exactBC':
		cent = approxBC(G, sample=1.0, **kwargs)

	elif alg=='approxBC':
		cent = approxBC(G, **kwargs);

	elif alg=='kBC':
		raise NotImplementedError, "k-betweenness centrality unimplemented"

	elif alg=='degree':
		raise NotImplementedError, "degree centrality unimplemented"
		
	else:
		raise KeyError, "unknown centrality algorithm (%s)" % alg

	return cent;


def approxBC(G, sample=0.05, chunk=-1):
	print "chunk=%d, sample=%5f" % (chunk, sample);
	# calculate chunk automatically if not specified

def cluster(alg, G, **kwargs):
#		ToDo:  Normalize option?
	if alg=='Markov' or alg=='markov':
		clus = markov(G, **kwargs)

	elif alg=='kNN' or alg=='knn':
		raise NotImplementedError, "k-nearest neighbors clustering not implemented"

	else:
		raise KeyError, "unknown clustering algorithm (%s)" % alg

	return clus;

def bc( G, K4approx, batchSize ):


    # transliteration of Lincoln Labs 2009Feb09 M-language version, 
    # by Steve Reinhardt 2010Sep16
    # 

    A = kdtsp.spbool(G);			# not needed; G already bool
    Aint = kdtsp.spones(G);			# needed?  non-bool not spted
    N = A.nvert()

    bc = ParVec(N);

    # ToDo:  original triggers off whether data created via RMAT to set nPasses
    #        and K4approx
    if (2**K4approx > N):
        K4approx = sc.floor(sc.log2(N))
        nPasses = 2**K4approx;
    else:
        nPasses = N;		

    numBatches = sc.ceil(nPasses/batchSize).astype(int)

    for p in range(numBatches):
        bfs = []		

        batch = ParVec.range(p*batchSize,min((p+1)*batchSize,N));
        curSize = len(batch);

        # original M version uses accumarray in following line
	# nsp == number of shortest paths
        #nsp = sp.csr_matrix((sc.tile(1,(1,curSize)).flatten(), (sc.arange(curSize),sc.array(batch))),shape=(curSize,N));
	nsp = DiGraph(ParVec.range(curSize), batch, 1, N);


        depth = 0;
        #OLD fringe = Aint[batch,:];   
	fringe = A[batch,ParVec.range(N)];

        while fringe.getnnz() > 0:
            depth = depth+1;
            #print (depth, fringe.getnnz()) 
            # add in shortest path counts from the fringe
            nsp = nsp+fringe
            bfs = sc.append(bfs,kdtsp.spbool(fringe));
            tmp = fringe * A;		#FIX:  can't be in-line in next line due to SciPy bug
					#avoid creating not(nsp) if possible
            fringe = tmp.multiply(kdtsp.spnot(nsp));

        [nspi, nspj, nspv] = kdtsp.find(nsp);
        nspInv = sp.csr_matrix((1/(nspv.astype(sc.float64)),(nspi,nspj)), shape=(curSize, N));

        bcu = sp.csr_matrix(sc.ones((curSize, N)));		#FIX:  too big in real cases?

        # compute the bc update for all vertices except the sources
        for depth in range(depth-1,0,-1):
            # compute the weights to be applied based on the child values
            w = bfs[depth].multiply(nspInv).multiply(bcu);
            # Apply the child value weights and sum them up over the parents
            # then apply the weights based on parent values
            bcu = bcu + (A*w.T).T.multiply(bfs[depth-1]).multiply(nsp);

        # upcate the bc with the bc update
        bc = bc + bcu.sum(0)

    # subtract off the additional values added in by precomputation
    bc = bc - nPasses;
    return bc;

