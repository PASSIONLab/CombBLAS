import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import Graph as gr

class DiGraph(gr.Graph):

	#print "in DiGraph"

	#FIX:  just building a Graph500 graph by default for now
	def __init__(self,*args):
		if len(args) == 0:
			self.spm = pcb.pySpParMat();
		elif len(args) == 4:
			[i,j,v,nv] = args;
			if type(v) == int:
				v = ParVec.broadcast(len(i),v);
			self.spm = pcb.pySpParMat(nv,nv,i.dpv,j.dpv,v.dpv);
		elif len(args) == 5:
			[i,j,v,nv1,nv2] = args;
			if type(v) == int:
				v = ParVec.broadcast(len(i),v);
			self.spm = pcb.pySpParMat(nv1,nv2,i.dpv,j.dpv,v.dpv);
		else:
			raise NotImplementedError, "only 0, 4, and 5 argument cases supported"

	def __getitem__(self, key):
		if type(key)==tuple:
			if len(key)==1:
				[key1] = key; key2 = -1;
			elif len(key)==2:
				[key1, key2] = key;
			else:
				raise KeyError, 'Too many indices'
		else:
			key1 = key;  key2 = key;
		if type(key1)==slice and key1==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key1mn = 0; key1mx = self.nvert()-1;
		else:
			key1mn = key1.min(); key1mx = key1.max()
			if not (key1==ParVec.range(key1mn,key1mx+1)).all():
				raise KeyError, 'Vector first index not a range'
		if type(key2)==slice and key2==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key2mn = 0; key2mx = self.nvert()-1;
		else:
			if not (key2==ParVec.range(key2mn,key2mx+1)).all():
				raise KeyError, 'Vector second index not a range'
			key2mn = key2.min(); key2mx = key2.max()
		[i, j, v] = self.toParVec();
		sel = ((i >= key1mn) & (i <= key1mx) & (j >= key2mn) & (j <= key2mx)).findInds();
		newi = i[sel] - key1mn;
		newj = j[sel] - key2mn;
		newv = v[sel];
		ret = DiGraph(newi, newj, newv, key1mx-key1mn+1, key2mx-key2mn+1);
		
		#ToDo:  check for isBool, or does lower level handle it?
		return ret;

	def copy(self):
		ret = DiGraph();
		ret.spm = self.spm.copy();
		return ret;
		
	def degree(self, dir=gr.Graph.InOut()):
		if dir == gr.Graph.InOut():
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.set(1));
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.Graph.In():
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(ret);
		elif dir == gr.Graph.Out():
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	def genGraph500Edges(self, scale, degrees):
		elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spm, scale, degrees.dpv);
	 	return elapsedTime;

	@staticmethod
	def load(fname):
		ret = DiGraph();
		ret.spm = pcb.pySpParMat();
		ret.spm.load(fname);
		return ret;

	def max(self, dir=gr.Graph.InOut()):
		#ToDo:  is default to InOut best?
		if dir == gr.Graph.InOut():
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.max());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.max());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.Graph.In():
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.max());
			return ParVec.toParVec(ret);
		elif dir == gr.Graph.Out():
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.max());
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	def min(self, dir=gr.Graph.InOut()):
		#ToDo:  is default to InOut best?
		if dir == gr.Graph.InOut():
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.min());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.min());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.Graph.In():
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.min());
			return ParVec.toParVec(ret);
		elif dir == gr.Graph.Out():
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.min());
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	#FIX:  good idea to have this return an int or a tuple?
	def nvert(self):
		nrow = self.spm.getnrow();
		ncol = self.spm.getncol();
		if nrow==ncol:
			ret = nrow;
		else:
			ret = (nrow, ncol)
		return ret;

	#in-place, so no return value
	def ones(self):
		self.spm.Apply(pcb.set(1));
		return;

	#in-place, so no return value
	def reverseEdges(self):
		self.spm.Transpose();

	def toParVec(self):
		nv = self.nvert()
		reti = ParVec(nv);
		retj = ParVec(nv);
		retv = ParVec(nv);
		self.spm.Find(reti.dpv, retj.dpv, retv.dpv);
		return (reti, retj, retv);

	# ==================================================================
	#  "complex ops" implemented below here
	# ==================================================================


	# returns a Boolean vector of which vertices are neighbors
	def neighbors(self, source, nhop=1):
		dest = pcb.pyDenseParVec(self.nvert(),0)
		fringe = pcb.pySpParVec(self.nvert());
		dest[fringe] = 1;
		fringe[source.dpv] = 1;
		for i in range(nhop):
			fringe.setNumToInd();
			self.spm.SpMV_SelMax_inplace(fringe);
			dest[fringe] = 1;
		return ParVec.toParVec(dest);
		
	# returns:
	#   - source:  a vector of the source vertex for each new vertex
	#   - dest:  a Boolean vector of the new vertices
	#ToDo:  nhop argument?
	def pathsHop(self, source):
		retDest = pcb.pyDenseParVec(self.nvert(),0)
		retSource = pcb.pyDenseParVec(self.nvert(),0)
		fringe = pcb.pySpParVec(self.nvert());
		retDest[fringe] = 1;
		fringe[source.dpv] = 1;
		fringe.setNumToInd();
		self.spm.SpMV_SelMax_inplace(fringe);
		retDest[fringe] = 1;
		retSource[fringe] = fringe;
		return ParVec.toParVec(retSource), ParVec.toParVec(retDest);
	
#def DiGraphGraph500():
#	self = DiGraph();
#	self.spm = GenGraph500Edges(sc.log2(nvert));

class ParVec(gr.ParVec):
	pass;

class SpParVec(gr.SpParVec):
	pass;

		

master = gr.master;
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
	parents = pcb.pyDenseParVec(G.nvert(), -1);
	# NOTE:  values in fringe go from 1:n instead of 0:(n-1) so can
	# distinguish vertex0 from empty element
	fringe = pcb.pySpParVec(G.nvert());
	parents[start] = start;
	fringe[start] = start+1;
	while fringe.getnnz() > 0:
		#FIX:  setNumToInd -> SPV.range()
		fringe.setNumToInd();
		G.spm.SpMV_SelMax_inplace(fringe);	
		pcb.EWiseMult_inplacefirst(fringe, parents, True, -1);
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
	parents2 = ParVec.zeros(nvertG) - 1;
	fringe = pcb.pySpParVec(nvertG);
	parents2[root] = root;
	fringe[root] = root+1;	#fix
	levels = ParVec.zeros(nvertG) - 1;
	levels[root] = 0;

	level = 1;
	#FIX getnnz() -> SPV.getnnn()
	while fringe.getnnz() > 0:
		fringe.setNumToInd();		#ToDo: sparse range()
		#FIX:  create PCB graph-level op
		G.spm.SpMV_SelMax_inplace(fringe);
		#FIX:  create PCB graph-level op
		pcb.EWiseMult_inplacefirst(fringe, parents2.dpv, True, -1);
		parents2.dpv[fringe] = fringe;
		levels.dpv[fringe] = level;
		level += 1;
	
	# spec test #1
	#	Not implemented
	

	# spec test #2
	#    tree edges should be between verts whose levels differ by 1
	
	tmp2 = parents != ParVec.range(nvertG);
	treeEdges = (parents != -1) & tmp2;  
	treeI = parents[treeEdges.findInds()]
	treeJ = ParVec.range(nvertG)[treeEdges.findInds()];
	if (levels[treeI]-levels[treeJ] != -1).any():
		ret = -2;

	return (ret, ParVec.toParVec(levels))

# returns a Boolean vector of which vertices are neighbors
def neighbors(G, source, nhop=1):
	dest = pcb.pyDenseParVec(G.nvert(),0)
	fringe = pcb.pySpParVec(G.nvert());
	dest[fringe] = 1;
	fringe[source.dpv] = 1;
	for i in range(nhop):
		fringe.setNumToInd();
		G.spm.SpMV_SelMax_inplace(fringe);
		dest[fringe] = 1;
	return ParVec.toParVec(dest);
	
# returns:
#   - source:  a vector of the source vertex for each new vertex
#   - dest:  a Boolean vector of the new vertices
#ToDo:  nhop argument?
def pathHop(G, source):
	retDest = pcb.pyDenseParVec(G.nvert(),0)
	retSource = pcb.pyDenseParVec(G.nvert(),0)
	fringe = pcb.pySpParVec(G.nvert());
	retDest[fringe] = 1;
	fringe[source.dpv] = 1;
	fringe.setNumToInd();
	G.spm.SpMV_SelMax_inplace(fringe);
	retDest[fringe] = 1;
	retSource[fringe] = fringe;
	return ParVec.toParVec(retSource), ParVec.toParVec(retDest);
	
def centrality(alg, G, **kwargs):
#		ToDo:  Normalize option?
	if alg=='exactBC':
		cent = _approxBC(G, sample=1.0, **kwargs)

	elif alg=='_approxBC':
		cent = _approxBC(G, **kwargs);

	elif alg=='kBC':
		raise NotImplementedError, "k-betweenness centrality unimplemented"

	elif alg=='degree':
		raise NotImplementedError, "degree centrality unimplemented"
		
	else:
		raise KeyError, "unknown centrality algorithm (%s)" % alg

	return cent;


def _approxBC(G, sample=0.05, chunk=-1):
	print "chunk=%d, sample=%5f" % (chunk, sample);
	# calculate chunk automatically if not specified

def _bc( G, K4approx, batchSize ):


    # transliteration of Lincoln Labs 2009Feb09 M-language version, 
    # 

    A = G.ones();			
    Aint = kdtsp.spones(G);	# not needed;  Gs only int for now
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
        #nspInv = sp.csr_matrix((1/(nspv.astype(sc.float64)),(nspi,nspj)), shape=(curSize, N));

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

def cluster(alg, G, **kwargs):
#		ToDo:  Normalize option?
	if alg=='Markov' or alg=='markov':
		clus = _markov(G, **kwargs)

	elif alg=='kNN' or alg=='knn':
		raise NotImplementedError, "k-nearest neighbors clustering not implemented"

	else:
		raise KeyError, "unknown clustering algorithm (%s)" % alg

	return clus;

