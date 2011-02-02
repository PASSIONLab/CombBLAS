import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import Graph as gr

class DiGraph(gr.Graph):

	#print "in DiGraph"

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
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

	def __add__(self, other):
		#FIX:  ****RESULTS INVALID****
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		return self;

		raise NotImplementedError
		if type(other) == int:
			raise NotImplementedError
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other, DiGraph):
			ret = self.copy();
			#ret.spm.xxxxxxx
		return ret;

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
		if type(key1) == int:
			tmp = ParVec(1);
			tmp[0] = key1;
			key1 = tmp;
		if type(key2) == int:
			tmp = ParVec(1);
			tmp[0] = key2;
			key2 = tmp;
		if type(key1)==slice and key1==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key1mn = 0; key1mx = self.nvert()-1;
		else:
			key1mn = key1.min(); key1mx = key1.max()
			if len(key1)!=(key1mx-key1mn+1) or not (key1==ParVec.range(key1mn,key1mx+1)).all():
				raise KeyError, 'Vector first index not a range'
		if type(key2)==slice and key2==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key2mn = 0; key2mx = self.nvert()-1;
		else:
			key2mn = key2.min(); key2mx = key2.max()
			if len(key2)!=(key2mx-key2mn+1) or not (key2==ParVec.range(key2mn,key2mx+1)).all():
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

	def __mul__(self, other):
		#FIX:  ****RESULTS INVALID****
		if self.nvert()[1] != other.nvert()[0]:
			raise IndexError, 'First graph #out-verts must equal second graph #in-verts'
		ones = ParVec.range(min(self.nvert()[0], other.nvert()[1]));
		ret = DiGraph(ones, ones, ones, self.nvert()[0], other.nvert()[1]);
		return ret;

	def boolWeight(self):
		#ToDo:  change for real Boolean matrices
		return DiGraph.onesWeight(self);

	def copy(self):
		ret = DiGraph();
		ret.spm = self.spm.copy();
		return ret;
		
	def degree(self, dir=gr.Graph.Out()):
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

	def divWeight(self, other):
		if type(other) != int:
			raise NotImplementedError
		else:
			self.spm.Apply(pcb.bind2nd(pcb.divides(),other));
		return;

	@staticmethod
	def fullyConnected(n,m):
		#ToDo:  if only 1 input, assume square
		i = ParVec.range(n*m) % n;
		j = ParVec.range(n*m) / n;
		v = ParVec.range(n*m);
		ret = DiGraph(i,j,v,n,m);
		return ret;

	def genGraph500Edges(self, scale, degrees):
		elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spm, scale, degrees.dpv);
	 	return elapsedTime;

	@staticmethod
	def load(fname):
		ret = DiGraph();
		ret.spm = pcb.pySpParMat();
		ret.spm.load(fname);
		return ret;

	def maxWeight(self, dir=gr.Graph.InOut()):
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

	def minWeight(self, dir=gr.Graph.InOut()):
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

	def mulWeight(self, other):
		if type(other) != int:
			raise NotImplementedError
		else:
			self.spm.Apply(pcb.bind2nd(pcb.multiplies(),other));
		return;

	def notMulWeight(self, other):
		#FIX:  ****RESULTS INVALID****
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		return self;

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
	def onesWeight(self):
		self.spm.Apply(pcb.set(1));
		return;

	#in-place, so no return value
	def reverseEdges(self):
		self.spm.Transpose();

	def subgraph(self, *args):
		if len(args) == 1:
			[ndx1] = args;
			ret = self[ndx1, ndx1];
		elif len(args) == 2:
			[ndx1, ndx2] = args;
			ret = self[ndx1, ndx2];
		else:
			raise IndexError, 'Too many indices'
		return ret;

	def sumWeight(self, dir=gr.Graph.Out()):
		if dir == gr.Graph.InOut():
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.Graph.In():
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity());
			return ParVec.toParVec(ret);
		elif dir == gr.Graph.Out():
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity());
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	T = reverseEdges;

	def toParVec(self):
		ne = self.nedge()
		reti = ParVec(ne);
		retj = ParVec(ne);
		retv = ParVec(ne);
		self.spm.Find(reti.dpv, retj.dpv, retv.dpv);
		#ToDo:  return nvert() of original graph, too
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
	


	@staticmethod
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
	#	each vertex in the tree; unreached vertices have parent == -1.
	#
	def bfsTree(self, start):
		parents = pcb.pyDenseParVec(self.nvert(), -1);
		# NOTE:  values in fringe go from 1:n instead of 0:(n-1) so can
		# distinguish vertex0 from empty element
		fringe = pcb.pySpParVec(self.nvert());
		parents[start] = start;
		fringe[start] = start;
		while fringe.getnnz() > 0:
			#FIX:  setNumToInd -> SPV.range()
			fringe.setNumToInd();
			self.spm.SpMV_SelMax_inplace(fringe);	
			pcb.EWiseMult_inplacefirst(fringe, parents, True, -1);
			parents[fringe] = 0
			parents += fringe;
		return ParVec.toParVec(parents);
	
	
		# returns tuples with elements
		# 0:  True/False of whether it is a BFS tree or not
		# 1:  levels of each vertex in the tree (root is 0, -1 if not reached)
	def isBfsTree(self, root, parents):
	
		ret = 1;	# assume valid
		nvertG = self.nvert();
	
		# calculate level in the tree for each vertex; root is at level 0
		# about the same calculation as bfsTree, but tracks levels too
		parents2 = ParVec.zeros(nvertG) - 1;
		parents2[root] = root;
		fringe = SpParVec(nvertG);
		fringe[root] = root;	#fix
		levels = ParVec.zeros(nvertG) - 1;
		levels[root] = 0;
	
		level = 1;
		while fringe.nnn() > 0:
			fringe.sprange();
			#FIX:  create PCB graph-level op
			self.spm.SpMV_SelMax_inplace(fringe.spv);
			#FIX:  create PCB graph-level op
			pcb.EWiseMult_inplacefirst(fringe.spv, parents2.dpv, True, -1);
			parents2[fringe] = fringe;
			levels[fringe] = level;
			level += 1;
		
		# spec test #1
		# Confirm that the tree is a tree;  i.e., that it does not
		# have any cycles (visited more than once while building
		# the tree) and that every vertex with a parent is
		# in the tree. 

		# build a new graph from just tree edges
		tmp2 = parents != ParVec.range(nvertG);
		treeEdges = (parents != -1) & tmp2;  
		treeI = parents[treeEdges.findInds()]
		treeJ = ParVec.range(nvertG)[treeEdges.findInds()];
		# root cannot be destination of any tree edge
		if (treeJ == root).any():
			ret = -1;
			return ret;
		builtGT = DiGraph(treeI, treeJ, 1, nvertG);
		builtGT.reverseEdges();
		visited = ParVec.zeros(nvertG);
		visited[root] = 1;
		fringe = SpParVec(nvertG);
		fringe[root] = root;
		cycle = False;
		multiparents = False;
		while fringe.nnn() > 0 and not cycle and not multiparents:
			fringe.spones();
			newfringe = SpParVec.toSpParVec(builtGT.spm.SpMV_PlusTimes(fringe.spv));
			if visited[newfringe.denseNonnulls().findInds()].any():
				cycle = True;
				break;
			if (newfringe > 1).any():
				multiparents = True;
			fringe = newfringe;
			visited[fringe] = 1;
		if cycle or multiparents:
			ret = -1;	
			return ret;
		
		# spec test #2
		#    tree edges should be between verts whose levels differ by 1
		
		if (levels[treeI]-levels[treeJ] != -1).any():
			ret = -2;
			return ret;
	
		return (ret, levels)
	
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
	def pathHop(self, source):
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
		
	def centrality(self, alg, **kwargs):
	#		ToDo:  Normalize option?
		if alg=='exactBC':
			cent = _approxBC(self, sample=1.0, **kwargs)
	
		elif alg=='_approxBC':
			cent = _approxBC(self, **kwargs);
	
		elif alg=='kBC':
			raise NotImplementedError, "k-betweenness centrality unimplemented"
	
		elif alg=='degree':
			raise NotImplementedError, "degree centrality unimplemented"
			
		else:
			raise KeyError, "unknown centrality algorithm (%s)" % alg
	
		return cent;
	
	
	def _approxBC(self, sample=0.05, chunk=-1):
		print "chunk=%d, sample=%5f" % (chunk, sample);
		# calculate chunk automatically if not specified
	
	def _bc(self, K4approx, batchSize ):
	
	
	    # transliteration of Lincoln Labs 2009Feb09 M-language version, 
	    # 
	
	    A = self.onesWeight();			
	    Aint = self.onesWeight();	# not needed;  Gs only int for now
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
	            fringe = tmp.notMulWeight(nsp);
	
	        #old [nspi, nspj, nspv] = kdtsp.find(nsp);
	        #nspInv = sp.csr_matrix((1/(nspv.astype(sc.float64)),(nspi,nspj)), shape=(curSize, N));
	
	        #bcu = sp.csr_matrix(sc.ones((curSize, N)));		#FIX:  too big in real cases?
		bcu = DiGraph.fullyConnected(curSize,N);
	
	        # compute the bc update for all vertices except the sources
	        for depth in range(depth-1,0,-1):
	            # compute the weights to be applied based on the child values
	            #old w = bfs[depth].multiply(nspInv).multiply(bcu);
	            w = bfs[depth].divWeight(nsp).mulWeight(bcu);
	            # Apply the child value weights and sum them up over the parents
	            # then apply the weights based on parent values
	            bcu = bcu + (A*w.T).T.mulWeight(bfs[depth-1]).mulWeight(nsp);
	
	        # update the bc with the bc update
	        bc = bc + bcu.sum(0)	#FIX:  dir?
	
	    # subtract off the additional values added in by precomputation
	    bc = bc - nPasses;
	    return bc;
	
	def cluster(self, alg, **kwargs):
	#		ToDo:  Normalize option?
		if alg=='Markov' or alg=='markov':
			clus = _markov(self, **kwargs)
	
		elif alg=='kNN' or alg=='knn':
			raise NotImplementedError, "k-nearest neighbors clustering not implemented"
	
		else:
			raise KeyError, "unknown clustering algorithm (%s)" % alg
	
		return clus;

class ParVec(gr.ParVec):
	pass;

class SpParVec(gr.SpParVec):
	pass;

		

master = gr.master;
sendFeedback = gr.sendFeedback;

print "\n\n	***NOTE: DiGraph*DiGraph, DiGraph+DiGraph and \n\t DiGraph.notMulWeight() are dummy functions\n\n";
