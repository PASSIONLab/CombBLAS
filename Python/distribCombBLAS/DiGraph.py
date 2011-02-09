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
		if len(args) == 1:
			[arg] = args;
			if arg < 0:
				self.spm = pcb.pySpParMat();
			else:
				raise NotImplementedError, '1-argument case only accepts negative value'
		elif len(args) == 4:
			[i,j,v,nv] = args;
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v);
			self.spm = pcb.pySpParMat(nv,nv,i.dpv,j.dpv,v.dpv);
		elif len(args) == 5:
			[i,j,v,nv1,nv2] = args;
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v);
			if i.max() > nv1-1:
				raise KeyError, 'at least one first index greater than #vertices'
			if j.max() > nv2-1:
				raise KeyError, 'at least one second index greater than #vertices'
			self.spm = pcb.pySpParMat(nv1,nv2,i.dpv,j.dpv,v.dpv);
		else:
			raise NotImplementedError, "only 1, 4, and 5 argument cases supported"

	def __add__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other, DiGraph):
			ret = self.copy();
			ret.spm += other.spm;
			#ret.spm = pcb.EWiseApply(self.spm, other.spm, pcb.plus());  # only adds if both mats have nonnull elems!!
		return ret;

	def __div__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy();
			ret.spm.Apply(pcb.bind2nd(pcb.divides(),other));
		elif self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other,DiGraph):
			ret = self.copy();
			ret.spm = pcb.EWiseApply(self.spm, other.spm, pcb.divides());
		else:
			raise NotImplementedError
		return ret;

	def __getitem__(self, key):
		#ToDo:  accept slices for key1/key2 besides ParVecs
		if type(key)==tuple:
			if len(key)==1:
				[key1] = key; key2 = -1;
			elif len(key)==2:
				[key1, key2] = key;
			else:
				raise KeyError, 'Too many indices'
		else:
			key1 = key;  key2 = key;
		if type(key1) == int or type(key1) == long or type(key1) == float:
			tmp = ParVec(1);
			tmp[0] = key1;
			key1 = tmp;
		if type(key2) == int or type(key1) == long or type(key1) == float:
			tmp = ParVec(1);
			tmp[0] = key2;
			key2 = tmp;
		if type(key1)==slice and key1==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key1mn = 0; key1mx = self.nvert()-1;
		else:
			key1mn = int(key1.min()); key1mx = int(key1.max());
			if len(key1)!=(key1mx-key1mn+1) or not (key1==ParVec.range(key1mn,key1mx+1)).all():
				raise KeyError, 'Vector first index not a range'
		if type(key2)==slice and key2==slice(None,None,None):
			#ToDo: will need to handle nvert() 2-return case
			key2mn = 0; key2mx = self.nvert()-1;
		else:
			key2mn = int(key2.min()); key2mx = int(key2.max());
			if len(key2)!=(key2mx-key2mn+1) or not (key2==ParVec.range(key2mn,key2mx+1)).all():
				raise KeyError, 'Vector second index not a range'
		[i, j, v] = self.toParVec();
		sel = ((i >= key1mn) & (i <= key1mx) & (j >= key2mn) & (j <= key2mx)).findInds();
		newi = i[sel] - key1mn;
		newj = j[sel] - key2mn;
		newv = v[sel];
		ret = DiGraph(newi, newj, newv, key1mx-key1mn+1, key2mx-key2mn+1);
		return ret;

	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other, DiGraph):
			#dead tmp = pcb.EWiseApply(self.spm, other.spm, pcb.plus());
			self.spm += other.spm;
		return self;

	def __imul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self.spm.Apply(pcb.bind2nd(pcb.multiplies(),other));
		elif isinstance(other,DiGraph):
			self.spm = pcb.EWiseApply(self.spm,other.spm, pcb.multiplies());
		else:
			raise NotImplementedError
		return self;

	def __mul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy();
			ret.spm.Apply(pcb.bind2nd(pcb.multiplies(),other));
		elif self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other,DiGraph):
			ret = self.copy();
			ret.spm = pcb.EWiseApply(self.spm,other.spm, pcb.multiplies());
		else:
			raise NotImplementedError
		return ret;

	_REPR_MAX = 100;
	def __repr__(self):
		if self.nvert()==1:
			[i, j, v] = self.toParVec();
			print "%d %f" % (v[0], v[0]);
		else:
			[i, j, v] = self.toParVec();
			if len(i) < DiGraph._REPR_MAX:
				print i,j,v
		return ' ';

	def _SpMM(self, other):
		selfnv = self.nvert()
		if type(selfnv) == tuple:
			[selfnv1, selfnv2] = selfnv;
		else:
			selfnv1 = selfnv; selfnv2 = selfnv;
		othernv = other.nvert()
		if type(othernv) == tuple:
			[othernv1, othernv2] = othernv;
		else:
			othernv1 = othernv; othernv2 = othernv;
		if selfnv2 != othernv1:
			raise ValueError, '#in-vertices of first graph not equal to #out-vertices of the second graph '
		ret = DiGraph(-1);
		ret.spm = self.spm.SpMM(other.spm);
		return ret;

	def bool(self):
		#ToDo:  change for real Boolean matrices
		return DiGraph.ones(self);

	def copy(self):
		ret = DiGraph(-1);
		ret.spm = self.spm.copy();
		return ret;
		
	def degree(self, dir=gr.Out):
		if dir == gr.InOut:
			#ToDo:  can't do InOut if nonsquare graph
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.set(1));
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.In:
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(ret);
		elif dir == gr.Out:
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.set(1));
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	@staticmethod
	def fullyConnected(n,m=None):
		if m == None:
			m = n;
		i = (ParVec.range(n*m) % n).floor();
		j = (ParVec.range(n*m) / n).floor();
		v = ParVec.ones(n*m);
		ret = DiGraph(i,j,v,n,m);
		return ret;

	def genGraph500Edges(self, scale, degrees):
		elapsedTime = pcb.pySpParMat.GenGraph500Edges(self.spm, scale, degrees.dpv);
	 	return elapsedTime;

	@staticmethod
	def load(fname):
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = DiGraph(-1);
		ret.spm = pcb.pySpParMat();
		ret.spm.load(fname);
		return ret;

	def max(self, dir=gr.InOut):
		#ToDo:  is default to InOut best?
		if dir == gr.InOut:
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.max());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.max());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.In:
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.max());
			return ParVec.toParVec(ret);
		elif dir == gr.Out:
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.max());
			return ParVec.toParVec(ret);
		else:
			raise KeyError, 'Invalid edge direction'

	def min(self, dir=gr.InOut):
		#ToDo:  is default to InOut best?
		if dir == gr.InOut:
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.min());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.min());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.In:
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.min());
			return ParVec.toParVec(ret);
		elif dir == gr.Out:
			ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.min());
			return ParVec.toParVec(ret);

	def mulNot(self, other):
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		else:
			ret = DiGraph(-1);
			ret.spm = pcb.EWiseApply(self.spm, other.spm, pcb.multiplies(), True);
		return ret;

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

	#in-place, so no return value
	def set(self, value):
		self.spm.Apply(pcb.set(value));
		return;

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

	def sum(self, dir=gr.Out):
		if dir == gr.InOut:
			tmp1 = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity());
			tmp2 = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity());
			return ParVec.toParVec(tmp1+tmp2);
		elif dir == gr.In:
			ret = self.spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity());
			return ParVec.toParVec(ret);
		elif dir == gr.Out:
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

	@staticmethod
	def twoDTorus(n):
		N = n*n;
		nvec =   ((ParVec.range(N*4)%N) / n).floor()	 # [0,0,0,...., n-1,n-1,n-1]
		nvecil = ((ParVec.range(N*4)%N) % n).floor()	 # [0,1,...,n-1,0,1,...,n-2,n-1]
		north = gr.Graph._sub2ind((n,n),(nvecil-1) % n,nvec)	
		south = gr.Graph._sub2ind((n,n),(nvecil+1) % n,nvec)
		west = gr.Graph._sub2ind((n,n),nvecil, (nvec-1) % n)
		east = gr.Graph._sub2ind((n,n),nvecil, (nvec+1) % n)
		Ndx = ParVec.range(N*4)
		northNdx = Ndx < N
		southNdx = (Ndx >= N) & (Ndx < 2*N)
		westNdx = (Ndx >= 2*N) & (Ndx < 3*N)
		eastNdx = Ndx >= 3*N
		col = ParVec.zeros(N*4)
		col[northNdx] = north
		col[southNdx] = south
		col[westNdx] = west
		col[eastNdx] = east
		row = ParVec.range(N*4) % N;
		ret = DiGraph(row, col, 1, N)
		return ret

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
		dest = ParVec(self.nvert(),0)
		fringe = SpParVec(self.nvert());
		fringe[source] = 1;
		for i in range(nhop):
			fringe.sprange();
			self.spm.SpMV_SelMax_inplace(fringe.spv);
			dest[fringe] = 1;
		return dest;
		
	# returns:
	#   - source:  a vector of the source vertex for each new vertex
	#   - dest:  a Boolean vector of the new vertices
	#ToDo:  nhop argument?
	def pathsHop(self, source):
		retDest = ParVec(self.nvert(),0)
		retSource = ParVec(self.nvert(),0)
		fringe = SpParVec(self.nvert());
		retDest[fringe] = 1;
		fringe.sprange();
		self.spm.SpMV_SelMax_inplace(fringe.spv);
		retDest[fringe] = 1;
		retSource[fringe] = fringe;
		return (retSource, retDest);
		
	def centrality(self, alg, **kwargs):
	#		ToDo:  Normalize option?
		if alg=='exactBC':
			#cent = DiGraph._approxBC(self, sample=1.0, **kwargs)
			cent = DiGraph._bc(self, 1.0, self.nvert())
		elif alg=='approxBC':
			cent = DiGraph._approxBC(self, **kwargs);
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

            A = self.copy();
	    self.ones();			
	    #Aint = self.ones();	# not needed;  Gs only int for now
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
		nsp = DiGraph(ParVec.range(curSize), batch, 1, curSize, N);
	
	
	        depth = 0;
	        #OLD fringe = Aint[batch,:];   
		fringe = A[batch,ParVec.range(N)];
	
	        while fringe.nedge() > 0:
	            depth = depth+1;
	            #print (depth, fringe.getnnz()) 
	            # add in shortest path counts from the fringe
	            nsp = nsp+fringe
                    tmp = fringe.copy();
                    tmp.bool();
	            bfs.append(tmp);
	            tmp = fringe._SpMM(A);
	            fringe = tmp.mulNot(nsp);
	
		bcu = DiGraph.fullyConnected(curSize,N);
	
	        # compute the bc update for all vertices except the sources
	        for depth in range(depth-1,0,-1):
	            # compute the weights to be applied based on the child values
	            #old w = bfs[depth].multiply(nspInv).multiply(bcu);
                    w = bfs[depth] / nsp * bcu;
	            # Apply the child value weights and sum them up over the parents
	            # then apply the weights based on parent values
                    w.T()
                    w = A._SpMM(w)
                    w.T()
                    w *= bfs[depth-1]
                    w *= nsp
	            bcu += w
	            #old bcu = bcu + (A*w.T).T.mul(bfs[depth-1]).mul(nsp);
	
	        # update the bc with the bc update
	        bc = bc + bcu.sum(Out)	# column sums
	
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
InOut = gr.InOut;
In = gr.In;
Out = gr.Out;
