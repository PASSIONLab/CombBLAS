import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import PyCombBLAS as PCB
import feedback

class Graph:
	#ToDo: privatize .spm name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	#print "in Graph"

	def __init__(self, *args):
		if len(args) == 0:
                        self.spm = PCB.PySpParMat();
                elif len(args) == 4:
                        #create a DiGraph from i/j/v ParVecs and nv nverts
                        [i,j,v,nv] = args;
                        pass;
                else:
                        raise NotImplementedError, "only zero and three arguments supported"


	# which direction(s) of edges to include
	@staticmethod
	def InOut():
		return 1;

	@staticmethod
	def In():
		return 2;

	@staticmethod
	def Out():
		return 3;

	#FIX:  when pcb.find() exposed
	@staticmethod
	def toVectors(spmat):		# similar to toEdgeV, except returns arrays
		[i, j] = spmat.nonzero();		# NEW:  don't see a way to do this with current SpParMat
		v = spmat[i,j];				# not NEW:  looks like already supported by general indexing
		return ((i,j), v)

        def copy(self):
                ret = Graph();
                ret.spm = self.spm.copy();
                return ret;

	def degree(self):
		ret = self.spm.pySPM.Reduce(pcb.pySpParMat.Column(),pcb.plus());
                return ParVec.toParVec(PCB.PyDenseParVec.toPyDenseParVec(ret));

        @staticmethod
        def load(fname):
                ret = Graph();
                ret.spm = PCB.PySpParMat.load(fname);
                return ret;

	def nedge(self):
		return self.spm.nedge();

	def nvert(self):
		return self.spm.nvert();

	#FIX: use Apply();  change name to ones()?
	@staticmethod
	def _spones(spmat):		
		[nr, nc] = spmat.shape();
		[ij, ign] = Graph._toEdgeV(spmat);
		return Graph.Graph(ij, spv.ones(len(ign)));

	@staticmethod
	def _sub2ind(size, row, col):		# ToDo:  extend to >2D
		(nr, nc) = size;
		ndx = row + col*nr;
		return ndx

	@staticmethod
	def _SpMV_times_max(X,Y):		# ToDo:  extend to 2-/3-arg versions 
		[nrX, ncX] = X.shape()
		[nrY, ncY] = Y.shape()
		if ncX != nrY:
			raise ValueError, "Inner dimensions of X and Y do not match"
		if ncY > 1:
			raise ValueError, "Y must be a column vector"

		Z = spm.SpMV_SelMax(X, Y);	# assuming function creates its own output array

		return Z
		

class ParVec:

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
		if type(other) == int:
			ret = self.copy();
			ret.dpv.pyDPV.Apply(pcb.bind2nd(pcb.divides(), other));
		else:
			ret = ParVec(len(self));
			selfcopy = self.copy();
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

	def __invert__(self):
		if not self.isBool():
			raise NotImplementedError, "only implemented for Boolean"
		ret = self.copy();
		ret.dpv.pyDPV.Apply(pcb.logical_not());
		return ret;

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
		if type(other) == int:
			ret.dpv.pyDPV.Apply(pcb.bind2nd(pcb.modulus(), other));
		else:
			while (ret >= other).any():
				tmp = ret >= other;
				ret[tmp] = ret - other;
		return ret;

	def __mul__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret.dpv = (self.dpv * PCB.PyDenseParVec(len(self),other).sparse()).dense();
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
				if not key.isBool():
					raise NotImplementedError, "Only Boolean vector indexing implemented"
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

	def findInds(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.findInds();
		return ret;

	def nn(self):
		return len(self) - self.dpv.getnnn();

	def nnn(self):
		return self.dpv.getnnn();

	def isBool(self):
		tmp1 = len((self<0).findInds())==0;
		tmp2 = len((self>1).findInds())==0;
		return tmp1 & tmp2;

	#FIX:  "logicalNot"?
	def logical_not(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.logical_not();
		return ret;

	@staticmethod
	def ones(sz):
		ret = ParVec(-1);
		ret.dpv = PCB.PyDenseParVec.ones(sz);
		return ret;
	
	def printall(self):
		return self.dpv.printall();

	def randPerm(self):
		self.dpv.randPerm();
		#FIX:  have no return value, since changing in place?
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
		ret.dpv = PCB.PyDenseParVec.zeros(sz);
		return ret;
	

#class SpParVec:
#	#print "in SpVertexV"
#
#	def __init__(self, length):
#		self.spv = pcb.pySpParVec(length);

sendFeedback = feedback.sendFeedback;

