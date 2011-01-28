import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import feedback

class Graph:
	#ToDo: privatize .spm name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	#print "in Graph"

	def __init__(self, *args):
		if len(args) == 0:
                        self.spm = pcb.pySpParMat();
                elif len(args) == 4:
                        #create a DiGraph from i/j/v ParVecs and nv nverts
                        [i,j,v,nv] = args;
                        pass;
                else:
                        raise NotImplementedError, "only zero and three arguments supported"

	def __getitem__(self, key):
                raise NotImplementedError, "__getitem__ not supported"

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
		ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus());
                return ParVec.toParVec(pcb.pyDenseParVec.toPyDenseParVec(ret));

        @staticmethod
        def load(fname):
                ret = Graph();
                ret.spm = pcb.pySpParMat.load(fname);
                return ret;

	def nedge(self):
		return self.spm.getnnz();

	def nvert(self):
		return self.spm.getnrow();

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
			self.dpv = pcb.pyDenseParVec(length,0);

	def __abs__(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.abs()
		return ret;

	def __add__(self, other):
		if isinstance(other,SpParVec):
			ret = other + self;	# SPV = SPV + DPV
		elif type(other) == int:
			ret = ParVec(-1);
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.plus(), other));
		else:	#elif  instance(other,ParVec):
			ret = ParVec(-1);
			ret.dpv = self.dpv + other.dpv;
		return ret;

	def __and__(self, other):
		ret = ParVec(-1);
		if type(other) == int:
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_and(), other));
		else: 	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv & other.dpv;
		return ret;

	def __div__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.divides(), other));
		else:
			#FIX:  only works for positive integers
			ret = ParVec(len(self));
			selfcopy = self.copy();
			while (selfcopy >= other).any():
				tmp = selfcopy >= other;
				selfcopy[tmp] = selfcopy - other;
				ret[tmp] = ret+1;
		return ret;

	def __eq__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.equal_to(), other));
		else:	#elif isinstance(other,ParVec):
			ret.dpv = self.dpv == other.dpv;
		return ret;

	def __getitem__(self, key):
		#ToDo:  when generalized unary operations are supported, 
		#    support SPV = DPV[unary-op()]
		if type(key) == int:
			if key > self.dpv.len()-1:
				raise IndexError;
			ret = self.dpv[key];
		else:	#elif isinstance(other,ParVec):
			ret = ParVec(-1);
			ret.dpv = self.dpv[key.dpv];
		return ret;

	def __ge__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater_equal(), other));
		else:	#elif isinstance(other,ParVec):
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.greater_equal(), int(0)));
		return ret;

	def __gt__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater(), other));
		else:	#elif isinstance(other,ParVec):
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.greater(), int(0)));
		return ret;

	def __iadd__(self, other):
		if type(other) == int:
			self.dpv.Apply(pcb.bind2nd(pcb.plus(), other));
		elif isinstance(other,ParVec):
			#ToDo:  need to test that self and other are distinct;
			#    += doesn't work if same array on both sides
			self.dpv += other.dpv;
		elif isinstance(other,SpParVec):
			raise NotImplementedError, 'ParVec += SpParVec not implemented'
		return self;

	def __invert__(self):
		if not self.isBool():
			raise NotImplementedError, "only implemented for Boolean"
		ret = self.copy();
		ret.dpv.Apply(pcb.logical_not());
		return ret;

	def __isub__(self, other):
		if type(other) == int:
			self.dpv.Apply(pcb.bind2nd(pcb.minus(), other));
		elif isinstance(other,ParVec):
			self.dpv -= other.dpv;
		elif isinstance(other,SpParVec):
			raise NotImplementedError, 'ParVec -= SpParVec not implemented'
		return self;

	def __le__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.less_equal(), other));
		else:	#elif isinstance(other,ParVec):
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.less_equal(), int(0)));
		return ret;

	def __len__(self):
		return self.dpv.len();

	def __lt__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.less(), other));
		else:	#elif isinstance(other,ParVec):
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.less(), int(0)));
		return ret;

	def __mod__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.modulus(), other));
		else:
			#FIX:  only works for non-negative integers
			while (ret >= other).any():
				tmp = ret >= other;
				ret[tmp] = ret - other;
		return ret;

	def __mul__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.multiplies(), other));
		else:
			ret.dpv = other.dpv * self.dpv.sparse();
		return ret;

	def __ne__(self, other):
		ret = self.copy();
		if type(other) == int:
			ret.dpv.Apply(pcb.bind2nd(pcb.not_equal_to(), other));
		else:	
			ret.dpv = self.dpv != other.dpv;
		return ret;

	def __neg__(self):
		ret = ParVec(len(self)) - self;
		return ret;

	def __repr__(self):
		self.dpv.printall();
		return ' ';

	def __setitem__(self, key, value):
		if type(key) == int:
			self.dpv[key] = value;
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise NotImplementedError, "Only Boolean vector indexing implemented"
			elif type(value) == int:
                                self.dpv.ApplyMasked(pcb.set(0), key.dpv.sparse());
                                tmp = key.dpv.sparse();
                                tmp.Apply(pcb.set(value));
                                self.dpv += tmp;
			else:
				value[key.logical_not()] = 0;
				self[key] = 0;
				self += value; 
		elif isinstance(key,SpParVec):
			raise NotImplementedError, "indexing of ParVec by SpParVec not implemented"
		else:
			raise KeyError, "Unknown key type"
			

	def __sub__(self, other):
		if isinstance(other,SpParVec):
			raise NotImplementedError, 'ParVec - SpParVec not supported'
		elif type(other) == int:
			ret = ParVec(-1);
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.minus(), other));
		else:	#elif isinstance(other,ParVec):
			ret = ParVec(-1);
			ret.dpv = self.dpv - other.dpv;
		return ret;

	def all(self):
		ret = self.dpv.Count(pcb.identity()) == len(self);
		return ret;

	def any(self):
		ret = self.dpv.any();
		return ret;

	@staticmethod
	def broadcast(sz,val):
		ret = ParVec(-1);
		ret.dpv = pcb.pyDenseParVec(sz,val);
		return ret;
	
	def copy(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.copy()
		return ret;

	def find(self):
		ret = SpParVec(-1);
		ret.spv = self.dpv.Find(pcb.bind2nd(pcb.not_equal_to(),0));
		return ret;

	def findInds(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.FindInds(pcb.bind2nd(pcb.not_equal_to(),0));
		return ret;

	def isBool(self):
		ret = self.min() == 0 and self.max() == 1;
		return ret;

	#FIX:  "logicalNot"?
	def logical_not(self):
		ret = self.copy()
		ret.dpv.Apply(pcb.logical_not());
		return ret;

	def max(self):
		#FIX: SPV.max() gives wrong answer if all elements negative
		#ToDo:  avoid conversion to sparse when PV.max() avail
		return self.dpv.sparse().Reduce(pcb.max())

	def min(self):
		#FIX: SPV.min() gives wrong answer if all elements >0
		#ToDo:  avoid conversion to sparse when PV.max() avail
		return self.dpv.sparse().Reduce(pcb.min())

	def nn(self):
		return len(self) - self.dpv.getnnz();

	def nnn(self):
		return self.dpv.getnnz();

	@staticmethod
	def ones(sz):
		ret = ParVec(-1);
		ret.dpv = pcb.pyDenseParVec(sz,1);
		return ret;
	
	def randPerm(self):
		self.dpv.RandPerm();

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
		if start > stop:
			raise ValueError, "start > stop"
		ret = ParVec(0);
		ret.dpv = pcb.pyDenseParVec.range(stop-start,start);
		return ret;
	
	def sum(self):
		#ToDo: avoid converseion to sparse when PV.reduce() avail
		return self.dpv.sparse().Reduce(pcb.plus());

	def sparse(self):
		#ToDo:  allow user to pass/set null value
		ret = ParVec(-1);
		ret.dpv = self.dpv.sparse();
		return ret;

	#TODO:  check for class being PyDenseParVec?
	@staticmethod
	def toParVec(DPV):
		ret = ParVec(-1);
		ret.dpv = DPV;
		return ret;
	
	@staticmethod
	def zeros(sz):
		ret = ParVec(-1);
		ret.dpv = pcb.pyDenseParVec(sz,0);
		return ret;
	

class SpParVec:
	#FIX:  all comparison ops (__ne__, __gt__, etc.) only compare against
	#   the non-null elements

	def __init__(self, length):
		if length > 0:
			self.spv = pcb.pySpParVec(length);

	def __abs__(self):
		ret = self.copy();
		ret.spv.Apply(pcb.abs())
		return ret;

	def __add__(self, other):
		ret = self.copy();
		if isinstance(other,SpParVec):
			ret.spv = self.spv + other.spv;
		else:
			ret.spv = self.spv + other.dpv;
		return ret;

	def __and__(self, other):
		raise NotImplementedError
		ret = self.copy();
		ret.spv = self.spv & other.spv;
		return ret;

	def __delitem__(self, key):
		if type(key) == int:
			del self.spv[key];
		else:
			del self.spv[key.dpv];	
		return;

	def __div__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.divides(), other));
		else:
			raise NotImplementedError, 'SpParVec:__div__: no SpParVec / SpParVec division'
			ret = self.copy();
			#ret.spv.EWiseApply(.....pcb.divides())
		return ret;

	def __eq__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(),int(0)));
		return ret;

	def __getitem__(self, key):
		if type(key) == int:
			if key > len(self.spv)-1:
				raise IndexError;
			ret = self.spv[key];
		elif isinstance(key,SpParVec):
			ret = SpParVec(-1);
			ret.spv = self.spv[key.spv];
		else:
			raise KeyError, 'SpParVec indexing only by SpParVec or scalar'
		return ret;

	def __ge__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(),int(0)));
		return ret;

	def __gt__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.greater(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.greater(),int(0)));
		return ret;

	def __iadd__(self, other):
		if type (other) == int:
			self.spv.Apply(pcb.bind2nd(pcb.plus(), other));
		else:
			if isinstance(other, SpParVec):
				self.spv += other.spv;
			else:
				self.spv += other.dpv;
		return self;
		
	def __isub__(self, other):
		if type (other) == int:
			self.spv.Apply(pcb.bind2nd(pcb.minus(), other));
		else:
			if isinstance(other, SpParVec):
				self.spv -= other.spv;
			else:
				self.spv -= other.dpv;
		return self;
		
	def __len__(self):
		return len(self.spv);

	def __le__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(),int(0)));
		return ret;

	def __lt__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.less(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.less(),int(0)));
		return ret;

	def __mod__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.modulus(), other));
		else:
			raise NotImplementedError, 'SpParVec:__mod__: no SpParVec / SpParVec modulus'
			ret = self.copy();
			#ret.spv.EWiseApply(.....pcb.modulus())
		return ret;

	def __mul__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.multiplies(), other));
		else:
			if not isinstance(other, ParVec):
				raise NotImplementedError, 'SpParVec:__mul__: only SpParVec * ParVec'
			if len(self) != len(other):
				raise ValueError, 'SpParVec:__mul__: different length vectors'
			ret = self.copy();
			pcb.EWiseMult_inplacefirst(ret.spv, other.dpv, False, 0);
		return ret;

	def __ne__(self, other):
		if type(other) == int:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.not_equal_to(), other));
		else:
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.not_equal_to(),int(0)));
		return ret;

	def __neg__(self):
		#ToDo:  best to do with unary_minus() when available
		tmp1 = self.copy();
		tmp1['nonnull'] = 0;
		ret = tmp1 - self;
		return ret;

	def __repr__(self):
		self.spv.printall();
		return ' ';

	def __setitem__(self, key, value):
		if type(key) == int:
			if key > len(self.spv)-1:
				raise IndexError;
			self.spv[key] = value;
		elif isinstance(key,ParVec) and isinstance(value,ParVec):
			if len(self.spv) != len(key.spv) or len(self.spv) != len(value.spv):
				raise IndexError, 'Key must same length as SpParVec'
			self.spv[key.dpv] = value.dpv;
		elif type(key) == str and key == 'nonnull':
			self.spv.Apply(pcb.set(value));
		else:
			raise KeyError, 'Unknown key type'
		return;
		

	def __sub__(self, other):
		ret = self.copy();
		if isinstance(other,SpParVec):
			ret.spv = self.spv - other.spv;
		else:
			ret.spv = self.spv - other.dpv;
		return ret;

	def all(self):
		ret = self.spv.Count(pcb.identity()) == self.nnn();
		return ret;

	def any(self):
		return self.spv.any();

	def copy(self):
		ret = SpParVec(-1);
		ret.spv = self.spv.copy();
		return ret;

	def denseNonnulls(self):	
		ret = ParVec(-1);
		ret.dpv = self.spv.dense();
		return ret;

	#def find(self):

	#def findInds(self):

	def nn(self):
		return len(self) - self.spv.getnnz();

	def nnn(self):
		return self.spv.getnnz();

	@staticmethod
	def ones(sz):
		ret = SpParVec(-1);
		ret.spv = pcb.pySpParVec.range(sz,0);
		ret.spv.Apply(pcb.set(1));
		return ret;

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
		if start > stop:
			raise ValueError, "start > stop"
		ret = SpParVec(-1);
		ret.spv = pcb.pySpParVec.range(stop-start,start);
		return ret;
	
	def sum(self):
		ret = self.spv.Reduce(pcb.plus());
		return ret;

def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root();

sendFeedback = feedback.feedback.sendFeedback;

