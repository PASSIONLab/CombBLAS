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

	# works in place, so no return value
	def spones(self):		
		self.spm.Apply(pcb.set(1));
		return;

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

	def __init__(self, length, init=0):
		if length >= 0:
			self.dpv = pcb.pyDenseParVec(length, init);

	def __abs__(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.abs()
		return ret;

	def __add__(self, other):
		if type(other) == int or type(other) == float:
			ret = ParVec(-1);
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.plus(), other));
			return ret;
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		if isinstance(other,SpParVec):
			ret = other + self;	# SPV = SPV + DPV
		else:	#elif  instance(other,ParVec):
			ret = ParVec(-1);
			ret.dpv = self.dpv + other.dpv;
		return ret;

	def __and__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_and(), other));
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = ParVec(-1);
			ret.dpv = self.dpv & other.dpv;
		return ret;

	def __div__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.divides(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			ret.dpv.EWiseApply(other.dpv, pcb.divides());
		return ret;

	def __eq__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.equal_to(), other));
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv = self.dpv == other.dpv;
		return ret;

	def __getitem__(self, key):
		#ToDo:  when generalized unary operations are supported, 
		#    support SPV = DPV[unary-op()]
		if type(key) == int or type(key) == float:
			if key > self.dpv.len()-1:
				raise IndexError;
			ret = self.dpv[key];
		elif isinstance(key,ParVec):
			if not key.allCloseToInt():
				raise KeyError, 'ParVec key must be all integer'
			ret = ParVec(-1);
			ret.dpv = self.dpv[key.dpv];
		elif isinstance(key,SpParVec):
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			ret = SpParVec(-1);
			ret.spv = self.dpv.sparse()[key.spv];
		else:
			raise KeyError, 'Key must be integer scalar, ParVec, or SpParVec'
		return ret;

	def __ge__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater_equal(), other));
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.greater_equal(), int(0)));
		return ret;

	def __gt__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater(), other));
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.greater(), int(0)));
		return ret;

	def __iadd__(self, other):
		if type(other) == int or type(other) == float:
			self.dpv.Apply(pcb.bind2nd(pcb.plus(), other));
		elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
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
		if type(other) == int or type(other) == float:
			self.dpv.Apply(pcb.bind2nd(pcb.minus(), other));
		elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			self.dpv -= other.dpv;
		elif isinstance(other,SpParVec):
			raise NotImplementedError, 'ParVec -= SpParVec not implemented'
		return self;

	def __le__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.less_equal(), other));
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.less_equal(), int(0)));
		return ret;

	def __len__(self):
		return self.dpv.len();

	def __lt__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.less(), other));
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv -= other.dpv;
			ret.dpv.Apply(pcb.bind2nd(pcb.less(), int(0)));
		return ret;

	def __mod__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.modulus(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			ret.dpv.EWiseApply(other.dpv, pcb.modulus());
		return ret;

	def __mul__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.multiplies(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv = other.dpv * self.dpv.sparse();
		return ret;

	def __ne__(self, other):
		ret = self.copy();
		if type(other) == int or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.not_equal_to(), other));
		else:	
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv = self.dpv != other.dpv;
		return ret;

	def __neg__(self):
		ret = ParVec(len(self)) - self;
		return ret;

	def __or__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_or(), other));
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = ParVec.zeros(len(self));
			tmp1 = self.copy();
			tmp1 += other;
			tmp2 = SpParVec.toSpParVec(tmp1.dpv.Find(pcb.bind2nd(pcb.greater(),0)));
			ret[tmp2] = 1;
		return ret;

	def __repr__(self):
		self.dpv.printall();
		return ' ';

	def __setitem__(self, key, value):
		if type(key) == int or type(key) == float:
			self.dpv[key] = value;
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise NotImplementedError, "Only Boolean vector indexing implemented"
			elif type(value) == int or type(value) == float:
                                self.dpv.ApplyMasked(pcb.set(0), key.dpv.sparse());
                                tmp = key.dpv.sparse();
                                tmp.Apply(pcb.set(value));
                                self.dpv += tmp;
			else:
				value[key.logical_not()] = 0;
				self[key] = 0;
				self += value; 
		elif isinstance(key,SpParVec):
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			if type(value) == int or type(value) == float:
				self.dpv.ApplyMasked(pcb.set(value), key.spv);
			elif isinstance(value, SpParVec):
				#ToDo:  check that key and value have the same
				# nonnull positions
				self.dpv.ApplyMasked(pcb.set(0),key.spv);
				self.dpv.add(value.spv);
			else:
				raise NotImplementedError, "Indexing of ParVec by SpParVec only allowed for scalar or SpParVec right-hand side"
		else:
			raise KeyError, "Unknown key type"
			

	def __sub__(self, other):
		if type(other) == int or type(other) == float:
			ret = ParVec(-1);
			ret = self.copy();
			ret.dpv.Apply(pcb.bind2nd(pcb.minus(), other));
			return ret;
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		if isinstance(other,SpParVec):
			raise NotImplementedError, 'ParVec - SpParVec not supported'
		else:	#elif isinstance(other,ParVec):
			ret = ParVec(-1);
			ret.dpv = self.dpv - other.dpv;
		return ret;

	def abs(self):
		return abs(self);

	def all(self):
		ret = self.dpv.Count(pcb.identity()) == len(self);
		return ret;

	def allCloseToInt(self):
		eps = float(np.finfo(np.float).eps);
		ret = ((self % 1.0) < eps).all()
		return ret;

	def any(self):
		ret = self.dpv.any();
		return ret;

	@staticmethod
	def broadcast(sz,val):
		ret = ParVec(-1);
		ret.dpv = pcb.pyDenseParVec(sz,val);
		return ret;
	
	def ceil(self):
		ret = -((-self).floor());
		return ret;

	def copy(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.copy()
		return ret;

	def find(self):
		ret = SpParVec(-1);
		ret.spv = self.dpv.Find(pcb.bind2nd(pcb.not_equal_to(),0.0));
		return ret;

	def findInds(self):
		ret = ParVec(-1);
		ret.dpv = self.dpv.FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0));
		return ret;

	def floor(self):
		ret = ParVec.zeros(len(self));
		neg = self < 0;
		sgn = self.sign();
		retneg = -(abs(self) + 1 - abs(self % 1));
		retpos = self - (self % 1);
		ret[neg] = retneg;
		ret[neg.logical_not()] = retpos;
		return ret;

	def int_(self):
		ret = (self + 0.5).floor();
		return ret;

	def isBool(self):
		eps = float(np.finfo(np.float).eps);
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all();
		return ret;

	#FIX:  "logicalNot"?
	def logical_not(self):
		ret = self.copy()
		ret.dpv.Apply(pcb.logical_not());
		return ret;

	def max(self):
		#ToDo:  avoid conversion to sparse when PV.max() avail
		ret = self.dpv.sparse().Reduce(pcb.max())
		return ret;

	def min(self):
		#ToDo:  avoid conversion to sparse when PV.min() avail
		ret = self.dpv.sparse().Reduce(pcb.min())
		return ret;

	def nn(self):
		return len(self) - self.dpv.getnnz();

	def nnn(self):
		return self.dpv.getnnz();

	def norm(self,ord=None):
		if ord==1:
			ret = self.dpv.Reduce(pcb.plus(),pcb.abs());
			return ret;
		else:
			raise ValueError, 'Unknown order for norm'

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
		ret = ParVec(-1);
		ret.dpv = pcb.pyDenseParVec.range(stop-start,start);
		return ret;

	def sign(self):
		ret = self / abs(self);
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
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
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
		if type(key) == int or type(key) == float:
			del self.spv[key];
		else:
			del self.spv[key.dpv];	
		return;

	def __div__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.divides(), other));
		else:
			raise NotImplementedError, 'SpParVec:__div__: no SpParVec / SpParVec division'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			#ret.spv.EWiseApply(.....pcb.divides())
		return ret;

	def __eq__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(),int(0)));
		return ret;

	def __getitem__(self, key):
		if type(key) == int or type(key) == float:
			if key > len(self.spv)-1:
				raise IndexError;
			ret = self.spv[key];
		elif isinstance(key,SpParVec):
			ret = SpParVec(-1);
			ret.spv = self.spv[key.spv];
		else:
			raise KeyError, 'SpParVec indexing only by SpParVec or integer scalar'
		return ret;

	def __ge__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(),int(0)));
		return ret;

	def __gt__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.greater(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.greater(),int(0)));
		return ret;

	def __iadd__(self, other):
		if type(other) == int or type(other) == float:
			self.spv.Apply(pcb.bind2nd(pcb.plus(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self.spv += other.spv;
			else:
				self.spv += other.dpv;
		return self;
		
	def __isub__(self, other):
		if type(other) == int or type(other) == float:
			self.spv.Apply(pcb.bind2nd(pcb.minus(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self.spv -= other.spv;
			else:
				self.spv -= other.dpv;
		return self;
		
	def __len__(self):
		return len(self.spv);

	def __le__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(),int(0)));
		return ret;

	def __lt__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.less(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			ret.spv = self.spv - other.spv;
			ret.spv.Apply(pcb.bind2nd(pcb.less(),int(0)));
		return ret;

	def __mod__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.modulus(), other));
		else:
			raise NotImplementedError, 'SpParVec:__mod__: no SpParVec / SpParVec modulus'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			#ret.spv.EWiseApply(.....pcb.modulus())
		return ret;

	def __mul__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.multiplies(), other));
		else:
			if not isinstance(other, ParVec):
				raise NotImplementedError, 'SpParVec:__mul__: only SpParVec * ParVec'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy();
			pcb.EWiseMult_inplacefirst(ret.spv, other.dpv, False, 0);
		return ret;

	def __ne__(self, other):
		if type(other) == int or type(other) == float:
			ret = self.copy();
			ret.spv.Apply(pcb.bind2nd(pcb.not_equal_to(), other));
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
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
		if type(key) == int or type(key) == float:
			if key > len(self.spv)-1:
				raise IndexError;
			self.spv[key] = value;
		elif isinstance(key,ParVec):
			if isinstance(value,ParVec):
				pass;
			elif type(value) == float or type(value) == int:
				value = ParVec(len(key),value);
			else:
				raise KeyError, 'Unknown value type'
			if len(self.spv) != len(key.dpv) or len(self.spv) != len(value.dpv):
				raise IndexError, 'Key must same length as SpParVec'
			self.spv[key.dpv] = value.dpv;
		elif type(key) == str and key == 'nonnull':
			self.spv.Apply(pcb.set(value));
		else:
			raise KeyError, 'Unknown key type'
		return;
		

	def __sub__(self, other):
		ret = self.copy();
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		if isinstance(other,SpParVec):
			ret.spv = self.spv - other.spv;
		else:
			ret.spv = self.spv - other.dpv;
		return ret;

	def all(self):
		#FIX: is counting #nonnulls, not #Trues (nonzeros)
		ret = self.spv.Count(pcb.identity()) == self.nnn();
		return ret;

	def allCloseToInt(self):
		return True;
		eps = float(np.finfo(np.float).eps);
		ret = ((self % 1.0) < eps).all()

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

	#ToDO:  need to implement Find when pyCombBLAS method available
	#def find(self):

	#ToDO:  need to implement FindInds when pyCombBLAS method available
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
	
	#in-place, so no return value;
	def spones(self):
		self.spv.Apply(pcb.set(1));
		return;

	#in-place, so no return value;
	def sprange(self):
		self.spv.setNumToInd();

	def sum(self):
		ret = self.spv.Reduce(pcb.plus());
		return ret;

	#TODO:  check for class being PyDenseParVec?
	@staticmethod
	def toSpParVec(SPV):
		ret = SpParVec(-1);
		ret.spv = SPV;
		return ret;
	
def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root();

sendFeedback = feedback.feedback.sendFeedback;

