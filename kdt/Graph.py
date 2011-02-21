import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import feedback
import UFget as uf

class Graph:
	#ToDo: privatize .spm name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	#print "in Graph"

	def __init__(self, *args):
		if len(args) == 0:
                        self.spm = pcb.pySpParMat()
                elif len(args) == 4:
                        #create a DiGraph from i/j/v ParVecs and nv nverts
                        [i,j,v,nv] = args
                        pass
                else:
                        raise NotImplementedError, "only zero and three arguments supported"

	def __getitem__(self, key):
                raise NotImplementedError, "__getitem__ not supported"

	def toParVec(self):	
		ne = self.nedge()
		reti = ParVec(ne)
		retj = ParVec(ne)
		retv = ParVec(ne)
		self.spm.Find(reti.dpv, retj.dpv, retv.dpv)
		return (reti, retj, rev)

        def copy(self):
                ret = Graph()
                ret.spm = self.spm.copy()
                return ret

	def degree(self):
		ret = self.spm.Reduce(pcb.pySpParMat.Column(),pcb.plus())
                return ParVec.toParVec(pcb.pyDenseParVec.toPyDenseParVec(ret))

        @staticmethod
        def load(fname):
                ret = Graph()
                ret.spm = pcb.pySpParMat.load(fname)
                return ret

	def nedge(self):
		"""
		returns the number of edges in the Graph instance, including 
		edges with zero weight.
		"""
		return self.spm.getnnz()

	def nvert(self):
		return self.spm.getnrow()

	# works in place, so no return value
	def ones(self):		
		"""
		sets every edge in the Graph to the value 1.

		Input Argument:
			self:  a Graph instance, modified in place.

		Output Argument:  
			None.

		SEE ALSO:  set
		"""
		self.spm.Apply(pcb.set(1))
		return

	# works in place, so no return value
	def set(self, value):		
		"""
		sets every edge in the Graph to the passed value.

		Input Argument:
			self:  a Graph instance, modified in place.
			value:  a scalar integer or double-precision floating-
			    point value.

		Output Argument:  
			None.

		SEE ALSO:  ones
		"""
		self.spm.Apply(pcb.set(1))
		return

	@staticmethod
	def _sub2ind(size, row, col):		# ToDo:  extend to >2D
		(nr, nc) = size
		ndx = row + col*nr
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
			self.dpv = pcb.pyDenseParVec(length, init)

	def __abs__(self):
		ret = ParVec(-1)
		ret.dpv = self.dpv.abs()
		return ret

	def __add__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.plus(), other))
			return ret
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret.dpv.EWiseApply(other.dpv,pcb.plus())
			else:
				ret.dpv.EWiseApply(other.spv,pcb.plus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __and__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_and(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.logical_and())
		return ret

	def __div__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			if other==0 or other==0.0:
				raise ZeroDivisionError
			ret.dpv.Apply(pcb.bind2nd(pcb.divides(), other))
			return ret
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			if isinstance(other, ParVec):
				ret.dpv.EWiseApply(other.dpv,pcb.divides())
			else:
				ret.dpv.EWiseApply(other.spv,pcb.divides())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __eq__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.equal_to(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.equal_to())
		return ret

	def __getitem__(self, key):
		#ToDo:  when generalized unary operations are supported, 
		#    support SPV = DPV[unary-op()]
		if type(key) == int or type(key) == long or type(key) == float:
			if key > self.dpv.len()-1:
				raise IndexError
			ret = self.dpv[key]
		elif isinstance(key,ParVec):
			if not key.allCloseToInt():
				raise KeyError, 'ParVec key must be all integer'
			ret = ParVec(-1)
			ret.dpv = self.dpv[key.dpv]
		elif isinstance(key,SpParVec):
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			if key.isBool():
				ret = ParVec(-1)
				ndx = key.copy()
				ndx.sprange()
				ret.dpv = self.dpv[ndx.spv.dense()]
			else:
				ret = SpParVec(-1)
				ret.spv = self.dpv.sparse()[key.spv]
		else:
			raise KeyError, 'Key must be integer scalar, ParVec, or SpParVec'
		return ret

	def __ge__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater_equal(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.greater_equal())
		return ret

	def __gt__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.greater(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.greater())
		return ret

	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self.dpv.Apply(pcb.bind2nd(pcb.plus(), other))
			return 
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self.dpv.EWiseApply(other.dpv,pcb.plus())
			else:
				self.dpv.EWiseApply(other.spv,pcb.plus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __imul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self.dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self.dpv.EWiseApply(other.dpv,pcb.multiplies())
			else:
				self.dpv.EWiseApply(other.spv,pcb.multiplies())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __invert__(self):
		if not self.isBool():
			raise NotImplementedError, "only implemented for Boolean"
		ret = self.copy()
		ret.dpv.Apply(pcb.logical_not())
		return ret

	def __isub__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self.dpv.Apply(pcb.bind2nd(pcb.minus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self.dpv.EWiseApply(other.dpv,pcb.minus())
			else:
				self.dpv.EWiseApply(other.spv,pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __le__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.less_equal(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.less_equal())
		return ret

	def __len__(self):
		return self.dpv.len()

	def __lt__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.less(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.less())
		return ret

	def __mod__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			if other==0 or other==0.0:
				raise ZeroDivisionError
			ret.dpv.Apply(pcb.bind2nd(pcb.modulus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			if isinstance(other, ParVec):
				ret.dpv.EWiseApply(other.dpv,pcb.modulus())
			else:
				ret.dpv.EWiseApply(other.spv,pcb.modulus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __mul__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret.dpv.EWiseApply(other.dpv,pcb.multiplies())
			else:
				ret.dpv.EWiseApply(other.spv,pcb.multiplies())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __ne__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.not_equal_to(), other))
		else:	
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.not_equal_to())
		return ret

	def __neg__(self):
		ret = self.copy()
		ret.dpv.Apply(pcb.negate())
		return ret

	def __or__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_or(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.logical_or())
		return ret

	def __radd__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.plus(), other))
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __rdiv__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			if (self == 0).any():
				raise ZeroDivisionError
			ret = ParVec.zeros(len(self)) + other
			ret.dpv.EWiseApply(self.dpv, pcb.divides())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret
	def __rmul__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	_REPR_MAX = 100;
	_REPR_WARN = 0
	def __repr__(self):
		"""
		prints the first N elements of the ParVec instance, where N
		is equal to the value of self._REPR_MAX.

		SEE ALSO:  printAll
		"""
		if len(self) > self._REPR_MAX:
			tmp = self[ParVec.range(self._REPR_MAX)]
			if self._REPR_WARN == 0:
				print "Limiting print-out to first %d elements" % self._REPR_MAX
				# NOTE: not setting WARN to 1, so will print
				# every time
				#self._REPR_WARN = 1
			tmp.dpv.printall()
		else:
			self.dpv.printall()
		return ' '

	def __rsub__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			ret = ParVec.zeros(len(self)) + other
			
			ret.dpv.EWiseApply(self.dpv, pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __setitem__(self, key, value):
		if type(key) == int or type(key) == long or type(key) == float:
			self.dpv[key] = value
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise NotImplementedError, "Only Boolean vector indexing implemented"
			else:
				# zero the target elements
                                self.dpv.ApplyMasked(pcb.set(0), key.dpv.sparse(0))
				if type(value) == int or type(value) == long or type(value) == float:
                                	tmp = key.dpv.sparse()
                                	tmp.Apply(pcb.set(value))
				else:
					# restrict the changed elements to key
					tmp = (value * key).dpv.sparse()
                                self.dpv += tmp
		elif isinstance(key,SpParVec):
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			if type(value) == int or type(value) == long or type(value) == float:
				self.dpv.ApplyMasked(pcb.set(value), key.spv)
			elif isinstance(value, SpParVec):
				#ToDo:  check that key and value have the same
				# nonnull positions
				self.dpv.ApplyMasked(pcb.set(0),key.spv)
				self.dpv.add(value.spv)
			else:
				raise NotImplementedError, "Indexing of ParVec by SpParVec only allowed for scalar or SpParVec right-hand side"
		else:
			raise KeyError, "Unknown key type"
			

	def __sub__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.minus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret.dpv.EWiseApply(other.dpv,pcb.minus())
			else:
				ret.dpv.EWiseApply(other.spv,pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __xor__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.dpv.Apply(pcb.bind2nd(pcb.logical_or(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret.dpv.EWiseApply(other.dpv,pcb.logical_xor())
		return ret

	def abs(self):
		return abs(self)

	def all(self):
		ret = self.dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == len(self)
		return ret

	def allCloseToInt(self):
		if len(self) == 0:
			return True;
		eps = float(np.finfo(np.float).eps)
		ret = (((self % 1.0) < eps) | ((1.0-(self % 1.0)) < eps)).all()
		return ret

	def any(self):
		ret = self.dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) > 0
		return ret

	@staticmethod
	def broadcast(sz,val):
		ret = ParVec(-1)
		ret.dpv = pcb.pyDenseParVec(sz,val)
		return ret
	
	def ceil(self):
		ret = -((-self).floor())
		return ret

	def copy(self):
		ret = ParVec(-1)
		ret.dpv = self.dpv.copy()
		return ret

	def find(self):
		ret = SpParVec(-1)
		ret.spv = self.dpv.Find(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret

	def findInds(self):
		ret = ParVec(-1)
		ret.dpv = self.dpv.FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret

	def floor(self):
		ret = ParVec.zeros(len(self))
		neg = self < 0
		sgn = self.sign()
		retneg = -(abs(self) + 1 - abs(self % 1))
		retpos = self - (self % 1)
		ret[neg] = retneg
		ret[neg.logical_not()] = retpos
		return ret

	def isBool(self):
		eps = float(np.finfo(np.float).eps)
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		return ret

	def logical_not(self):
		ret = self.copy()
		ret.dpv.Apply(pcb.logical_not())
		return ret

	#ToDo:  avoid conversion to sparse when PV.max() /min avail

	def max(self):
		ret = self.dpv.Reduce(pcb.max(), pcb.identity())
		return ret

	def min(self):
		ret = self.dpv.Reduce(pcb.min(), pcb.identity())
		return ret

	def nn(self):
		ret = self.dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(),0), pcb.set(1), pcb.set(0)))
		return ret

	def nnz(self):
		ret = self.dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		return ret

	def norm(self,ord=None):
		if ord==1:
			ret = self.dpv.Reduce(pcb.plus(),pcb.abs())
			return ret
		else:
			raise ValueError, 'Unknown order for norm'

	@staticmethod
	def ones(sz):
		ret = ParVec(-1)
		ret.dpv = pcb.pyDenseParVec(sz,1)
		return ret
	
	def printAll(self):
		"""
		prints all elements of a ParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self.dpv.printall()
		return ' '

	def randPerm(self):
		self.dpv.RandPerm()

	@staticmethod
	def range(arg1, *args):
		if len(args) == 0:
			start = 0
			stop = arg1
		elif len(args) == 1:	
			start = arg1
			stop = args[0]
		else:
			raise NotImplementedError, "No 3-argument range()"
		if start > stop:
			raise ValueError, "start > stop"
		ret = ParVec(-1)
		ret.dpv = pcb.pyDenseParVec.range(stop-start,start)
		return ret

	def round(self):
		ret = (self + 0.5).floor()
		return ret

	def sign(self):
		ret = (self >= 0) - (self < 0)
		return ret
	
	def sum(self):
		ret = self.dpv.Reduce(pcb.plus(), pcb.identity())
		return ret

	@staticmethod
	def toParVec(DPV):
		if not isinstance(DPV, pcb.pyDenseParVec):
			raise TypeError, 'Only supported for pyDenseParVec instances'
		ret = ParVec(-1)
		ret.dpv = DPV
		return ret
	
	#ToDo:  allow user to pass/set null value

	def toSpParVec(self):
		ret = SpParVec(-1)
		ret.spv = self.dpv.sparse()
		return ret

	@staticmethod
	def zeros(sz):
		ret = ParVec(-1)
		ret.dpv = pcb.pyDenseParVec(sz,0)
		return ret
	

class SpParVec:
	#Note:  all comparison ops (__ne__, __gt__, etc.) only compare against
	#   the non-null elements

	def __init__(self, length):
		if length > 0:
			self.spv = pcb.pySpParVec(length)

	def __abs__(self):
		ret = self.copy()
		ret.spv.Apply(pcb.abs())
		return ret

	def __add__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.spv.Apply(pcb.bind2nd(pcb.plus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other,SpParVec):
				ret.spv = self.spv + other.spv
			else:
				ret.spv = self.spv + other.dpv
		return ret

	def __and__(self, other):
		"""
		performs a logical And between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where both of the input vectors are True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		ret.toBool()
		tmpOther = other.copy()
		tmpOther.toBool()
		ret += tmpOther
		ret.spv.Apply(pcb.bind2nd(pcb.equal_to(),2))
		return ret


	def __delitem__(self, key):
		if type(other) == int or type(other) == long or type(other) == float:
			del self.spv[key]
		else:
			del self.spv[key.dpv];	
		return

	def __div__(self, other):
		"""
		divides each element of the first argument (a SpParVec instance),
		by either a scalar or the corresonding element of the second 
		SpParVec instance, with a non-null element where either of the 
		two input vectors was nonnull.
		
		Note:  ZeroDivisionException will be raised if any element of 
		the second argument is zero.

		Note:  For v0.1, the second argument may only be a scalar.
		"""
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.spv.Apply(pcb.bind2nd(pcb.divides(), other))
		else:
			raise NotImplementedError, 'SpParVec:__div__: no SpParVec / SpParVec division'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			#ret.spv.EWiseApply(.....pcb.divides())
		return ret

	def __eq__(self, other):
		"""
		calculates the Boolean equality of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the scalar and the corresponding
		element of the first argument are equal.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and equal.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.equal_to(),int(0)))
		return ret

	def __getitem__(self, key):
		if type(key) == int or type(key) == long or type(key) == float:
			if key > len(self.spv)-1:
				raise IndexError
			ret = self.spv[key]
		elif isinstance(key,SpParVec):
			ret = SpParVec(-1)
			ret.spv = self.spv[key.spv]
		else:
			raise KeyError, 'SpParVec indexing only by SpParVec or integer scalar'
		return ret

	def __ge__(self, other):
		"""
		calculates the Boolean greater-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than or equal to the second.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.greater_equal(),int(0)))
		return ret

	def __gt__(self, other):
		"""
		calculates the Boolean greater-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than the second.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.greater(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.greater(),int(0)))
		return ret

	def __iadd__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			self.spv.Apply(pcb.bind2nd(pcb.plus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self.spv += other.spv
			else:
				self.spv += other.dpv
		return self
		
	def __isub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where either of the two input 
		arguments was nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			self.spv.Apply(pcb.bind2nd(pcb.minus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self.spv -= other.spv
			else:
				self.spv -= other.dpv
		return self
		
	def __len__(self):
		"""
		returns the length (the maximum number of potential nonnull elements
		that could exist) of a SpParVec instance.
		"""
		return len(self.spv)

	def __le__(self, other):
		"""
		calculates the Boolean less-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than or equal to the second.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.less_equal(),int(0)))
		return ret

	def __lt__(self, other):
		"""
		calculates the Boolean less-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than the second.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.less(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.less(),int(0)))
		return ret

	def __mod__(self, other):
		"""
		calculates the modulus of each element of the first argument by the
		second argument (a scalar or a SpParVec instance), with a nonnull
		element where the input SpParVec argument(s) were nonnull.

		Note:  for v0.1, only a scalar divisor is supported.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.modulus(), other))
		else:
			raise NotImplementedError, 'SpParVec:__mod__: no SpParVec / SpParVec modulus'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			#ret.spv.EWiseApply(.....pcb.modulus())
		return ret

	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a SpParVec instance), with a nonnull element where 
		the input SpParVec argument(s) were nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		else:
			if not isinstance(other, ParVec):
				raise NotImplementedError, 'SpParVec:__mul__: only SpParVec * ParVec'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			pcb.EWiseMult_inplacefirst(ret.spv, other.dpv, False, 0)
		return ret

	def __ne__(self, other):
		"""
		calculates the Boolean not-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is not equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is not equal to the second.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.spv.Apply(pcb.bind2nd(pcb.not_equal_to(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret.spv = self.spv - other.spv
			ret.spv.Apply(pcb.bind2nd(pcb.not_equal_to(),int(0)))
		return ret

	def __neg__(self):
		"""
		negates each nonnull element of the passed SpParVec instance.
		"""
		ret = self.copy()
		ret.spv.Apply(pcb.negate())
		return ret

	def __or__(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where at least one of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		ret.toBool()
		tmpOther = other.copy()
		tmpOther.toBool()
		ret += tmpOther
		ret.spv.Apply(pcb.bind2nd(pcb.greater(),0))
		return ret

	_REPR_MAX = 100;
	_REPR_WARN = 0
	def __repr__(self):
		"""
		prints the first N elements of the SpParVec instance, where N
		is roughly equal to the value of self._REPR_MAX.

		SEE ALSO:  printAll
		"""
		if self.nnn() > self._REPR_MAX:
			tmplen = self._REPR_MAX*len(self)/self.nnn()
			tmpndx = SpParVec.range(tmplen)
			tmp = self[tmpndx]
			if self._REPR_WARN == 0:
				print "Limiting print-out to first %d elements" % tmp.nnn()
				# NOTE:  not setting WARN to 1
				#self._REPR_WARN = 1
			tmp.spv.printall()
		else:
			self.spv.printall()
		return ' '


	def __setitem__(self, key, value):
		if type(key) == int or type(key) == long or type(key) == float:
			if key > len(self.spv)-1:
				raise IndexError
			self.spv[key] = value
		elif isinstance(key,ParVec):
			if isinstance(value,ParVec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = ParVec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self.spv) != len(key.dpv) or len(self.spv) != len(value.dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self.spv[key.dpv] = value.dpv
		elif isinstance(key,SpParVec):
			if isinstance(value,ParVec):
				pass
			elif isinstance(value,SpParVec):
				value = value.toParVec()
			elif type(value) == float or type(value) == long or type(value) == int:
				tmp = value
				value = key.copy()
				value.set(tmp)
				value = value.toParVec()
			else:
				raise KeyError, 'Unknown value type'
			key = key.toParVec()
			if len(self.spv) != len(key.dpv) or len(self.spv) != len(value.dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self.spv[key.dpv] = value.dpv
		elif type(key) == str and key == 'nonnull':
			self.spv.Apply(pcb.set(value))
		else:
			raise KeyError, 'Unknown key type'
		return
		

	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where the input SpParVec argument(s)
		are nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			otherscalar = other
			other = self.copy()
			other.set(otherscalar)
		elif len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		if isinstance(other,SpParVec):
			ret.spv = self.spv - other.spv
		else:
			ret.spv = self.spv - other.dpv
		return ret

	def __xor__(self, other):
		"""
		performs a logical Xor between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where exactly one of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		ret.toBool()
		tmpOther = other.copy()
		tmpOther.toBool()
		ret += tmpOther
		ret.spv.Apply(pcb.bind2nd(pcb.equal_to(),1))
		return ret


	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpParVec instance are True (nonzero), and False otherwise.
		"""
		ret = self.spv.Reduce(pcb.logical_and(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == 1
		return ret

	def allCloseToInt(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpParVec instance have values within epsilon of an integer,
		and False otherwise.
		"""
		if self.nnn() == 0:
			return True;
		eps = float(np.finfo(np.float).eps)
		ret = (((self % 1.0) < eps) | (((-(self%1.0))+1.0)< eps)).all()
		return ret

	def any(self):
		"""
		returns a Boolean True if any of the nonnull elements of the
		SpParVec instance are True (nonzero), and False otherwise.
		"""
		ret = self.spv.Reduce(pcb.logical_or(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == 1
		return ret

	def copy(self):
		"""
		creates a deep copy of the input argument.
		"""
		ret = SpParVec(-1)
		ret.spv = self.spv.copy()
		return ret

	#ToDo:  implement find/findInds when problem of any zero elements
	#         in the sparse vector getting stripped out is solved
	#ToDO:  simplfy to avoid dense() when pySpParVec.Find available
	def find(self):
		"""
		returns the elements of a Boolean SpParVec instance that are both
		nonnull and nonzero.

		Input Argument:
			self:  a SpParVec instance

		Output Argument:
			ret:  a SpParVec instance

		SEE ALSO:  findInds
		"""
		if not self.isBool():
			raise NotImplementedError, 'only implemented for Boolean vectors'
		ret = SpParVec(-1)
		ret.spv = self.spv.dense().Find(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret


	#ToDO:  simplfy to avoid dense() when pySpParVec.FindInds available
	def findInds(self):
		"""
		returns the indices of the elements of a Boolean SpParVec instance
		that are both nonnull and nonzero.

		Input Argument:
			self:  a SpParVec instance

		Output Argument:
			ret:  a ParVec instance of length equal to the number of
			    nonnull and nonzero elements in self

		SEE ALSO:  find
		"""
		if not self.isBool():
			raise NotImplementedError, 'only implemented for Boolean vectors'
		ret = ParVec(-1)
		ret.dpv = self.spv.dense().FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret


	def isBool(self):
		"""
		returns a Boolean scalar denoting whether all elements of the input 
		SpParVec instance are equal to either True (1) or False (0).
		"""
		eps = float(np.finfo(np.float).eps)
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		return ret

	def max(self):
		"""
		returns the maximum value of the nonnull elements in the SpParVec 
		instance.
		"""
		ret = self.spv.Reduce(pcb.max())
		return ret

	def min(self):
		"""
		returns the minimum value of the nonnull elements in the SpParVec 
		instance.
		"""
		ret = self.spv.Reduce(pcb.min())
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		SpParVec instance.

		Note:  for x a SpParVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, nnz
		"""
		return len(self) - self.spv.getnnz()

	def nnn(self):
		"""
		returns the number of non-nulls (existent entries) in the
		SpParVec instance.
	
		Note:  for x a SpParVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nn, nnz
		"""
		#ret = self.spv.Reduce(pcb.plus(), pcb.set(1))
		# FIX: get Reduce version working
		ret = self.spv.getnee()
		return ret

	def nnz(self):
		"""
		returns the number of non-zero entries in the SpParVec
		instance.

		Note:  for x a SpParVec instance, x.nnz() is always less than or
		equal to x.nnn().

		SEE ALSO:  nn, nnn
		"""
		ret = self.spv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd.not_equal_to(),0), pcb.set(1), pcb.set(0))
		return ret

	@staticmethod
	def ones(sz):
		"""
		creates a SpParVec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		ret = SpParVec(-1)
		ret.spv = pcb.pySpParVec.range(sz,0)
		ret.spv.Apply(pcb.set(1))
		return ret

	def printAll(self):
		"""
		prints all elements of a SpParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self.spv.printall()
		return ' '

	@staticmethod
	def range(arg1, *args):
		"""
		creates a SpParVec instance with consecutive integer values.

	range(stop)
	range(start, stop)
		The first form creates a SpParVec instance of length stop whose
		values are all nonnull, starting at 0 and stopping at stop-1.
		The second form creates a SpParVec instance of length stop-start
		whose values are all nonnull, starting at start and stopping at
		stop-1.
		"""
		if len(args) == 0:
			start = 0
			stop = arg1
		elif len(args) == 1:	
			start = arg1
			stop = args[0]
		else:
			raise NotImplementedError, "No 3-argument range()"
		if start > stop:
			raise ValueError, "start > stop"
		ret = SpParVec(-1)
		ret.spv = pcb.pySpParVec.range(stop-start,start)
		return ret
	
	#in-place, so no return value
	def set(self, value):
		"""
		sets every non-null value in the SpParVec instance to the second
		argument, in-place.
		"""
		self.spv.Apply(pcb.set(value))
		return

	#in-place, so no return value
	def spones(self):
		"""
		sets every non-null value in the SpParVec instance to 1, in-place.
		"""
		self.spv.Apply(pcb.set(1))
		return

	#in-place, so no return value
	def sprange(self):
		"""
		sets every non-null value in the SpParVec instance to its position
		(offset) in the vector, in-place.
		"""
		self.spv.setNumToInd()

	def sum(self):
		"""
		returns the sum of all the non-null values in the SpParVec instance.
		"""
		ret = self.spv.Reduce(pcb.plus())
		return ret

	#in-place, so no return value
	def toBool(self):
		"""
		converts the input SpParVec instance, in-place, into Boolean
		values (1.0 for True, 0.0 for False) according to whether the
		initial values are nonzero or zero, respectively.
		"""
		self.spv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
		return

	def topK(self, k):
		"""
		returns the largest k non-null values in the passed SpParVec instance.

		Input Arguments:
			self:  a SpParVec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a SpParVec instance of length k containing the k largest
			    values from the input vector, in ascending order.
		"""
		# ToDo:  not sure a SpParVec return is that useful
		ret = SpParVec(0)
		ret.spv = self.spv.TopK(k)
		return ret

	def toParVec(self):	
		"""
		converts a SpParVec instance into a ParVec instance of the same
		length with the non-null elements of the SpParVec instance placed 
		in their corresonding positions in the ParVec instance.
		"""
		ret = ParVec(-1)
		ret.dpv = self.spv.dense()
		return ret

	@staticmethod
	def toSpParVec(SPV):
		if not isinstance(SPV, pcb.pySpParVec):
			raise TypeError, 'Only accepts pySpParVec instances'
		ret = SpParVec(-1)
		ret.spv = SPV
		return ret
	
def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root()

sendFeedback = feedback.feedback.sendFeedback
UFget = uf.UFget
UFdownload = uf.UFdownload

# which direction(s) of edges to include
InOut = 1
In = 2
Out = 3

