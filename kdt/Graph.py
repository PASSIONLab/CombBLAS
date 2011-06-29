import math
#import numpy as np # Adam: TRY TO AVOID THIS IF AT ALL POSSIBLE.
import pyCombBLAS as pcb
import feedback
import UFget as uf

class info:
	@staticmethod
	def eps():
		"""
		Return IEEE floating point machine epsilon.
		The problem with this operation is that Python only provides a portable way to get this
		value in v2.6 and NumPy isn't always available. This function attempts to use whatever
		knows this value or returns a reasonable default otherwise.
		"""
		# try Python v2.6+ float_info
		try:
			from sys import float_info as fi
			return fi.epsilon
		except ImportError:
			pass
			
		# try Numpy
		try:
			import numpy as np
			return float(np.finfo(np.float).eps)
		except ImportError:
			pass
		except AttributeError:
			pass
			
		# return a reasonable value
		return 2.220446049250313e-16;

class Graph:
	#ToDo: privatize ._spm name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	#print "in Graph"

	def __init__(self, *args):
		if len(args) == 0:
                        self._spm = pcb.pySpParMat()
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
		self._spm.Find(reti._dpv, retj._dpv, retv._dpv)
		return (reti, retj, rev)

        def copy(self):
                ret = Graph()
                ret._spm = self._spm.copy()
                return ret

	def degree(self):
		# ToDo:  double-check that Column is the right direction
		ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus())
                return ParVec.toParVec(pcb.pyDenseParVec.toPyDenseParVec(ret))

	@staticmethod
	def load(fname):
		ret = Graph()
		ret._spm = pcb.pySpParMat.load(fname)
		return ret

#	def nedge(self):
#		"""
#		returns the number of edges in the Graph instance, including 
#		edges with zero weight.
#		"""
#		if self.nvert() == 0:
#			return 0
#		else:
#			return self._spm.getnnz()

	def nvert(self):
		return self._spm.getnrow()

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
		self._spm.Apply(pcb.set(1))
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
		self._spm.Apply(pcb.set(1))
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
			self._dpv = pcb.pyDenseParVec(length, init)
	
	@staticmethod
	def load(filename):
		ret = ParVec(1)
		ret._dpv.load(filename)
		return ret

	def __abs__(self):
		ret = ParVec(-1)
		ret._dpv = self._dpv.abs()
		return ret

	def __add__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.plus(), other))
			return ret
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret._dpv.EWiseApply(other._dpv,pcb.plus())
			else:
				ret._dpv.EWiseApply(other._spv,pcb.plus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __and__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.logical_and(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.logical_and())
		return ret

	def __div__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			if other==0 or other==0.0:
				raise ZeroDivisionError
			ret._dpv.Apply(pcb.bind2nd(pcb.divides(), other))
			return ret
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			if isinstance(other, ParVec):
				ret._dpv.EWiseApply(other._dpv,pcb.divides())
			else:
				ret._dpv.EWiseApply(other._spv,pcb.divides())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __eq__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.equal_to(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.equal_to())
		return ret

	def __getitem__(self, key):
		"""
		performs indexing of a ParVec instance on the right-hand side of 
		an equation.  The following forms are supported:
	scalar = parvec[integer scalar]
	parvec = parvec[boolean parvec]
	parvec = parvec[non-boolean parvec]
	parvec = parvec[boolean spparvec]
	spparvec = parvec[non-boolean spparvec]

		Warning:  A boolean ParVec or SpParVec instance is defined as one
		in which every element has a value of True (1.0) or False (0.0).
		The v0.1 implementation does not distinguish a vector which happens
		to have those values from a vector created from a Boolean operation.

		The first form takes as the index an integer scalar and returns
		the corresponding element of the ParVec instance.  This is for
		convenience and is not high performance.

		The second form takes as the index a Boolean ParVec instance and
		returns a ParVec instance with length equal to the number of True
		elements in the ParVec index containing the values of those elements.

		The third form takes as the index a non-Boolean ParVec instance
		containing the integer indices of the desired elements
		and returns a ParVec instance with length equal to the index
		ParVec instance containing the values of the indicated elements

		The fourth form takes as the index a Boolean SpParVec instance
		and returns a ParVec instance with length equal to the number of
		True elements in the SpParVec index containing the values of the
		corresponding elements.

		The fifth form takes as the input a non-Boolean SpParVec instance
		containing the integer indices of the desired elements and returns
		an SpParVec instance with the number of nonnulls equal to the 
		number of nonnulls in the SpParVec index and containing the values
		of the corresponding elements. 
		"""
		#ToDo:  when generalized unary operations are supported, 
		#    support SPV = DPV[unary-op()]
		if type(key) == int or type(key) == long or type(key) == float:
			if key < 0 or key >= self._dpv.len():
				raise IndexError, 'scalar index out of bounds'
			ret = self._dpv[key]
		elif isinstance(key,ParVec):
			if len(key) == 0:
				return ParVec(0)
			if not key.allCloseToInt():
				raise KeyError, 'ParVec key must be all integer'
			if not key.isBool():
				if key.min() < 0 or key.max() >= self._dpv.len():
					raise IndexError, 'at least one index out of bounds'
				ret = ParVec(-1)
				ret._dpv = self._dpv[key._dpv]
			else:
				ret = self[key.toSpParVec().findInds()]
		elif isinstance(key,SpParVec):
			if len(key) == 0:
				return SpParVec(0)
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			if not key.isBool():
				if key.min() < 0 or key.max() >= self._dpv.len():
					raise IndexError, 'at least one index out of bounds'
				ret = SpParVec(-1)
				ret._spv = self._dpv.sparse()[key._spv]
			else:
				ret = ParVec(-1)
				ndx = key.copy()
				ndx2 = (ndx.toParVec()) == 0
				del ndx._spv[ndx2._dpv]
				ndx.spRange()
				ret = self[ndx.toParVec().findInds()]
		else:
			raise KeyError, 'Key must be integer scalar, ParVec, or SpParVec'
		return ret

	def __ge__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.greater_equal(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.greater_equal())
		return ret

	def __gt__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.greater(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.greater())
		return ret

	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self._dpv.Apply(pcb.bind2nd(pcb.plus(), other))
			return 
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self._dpv.EWiseApply(other._dpv,pcb.plus())
			else:
				self._dpv.EWiseApply(other._spv,pcb.plus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __imul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self._dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self._dpv.EWiseApply(other._dpv,pcb.multiplies())
			else:
				self._dpv.EWiseApply(other._spv,pcb.multiplies())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __invert__(self):
		if not self.isBool():
			raise NotImplementedError, "only implemented for Boolean"
		ret = self.copy()
		ret._dpv.Apply(pcb.logical_not())
		return ret

	def __isub__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self._dpv.Apply(pcb.bind2nd(pcb.minus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				self._dpv.EWiseApply(other._dpv,pcb.minus())
			else:
				self._dpv.EWiseApply(other._spv,pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return self

	def __le__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.less_equal(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.less_equal())
		return ret

	def __len__(self):
		return self._dpv.len()

	def __lt__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.less(), other))
		else:	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.less())
		return ret

	def __mod__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			if other==0 or other==0.0:
				raise ZeroDivisionError
			ret._dpv.Apply(pcb.bind2nd(pcb.modulus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if (other==0).any():
				raise ZeroDivisionError
			if isinstance(other, ParVec):
				ret._dpv.EWiseApply(other._dpv,pcb.modulus())
			else:
				ret._dpv.EWiseApply(other._spv,pcb.modulus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __mul__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret._dpv.EWiseApply(other._dpv,pcb.multiplies())
			else:
				ret._dpv.EWiseApply(other._spv,pcb.multiplies())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __ne__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.not_equal_to(), other))
		else:	
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.not_equal_to())
		return ret

	def __neg__(self):
		ret = self.copy()
		ret._dpv.Apply(pcb.negate())
		return ret

	def __or__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.logical_or(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.logical_or())
		return ret

	def __radd__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.plus(), other))
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __rdiv__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			if (self == 0).any():
				raise ZeroDivisionError
			ret = ParVec.zeros(len(self)) + other
			ret._dpv.EWiseApply(self._dpv, pcb.divides())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret
	def __rmul__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.multiplies(), other))
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
				if master():
					print "Limiting print-out to first %d elements" % self._REPR_MAX
				# NOTE: not setting WARN to 1, so will print
				# every time
				#self._REPR_WARN = 1
			tmp._dpv.printall()
		else:
			self._dpv.printall()
		return ' '

	def __rsub__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			ret = ParVec.zeros(len(self)) + other
			
			ret._dpv.EWiseApply(self._dpv, pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __setitem__(self, key, value):
		"""
		performs assignment of a ParVec instance on the left-hand side of
		an equation.  The following forms are supported:
	parvec[integer scalar] = scalar
	parvec[Boolean parvec] = scalar
	parvec[Boolean parvec] = parvec
	parvec[non-Boolean spparvec] = scalar
	parvec[non-Boolean spparvec] = spparvec

		For the first form, the element of the ParVec instance indicated
		by the index is set to the scalar value.

		For the second form, the elements of the ParVec instance corresponding
		to the True elements of the index ParVec instance are set to the
		scalar value.

		For the third form, the elements of the ParVec instance corresponding
		to the True elements of the index ParVec instance are set to the
		corresponding elements of the value ParVec instance.

		For the fourth form, the elements of the ParVec instance corresponding 
		to the nonnull elements of the index SpParVec instance are set to the
		scalar value.

		For the fifth form, the elements of the ParVec instance corresponding
		to the nonnull elements of the index SpParVec instance are set to
		the corresponding values of the value SpParVec instance.  Note that
		key and value SpParVec instances must have nonnull elements in the
		same locations.
		"""
		if type(key) == int or type(key) == long or type(key) == float:
			if key < 0 or key >= len(self):
				raise IndexError, 'Index out of bounds'
			self._dpv[key] = value
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise NotImplementedError, "Only Boolean vector indexing implemented"
			else:
				# zero the target elements
                                self._dpv.ApplyMasked(pcb.set(0), key._dpv.sparse(0))
				if type(value) == int or type(value) == long or type(value) == float:
                                	tmp = key._dpv.sparse()
                                	tmp.Apply(pcb.set(value))
				elif len(key) != len(value):
					raise ValueError, "Value must be same length as Key"
				else:
					# restrict the changed elements to key
					tmp = (value * key)._dpv.sparse()
                                self._dpv += tmp
		elif isinstance(key,SpParVec):
			if not key.allCloseToInt():
				raise KeyError, 'SpParVec key must be all integer'
			if type(value) == int or type(value) == long or type(value) == float:
				self._dpv.ApplyMasked(pcb.set(value), key._spv)
			elif isinstance(value, SpParVec):
				keytmp = key.copy()
				valuetmp = value.copy()
				keytmp.set(1)
				valuetmp.set(1)
				diffpos = ((keytmp+valuetmp) != 2).any()
				if (key.nnn() != value.nnn()) or diffpos:
					raise KeyError, 'SpParVec key and value must have nonnulls in the same positions'
				self._dpv.ApplyMasked(pcb.set(0),key._spv)
				self._dpv.add(value._spv)
			else:
				raise NotImplementedError, "Indexing of ParVec by SpParVec only allowed for scalar or SpParVec right-hand side"
		else:
			raise KeyError, "Unknown key type"
			

	def __sub__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.minus(), other))
		elif isinstance(other,ParVec) or isinstance(other, SpParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, ParVec):
				ret._dpv.EWiseApply(other._dpv,pcb.minus())
			else:
				ret._dpv.EWiseApply(other._spv,pcb.minus())
		else:
			raise NotImplementedError, 'unimplemented type for 2nd operand'
		return ret

	def __xor__(self, other):
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret._dpv.Apply(pcb.bind2nd(pcb.logical_or(), other))
		else: 	#elif isinstance(other,ParVec):
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret._dpv.EWiseApply(other._dpv,pcb.logical_xor())
		return ret


	def abs(self):
		return abs(self)

	def all(self):
		ret = self._dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == len(self)
		return ret

	def allCloseToInt(self):
		if len(self) == 0:
			return True;
		eps = info.eps()
		ret = (((self % 1.0) < eps) | ((1.0-(self % 1.0)) < eps)).all()
		return ret

	def any(self):
		ret = self._dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) > 0
		return ret

	def argmax(self):
		tmp = self._dpv.Reduce(pcb.max(), pcb.identity())
		ret = int((self==tmp).findInds()[0])
		return ret

	def argmin(self):
		tmp = self._dpv.Reduce(pcb.min(), pcb.identity())
		ret = int((self==tmp).findInds()[0])
		return ret

	@staticmethod
	def broadcast(sz,val):
		ret = ParVec(-1)
		ret._dpv = pcb.pyDenseParVec(sz,val)
		return ret
	
	def ceil(self):
		ret = -((-self).floor())
		return ret

	def copy(self):
		ret = ParVec(-1)
		ret._dpv = self._dpv.copy()
		return ret

	def find(self):
		ret = SpParVec(-1)
		ret._spv = self._dpv.Find(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret

	def findInds(self):
		ret = ParVec(-1)
		ret._dpv = self._dpv.FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret

	def floor(self):
		ret = ParVec.zeros(len(self))
		neg = self < 0
		sgn = self.sign()
		retneg = -(abs(self) + abs(self % 1))
		retpos = self - (self % 1)
		ret[neg] = retneg
		ret[neg.logical_not()] = retpos
		return ret

	def hist(self):
		"""
		ToDo:  write docstring
		"""
		if not self.allCloseToInt():
			raise KeyError, 'input values must be all integer'
		selfInt = self.round()
		rngV = ParVec.range(len(self))
		oneV = ParVec.ones(len(self))
		selfMax = int(self.max())
		tmpMat = pcb.pySpParMat(len(self), selfMax+1, 
				rngV._dpv, selfInt._dpv, oneV._dpv);
		retDPV = tmpMat.Reduce(pcb.pySpParMat.Column(), pcb.plus(),
					pcb.identity())
		ret = ParVec(-1)
		ret._dpv = retDPV
		return ret

	def isBool(self):
		if len(self) == 0:
			return True
		eps = info.eps()
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		return ret

	def logical_not(self):
		ret = self.copy()
		ret._dpv.Apply(pcb.logical_not())
		return ret

	def max(self):
		ret = self._dpv.Reduce(pcb.max(), pcb.identity())
		return ret

	def mean(self):
		"""
		calculates the mean (average) of a ParVec instance, returning
		a scalar.
		"""
		return self.sum() / len(self)

	def min(self):
		ret = self._dpv.Reduce(pcb.min(), pcb.identity())
		return ret

	def nn(self):
		ret = self._dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(),0), pcb.set(1), pcb.set(0)))
		return ret

	def nnz(self):
		ret = self._dpv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		return int(ret)

	def norm(self,ord):
		"""
		calculates the norm of a ParVec instance.  The supported norms
		include:
		- 1:  defined as max(sum(abs(x)))
		"""
		if ord <= 0:
			raise ValueError, 'Norm must be positive'
		elif ord==1:
			ret = self._dpv.Reduce(pcb.plus(),pcb.abs())
			return ret
		else:
			ret = self._dpv.Reduce(pcb.plus(),pcb.compose1(pcb.bind2nd(pcb.pow(), ord), pcb.abs()))
			ret = pow(ret, 1.0/ord)
			return ret

	@staticmethod
	def ones(sz):
		ret = ParVec(-1)
		ret._dpv = pcb.pyDenseParVec(sz,1)
		return ret
	
	def printAll(self):
		"""
		prints all elements of a ParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self._dpv.printall()
		return ' '

	def randPerm(self):
		self._dpv.RandPerm()

	@staticmethod
	def range(arg1, *args):
		#print "range with arg1 %d" % arg1
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
		#print "iota with start %d, stop %d" % (start, stop)
		ret._dpv = pcb.pyDenseParVec.range(stop-start,start)
		return ret

	def round(self):
		"""
		"""
		eps = info.eps()
		neg = self < 0
		sgn = self.sign()
		ret = abs(self.copy())
		evenp5 = (abs((ret%2.0) - 0.5)) < eps
		oddp5 =  (abs((ret%2.0) - 1.5)) < eps
		ret[evenp5] = ret-0.1
		ret[oddp5] = ret+0.1
		ret = ((ret+0.5).floor()) * sgn
		return ret

	def sign(self):
		ret = (self >= 0) - (self < 0)
		return ret
	
	# in-place, so no return value
	def sort(self):
		"""
		sorts the values in the passed ParVec instance in-place, in
		ascending order.

		Input Arguments:
			self:  a ParVec instance.

		Output Argument:
			None

		See Also:  sorted
		"""
		self._dpv.Sort()
		return

	def sorted(self):
		"""
		returns a new ParVec instance with the sorted values (in ascending
		order) from the input ParVec instance and a ParVec permutation 
		vector.

		Input Arguments:
			self:  a ParVec instance.

		Output Argument:
			ret:  a tuple containing as its first element a ParVec 
			    instance of the same length containing the sorted
			    values and as its second element the indices of the
			    sorted values in the input vector.

		See Also:  sort
		"""
		ret1 = self.copy();
		ret2 = ParVec(-1)
		ret2._dpv = ret1._dpv.Sort()
		return (ret1, ret2)

	def std(self):
		"""
		calculates the standard deviation of a ParVec instance, returning
		a scalar.  Calculated as sqrt((self-self.mean()).sum)
		"""
		mean = self.mean();
		diff = self - mean
		ret = math.sqrt((diff*diff).sum()/len(self))
		return ret 

	def sum(self):
		ret = self._dpv.Reduce(pcb.plus(), pcb.identity())
		return ret

	@staticmethod
	def toParVec(DPV):
		#if not isinstance(DPV, pcb.pyDenseParVec):
		if not DPV.__class__.__name__ == 'pyDenseParVec':
			raise TypeError, 'Only supported for pyDenseParVec instances'
		ret = ParVec(-1)
		ret._dpv = DPV
		return ret
	
	def topK(self, k):
		"""
		returns the largest k values in the passed SpParVec instance.

		Input Arguments:
			self:  a ParVec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a ParVec instance of length k containing the k largest
			    values from the input vector, in ascending order.
		"""
		ret = ParVec(0)
		ret._dpv = self._dpv.TopK(k)
		return ret

	#ToDo:  allow user to pass/set null value

	def toSpParVec(self):
		ret = SpParVec(-1)
		ret._spv = self._dpv.sparse()
		return ret

	def toSpParVecAll(self):
		"""
		is a hack place-holder for a routine that converts a ParVec to
		an SpParVec, preserving all zeros in the ParVec as nonnull zeros
		in the SpParVec.  This hack version is suitable for use on 
		integer ParVecs, as those deriving from indices.
		"""
		ret = ((self+0.1).toSpParVec())-0.1
		return ret

	@staticmethod
	def zeros(sz):
		ret = ParVec(-1)
		ret._dpv = pcb.pyDenseParVec(sz,0)
		return ret
	

class SpParVec:
	#Note:  all comparison ops (__ne__, __gt__, etc.) only compare against
	#   the non-null elements

	def __init__(self, length):
		if length > 0:
			self._spv = pcb.pySpParVec(length)

	@staticmethod
	def load(filename):
		ret = SpParVec(1)
		ret._spv.load(filename)
		return ret

	def __abs__(self):
		ret = self.copy()
		ret.apply(pcb.abs())
		return ret

	def __add__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		ret = self.copy()
		if type(other) == int or type(other) == long or type(other) == float:
			ret.apply(pcb.bind2nd(pcb.plus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other,SpParVec):
				ret._spv = self._spv + other._spv
			else:
				ret._spv = self._spv + other._dpv
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
		ret.apply(pcb.bind2nd(pcb.equal_to(),2))
		return ret


	def __delitem__(self, key):
		if type(other) == int or type(other) == long or type(other) == float:
			del self._spv[key]
		else:
			del self._spv[key._dpv];	
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
			ret.apply(pcb.bind2nd(pcb.divides(), other))
		else:
			raise NotImplementedError, 'SpParVec:__div__: no SpParVec / SpParVec division'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			#ret._spv.EWiseApply(.....pcb.divides())
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
			ret.apply(pcb.bind2nd(pcb.equal_to(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.equal_to(),int(0)))
		return ret

	def __getitem__(self, key):
		"""
		performs indexing of a SpParVec instance on the right-hand side
		of an equation.  The following forms are supported:
	scalar = spparvec[integer scalar]
	spparvec = spparvec[non-boolean parvec]

		The first form takes as the index an integer scalar and returns
		the corresponding element of the SpParVec instance.  This form is
		for convenience and is not highly performing.

		The second form takes as the index a non-Boolean SpParVec instance
		and returns an SpParVec instance of the same length with the 
		elements of the result corresponding to the nonnull values of the
		index set to the values of the base SpParVec instance. 
		"""
		if type(key) == int or type(key) == long or type(key) == float:
			if key < 0 or key > len(self._spv)-1:
				raise IndexError
			ret = self._spv[key]
		elif isinstance(key,ParVec):
			if key.isBool():
				raise KeyError, "Boolean indexing on right-hand side for SpParVec not supported"
			ret = ParVec(-1)
			ret._dpv = self._spv[key._dpv]
		else:
			raise KeyError, 'SpParVec indexing only by ParVec or integer scalar'
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
			ret.apply(pcb.bind2nd(pcb.greater_equal(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.greater_equal(),int(0)))
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
			ret.apply(pcb.bind2nd(pcb.greater(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.greater(),int(0)))
		return ret

	def __iadd__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			self.apply(pcb.bind2nd(pcb.plus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self._spv += other._spv
			else:
				self._spv += other._dpv
		return self
		
	def __isub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where either of the two input 
		arguments was nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			self.apply(pcb.bind2nd(pcb.minus(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if isinstance(other, SpParVec):
				self._spv -= other._spv
			else:
				self._spv -= other._dpv
		return self
		
	def __len__(self):
		"""
		returns the length (the maximum number of potential nonnull elements
		that could exist) of a SpParVec instance.
		"""
		return len(self._spv)

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
			ret.apply(pcb.bind2nd(pcb.less_equal(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.less_equal(),int(0)))
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
			ret.apply(pcb.bind2nd(pcb.less(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.less(),int(0)))
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
			ret.apply(pcb.bind2nd(pcb.modulus(), other))
		else:
			raise NotImplementedError, 'SpParVec:__mod__: no SpParVec / SpParVec modulus'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			#ret._spv.EWiseApply(.....pcb.modulus())
		return ret

	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a SpParVec instance), with a nonnull element where 
		the input SpParVec argument(s) were nonnull.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret.apply(pcb.bind2nd(pcb.multiplies(), other))
		else:
			if not isinstance(other, ParVec):
				raise NotImplementedError, 'SpParVec:__mul__: only SpParVec * ParVec'
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			pcb.EWiseMult_inplacefirst(ret._spv, other._dpv, False, 0)
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
			ret.apply(pcb.bind2nd(pcb.not_equal_to(), other))
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			ret = self.copy()
			ret._spv = self._spv - other._spv
			ret.apply(pcb.bind2nd(pcb.not_equal_to(),int(0)))
		return ret

	def __neg__(self):
		"""
		negates each nonnull element of the passed SpParVec instance.
		"""
		ret = self.copy()
		ret.apply(pcb.negate())
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
		ret.apply(pcb.bind2nd(pcb.greater(),0))
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
			tmpndx = ParVec.range(tmplen)
			tmp = self[tmpndx]
			if self._REPR_WARN == 0:
				if master():
					print "Limiting print-out to first %d elements" % tmp.nnn()
				# NOTE:  not setting WARN to 1
				#self._REPR_WARN = 1
			tmp._spv.printall()
		else:
			self._spv.printall()
		return ' '


	def __setitem__(self, key, value):
		"""
		performs assignment of an SpParVec instance on the left-hand side of
		an equation.  The following forms are supported.
	spparvec[integer scalar] = scalar
	spparvec[Boolean parvec] = scalar
	spparvec[Boolean parvec] = parvec
	spparvec[non-Boolean spparvec] = scalar
	spparvec[non-Boolean spparvec] = spparvec

		For the first form, the element of the SpParVec instance indicated 
		by the index to the is set to the scalar value.

		For the second form, the elements of the SpParVec instance
		corresponding to the True elements of the index ParVec instance
		are set to the scalar value.

		For the third form, the elements of the SpParVec instance 
		corresponding to the True elements of the index ParVec instance
		are set to the corresponding element of the value ParVec instance.

		For the fourth form, the elements of the SpParVec instance
		corresponding to the nonnull elements of the index SpParVec
		instance are set to the scalar value.

		For the fifth form, the elements of the SpParVec instance 
		corresponding to the nonnull elements of the index SpParVec
		instance are set to the corresponding value of the value
		SpParVec instance.  Note that the base, key, and value SpParVec
		instances must all be of the same length, though the base may
		have a different number of nonzeros from the key and value. 
		"""
		if type(key) == int or type(key) == long or type(key) == float:
			if key > len(self._spv)-1:
				raise IndexError
			self._spv[key] = value
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise KeyError, 'only Boolean ParVec indexing of SpParVecs supported'
			if isinstance(value,ParVec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = ParVec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self._spv) != len(key._dpv) or len(self._spv) != len(value._dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._spv[key._dpv] = value._dpv
		elif isinstance(key,SpParVec):
			if key.isBool():
				raise KeyError, 'Boolean SpParVec indexing of SpParVecs not supported'
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
			if len(self._spv) != len(key._dpv) or len(self._spv) != len(value._dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._spv[key._dpv] = value._dpv
		elif type(key) == str and key == 'nonnull':
			self.apply(pcb.set(value))
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
			ret._spv = self._spv - other._spv
		else:
			ret._spv = self._spv - other._dpv
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
		ret.apply(pcb.bind2nd(pcb.equal_to(),1))
		return ret


	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpParVec instance are True (nonzero), and False otherwise.
		"""
		ret = self._spv.Reduce(pcb.logical_and(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == 1
		return ret

	def allCloseToInt(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpParVec instance have values within epsilon of an integer,
		and False otherwise.
		"""
		if self.nnn() == 0:
			return True;
		eps = info.eps()
		ret = (((self % 1.0) < eps) | (((-(self%1.0))+1.0)< eps)).all()
		return ret

	def any(self):
		"""
		returns a Boolean True if any of the nonnull elements of the
		SpParVec instance are True (nonzero), and False otherwise.
		"""
		ret = self._spv.Reduce(pcb.logical_or(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0))) == 1
		return ret

	# in-place, so no return value
	def apply(self, op):
		"""
		ToDo:  write doc
		"""
		#filterPred = pcb.ifthenelse(pred, pcb.identity(), pcb.set(0))
		if not isinstance(op, pcb.UnaryFunction):
			self._spv.Apply(pcb.unary(op))
		else:
			self._spv.Apply(op)
		return

	def copy(self):
		"""
		creates a deep copy of the input argument.
		"""
		ret = SpParVec(-1)
		ret._spv = self._spv.copy()
		return ret

	# in-place, so no return value
	def count(self, op=None):
		"""
		ToDo:  write doc
		"""
		#filterPred = pcb.ifthenelse(pred, pcb.identity(), pcb.set(0))
		ret = self._spv.Reduce(pcb.plus(), pcb.set(1))
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
		ret._spv = self._spv.dense().Find(pcb.bind2nd(pcb.not_equal_to(),0.0))
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
		ret._dpv = self._spv.dense().FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret


	def isBool(self):
		"""
		returns a Boolean scalar denoting whether all elements of the input 
		SpParVec instance are equal to either True (1) or False (0).
		"""
		if self.nnn() == 0:
			return True
		eps = info.eps()
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		return ret

	def max(self):
		"""
		returns the maximum value of the nonnull elements in the SpParVec 
		instance.
		"""
		ret = self._spv.Reduce(pcb.max())
		return ret

	def min(self):
		"""
		returns the minimum value of the nonnull elements in the SpParVec 
		instance.
		"""
		ret = self._spv.Reduce(pcb.min())
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		SpParVec instance.

		Note:  for x a SpParVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, nnz
		"""
		return len(self) - self._spv.getnnz()

	def nnn(self):
		"""
		returns the number of non-nulls (existent entries) in the
		SpParVec instance.
	
		Note:  for x a SpParVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nn, nnz
		"""
		#ret = self._spv.Reduce(pcb.plus(), pcb.set(1))
		# FIX: get Reduce version working
		ret = self._spv.getnee()
		return ret

	def nnz(self):
		"""
		returns the number of non-zero entries in the SpParVec
		instance.

		Note:  for x a SpParVec instance, x.nnz() is always less than or
		equal to x.nnn().

		SEE ALSO:  nn, nnn
		"""
		ret = self._spv.Reduce(pcb.plus(), pcb.ifthenelse(pcb.bind2nd.not_equal_to(),0), pcb.set(1), pcb.set(0))
		return int(ret)

	@staticmethod
	def ones(sz):
		"""
		creates a SpParVec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		ret = SpParVec(-1)
		ret._spv = pcb.pySpParVec.range(sz,0)
		ret.apply(pcb.set(1))
		return ret

	def printAll(self):
		"""
		prints all elements of a SpParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self._spv.printall()
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
		ret._spv = pcb.pySpParVec.range(stop-start,start)
		return ret
	
	op_plus = 1
	op_max  = 2
	def reduce(self, op=op_plus):
		"""
		ToDo:  write doc
		"""
		if op == SpParVec.op_plus:
			realOp = pcb.plus()
		elif op == SpParVec.op_max:
			realOp = pcb.max()
		else:
			raise KeyError, 'invalid operation'
		#filterPred = pcb.ifthenelse(pred, pcb.identity(), pcb.set(0))
		ret = self._spv.Reduce(realOp)
		return ret
	
	#in-place, so no return value
	def set(self, value):
		"""
		sets every non-null value in the SpParVec instance to the second
		argument, in-place.
		"""
		self.apply(pcb.set(value))
		return

	# in-place, so no return value
	def sort(self):
		"""
		sorts the non-null values in the passed SpParVec instance in-place
		in ascending order.

		Input Arguments:
			self:  a SpParVec instance.

		Output Argument:
			None
		"""
		self._spv.Sort()
		return

	def sorted(self):
		"""
		returns a new SpParVec instance with the sorted values (in ascending
		order) from the input SpParVec instance and a SpParVec permutation 
		vector.

		Input Arguments:
			self:  a SpParVec instance.

		Output Argument:
			ret:  a tuple containing as its first element a SpParVec 
			    instance of the same length and same number and position
			    of non-nulls containing the sorted non-null values and
			    as its second element the indices of the sorted values
			    in the input vector.

		See Also:  sort
		"""
		ret1 = self.copy();
		ret2 = SpParVec(-1)
		ret2._spv = ret1._spv.Sort()
		return (ret1, ret2)

	#in-place, so no return value
	def spOnes(self):
		"""
		sets every non-null value in the SpParVec instance to 1, in-place.
		"""
		self.apply(pcb.set(1))
		return

	#in-place, so no return value
	def spRange(self):
		"""
		sets every non-null value in the SpParVec instance to its position
		(offset) in the vector, in-place.
		"""
		self._spv.setNumToInd()

	def sum(self):
		"""
		returns the sum of all the non-null values in the SpParVec instance.
		"""
		if self.nnn() == 0:
			ret = 0
		else:
			ret = self._spv.Reduce(pcb.plus())
		return ret

	#in-place, so no return value
	def toBool(self):
		"""
		converts the input SpParVec instance, in-place, into Boolean
		values (1.0 for True, 0.0 for False) according to whether the
		initial values are nonzero or zero, respectively.
		"""
		self.apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
		return

	def topK(self, k):
		"""
		returns the largest k non-null values in the passed SpParVec instance.

		Input Arguments:
			self:  a SpParVec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a ParVec instance of length k containing the k largest
			    values from the input vector, in ascending order.
		"""
		ret = ParVec(0)
		ret._dpv = self._spv.TopK(k)
		return ret

	def toParVec(self):	
		"""
		converts a SpParVec instance into a ParVec instance of the same
		length with the non-null elements of the SpParVec instance placed 
		in their corresonding positions in the ParVec instance.
		"""
		ret = ParVec(-1)
		ret._dpv = self._spv.dense()
		return ret

	@staticmethod
	def toSpParVec(SPV):
		#if not isinstance(SPV, pcb.pySpParVec):
		if SPV.__class__.__name__ != 'pySpParVec':
			raise TypeError, 'Only accepts pySpParVec instances'
		ret = SpParVec(-1)
		ret._spv = SPV
		return ret
	
def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root()

def version():
	"""
	Return KDT version number, as a string.
	"""
	return "0.1.0"

def revision():
	"""
	Return KDT revision number, as a string.
	"""
	return "r657"
