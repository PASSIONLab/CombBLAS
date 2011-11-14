import pyCombBLAS as pcb

#FIX:  add some doc here
#NOTE:  ObjX fields do not have all the standard operators (e.g., +=) defined
#	on them, and will give obscure errors if you use them

def defUserCallbacks(objList):
	def __copy__(self):
		if isinstance(self, pcb.Obj1):
			ret = pcb.Obj1(self)
		elif isinstance(self, pcb.Obj2):
			ret = pcb.Obj2(self)
		else:
			raise NotImplementedError,"unknown type in __copy__!"
		return ret
	for obj in objList:
		obj.__copy__ = __copy__

	def __abs__(self):
		ret = self.__copy__()
		ret.weight = abs(self.weight)
		return ret
	for obj in objList:
		obj.__abs__ = __abs__
	
	def __add__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = self.weight + other
		else:
			ret.weight = self.weight + other.weight
		return ret
	for obj in objList:
		obj.__add__ = __add__
	
	def __and__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = int(self.weight) & int(other)
		else:
			ret.weight = int(self.weight) & int(other.weight)
		return ret
	for obj in objList:
		obj.__and__ = __and__
	
	def __div__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = self.weight/ other
		else:
			ret.weight = self.weight / other.weight
		return ret
	for obj in objList:
		obj.__div__ = __div__
	
	def __eq__(self, other):
		if isinstance(other, (float, int, long)):
			return self.weight == other
		else:
			return self.weight == other.weight
	for obj in objList:
		obj.__eqPy__ = __eq__
	
	def __ge__(self, other):
		if isinstance(other, (float, int, long)):
			return self.weight >= other
		else:
			return self.weight >= other.weight
	for obj in objList:
		obj.__ge__ = __ge__
	
	def __gt__(self, other):
		if isinstance(other, (float, int, long)):
			return self.weight > other
		else:
			return self.weight > other.weight
	for obj in objList:
		obj.__gt__ = __gt__

	def __iadd__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight += other
		else:
			self.weight += other.weight
		return self
	for obj in objList:
		obj.__iadd__ = __iadd__
	
	def __invert__(self):
		ret = self.__copy__()
		ret.weight = ~int(self.weight)
		return ret
	for obj in objList:
		obj.__invert__ = __invert__
	
	def __isub__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight -= other
		else:
			self.weight = self.weight - other.weight
		return self
	for obj in objList:
		obj.__isub__ = __isub__
	
	def __le__(self, other):
		if isinstance(other, (float, int, long)):
			return self.weight <= other
		else:
			return self.weight <= other.weight
	for obj in objList:
		obj.__le__ = __le__

# The less-than operation is defined in C++ in kdt/pyCombBLAS/obj.h so that it is available to
# sort. It is not currently overrideable in Python, but will be in the future.
#	def __lt__(self, other):
#		if isinstance(other, (float, int, long)):
#			return self.weight < other
#		else:
#			return self.weight < other.weight
#	for obj in objList:
#		obj.__ltPy__ = __lt__
	
	def __mod__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = self.weight % other
		else:
			ret.weight = self.weight % other.weight
		return ret
	for obj in objList:
		obj.__mod__ = __mod__
	
	def __mul__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = self.weight * other
		else:
			ret.weight = self.weight * other.weight
		return ret
	for obj in objList:
		obj.__mul__ = __mul__
	
	def __ne__(self, other):
		if isinstance(other, (float, int, long)):
			return self.weight != other
		else:
			return self.weight != other.weight
	for obj in objList:
		obj.__nePy__ = __ne__
	
	def __neg__(self):
		ret = self.__copy__()
		ret.weight = -self.weight
		return ret
	for obj in objList:
		obj.__neg__ = __neg__
	
	def __or__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = int(self.weight) | int(other)
		else:
			ret.weight = int(self.weight) | int(other.weight)
		return ret
	for obj in objList:
		obj.__or__ = __or__
	
	def __radd__(self, other):
		# other must be a float/int/long;  float.op(Obj) case handled here
		if isinstance(other, (float, int, long)):
			return other + self.weight
		else:
			raise NotImplementedError
	for obj in objList:
		obj.__radd__ = __radd__
	
	def __rand__(self, other):
		if isinstance(other, (float, int, long)):
			return int(self.weight) & int(other)
		else:
			raise NotImplementedError
	for obj in objList:
		obj.__rand__ = __rand__
	
	def __rsub__(self, other):
		if isinstance(other, (float, int, long)):
			return other - self.weight
		else:
			raise NotImplementedError
	for obj in objList:
		obj.__rsub__ = __rsub__
	
	def truth(self, other):
		return bool(self.weight)
	for obj in objList:
		obj.truth = truth

	def __setitem__(self, key, value):
		if key is 'weight':
			self.weight = value
		elif key is 'category':
			self.category = value
		else:
			raise KeyError
		return self
	for obj in objList:
		obj.__setitem__ = __setitem__
	
	def __sub__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = self.weight - other
		else:
			ret.weight = self.weight - other.weight
		return ret
	for obj in objList:
		obj.__sub__ = __sub__
	
	def __xor__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.weight = int(self.weight) ^ int(other)
		else:
			ret.weight = int(self.weight) ^ int(other.weight)
		return ret
	for obj in objList:
		obj.__xor__ = __xor__
	
#	def _true_(self, other):
#		return True
#	for obj in objList:
#		obj._true_ = _true_
	
	def all(self, other):
		#print "self=", self, "other=", other
		ret = self.__copy__()
		ret.weight = (self.weight!=0) & (other.weight!=0)
		return ret
	for obj in objList:
		obj.all = all
	
	def any(self, other):
		ret = self.__copy__()
		#print "self=", self, "other=", other
		ret.weight = (self.weight!=0) | (other.weight!=0)
		return ret
	for obj in objList:
		obj.any = any
	
	def set(self, val):
		if isinstance(val, (float, int, long)):
			self.weight = val
		else:
			raise NotImplementedError,"operator="
		return self
	for obj in objList:
		obj.set = set

	def coerce(self, other, typeFirst):
		# creates a copy of one arg of the type of the other arg;
		# typeFirst flag, if True, says the type should be from the
		# first arg and the value from the second arg
		if typeFirst:
			if isinstance(self, (float, int, long)):
				if isinstance(other, (float, int, long)):
					self = other
				elif isinstance(other, pcb.Obj1):
					self = other.weight
				elif isinstance(other, pcb.Obj2):
					self = other.weight
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj1):
				if isinstance(other, (float, int, long)):
					self.weight = other
				elif isinstance(other, pcb.Obj1):
					self.weight = other.weight
				elif isinstance(other, pcb.Obj2):
					self.weight = other.weight
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj2):
				if isinstance(other, (float, int, long)):
					self.weight = other
				elif isinstance(other, pcb.Obj1):
					self.weight = other.weight
				elif isinstance(other, pcb.Obj2):
					self.weight = other.weight
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
			return self	
		else:
			if isinstance(self, (float, int, long)):
				if isinstance(other, (float, int, long)):
					other = self
				elif isinstance(other, pcb.Obj1):
					other.weight = self
				elif isinstance(other, pcb.Obj2):
					other.weight = self
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj1):
				if isinstance(other, (float, int, long)):
					other = self.weight
				elif isinstance(other, pcb.Obj1):
					other.weight = self.weight
				elif isinstance(other, pcb.Obj2):
					other.weight = self.weight
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj2):
				if isinstance(other, (float, int, long)):
					other = self.weight
				elif isinstance(other, pcb.Obj1):
					other.weight = self.weight
				elif isinstance(other, pcb.Obj2):
					other.weight = self.weight
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
			return other	
	for obj in objList:
		obj.coerce = coerce

#	def objLogicalAnd(self, other):
#		if isinstance(other, (float, int, long)):
#			self.weight = bool(self.weight) and bool(other)
#		else:
#			self.weight = bool(self.weight) and bool(other.weight)
#		return self
#	for obj in objList:
#		obj.logicalAnd = objLogicalAnd
	
#	def objLogicalOr(self, other):
#		if isinstance(other, (float, int, long)):
#			self.weight = bool(self.weight) or bool(other)
#		else:
#			self.weight = bool(self.weight) or bool(other.weight)
#		return self
#	for obj in objList:
#		obj.logicalOr = objLogicalOr
	
#	def objLogicalXor(self, other):
#		if isinstance(other, (float, int, long)):
#			self.weight = (bool(self.weight) or bool(other)) - (bool(self.weight) and bool(other))
#		else:
#			self.weight = (bool(self.weight) or bool(other.weight)) - (bool(self.weight) and bool(other.weight))
#		return self
#	for obj in objList:
#		obj.logicalXor = objLogicalXor
	
#	def objMax(self, other):
#		if isinstance(other, (float, int, long)):
#			self.weight = max(self.weight, other)
#		else:
#			self.weight = max(self.weight, other.weight)
#		return self
#	for obj in objList:
#		obj.max = objMax
	
#	def objMin(self, other):
#		if isinstance(other, (float, int, long)):
#			self.weight = min(self.weight, other)
#		else:
#			self.weight = min(self.weight, other.weight)
#		return self
#	for obj in objList:
#		obj.min = objMin
	
#	def ones(self):
#		if isinstance(self, (float, int, long)):
#			self = 1
#		else:
#			self.weight = 1
#		return self
#	for obj in objList:
#		obj.ones = ones
	
#	def prune(self):
#		if isinstance(self, (pcb.Obj1)):
#			return self.weight==0 and self.category==0
#		elif isinstance(self, (pcb.Obj2)):
#			return self.weight==0 and self.category==0
#		else:
#			raise NotImplementedError
#	for obj in objList:
#		obj.prune = prune

#	def spOnes(self):
#		if isinstance(self, (float, int, long)):
#			self = 1
#		else:
#			self.weight = 1
#		return self
#	for obj in objList:
#		obj.spOnes = spOnes

#	def spZeros(self):
#		if isinstance(self, (float, int, long)):
#			self = 0
#		else:
#			self.weight = 0
#		return self
#	for obj in objList:
#		obj.spZeros = spZeros
	
#	def spRange(self, other):
#	# only called when self is an Obj
#		self.weight = other
#		return self
#	for obj in objList:
#		obj.spRange = spRange

#----------------- methods for Semiring use-------------------
	def _SR_mul_(self, other):
		if isinstance(self, (float, int, long)):
			if isinstance(other, (float, int, long)):
				self *= other
			elif isinstance(other, pcb.Obj1):
				self *= other.weight
			elif isinstance(other, pcb.Obj2):
				self *= other.weight
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj1):
			if isinstance(other, (float, int, long)):
				self.weight *= other
			elif isinstance(other, pcb.Obj1):
				self.weight *= other.weight
			elif isinstance(other, pcb.Obj2):
				self.weight *= other.weight
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj2):
			if isinstance(other, (float, int, long)):
				self.weight *= other
			elif isinstance(other, pcb.Obj1):
				self.weight *= other.weight
			elif isinstance(other, pcb.Obj2):
				self.weight *= other.weight
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError
		return self	
	for obj in objList:
		obj._SR_mul_ = _SR_mul_

	def _SR_second_(self, other):
		if isinstance(self, (float, int, long)):
			if isinstance(other, (float, int, long)):
				self = other
			elif isinstance(other, pcb.Obj1):
				self = other.weight
			elif isinstance(other, pcb.Obj2):
				self = other.weight
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj1):
			if isinstance(other, (float, int, long)):
				self.weight = other
			elif isinstance(other, pcb.Obj1):
				self.weight = other.weight
			elif isinstance(other, pcb.Obj2):
				self.weight = other.weight
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj2):
			if isinstance(other, (float, int, long)):
				self.weight = other
			elif isinstance(other, pcb.Obj1):
				self.weight = other.weight
			elif isinstance(other, pcb.Obj2):
				self.weight = other.weight
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError
		return self	
	for obj in objList:
		obj._SR_second_ = _SR_second_

	def _SR_max_(self, other):
		if isinstance(self, (float, int, long)):
			if isinstance(other, (float, int, long)):
				self = max(self, other)
			elif isinstance(other, pcb.Obj1):
				self = max(self, other.weight)
			elif isinstance(other, pcb.Obj2):
				self = max(self, other.weight)
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj1):
			if isinstance(other, (float, int, long)):
				self.weight = max(self.weight, other)
			elif isinstance(other, pcb.Obj1):
				self.weight = max(self.weight, other.weight)
			elif isinstance(other, pcb.Obj2):
				self.weight = max(self.weight, other.weight)
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj2):
			if isinstance(other, (float, int, long)):
				self.weight = max(self.weight, other)
			elif isinstance(other, pcb.Obj1):
				self.weight = max(self.weight, other.weight)
			elif isinstance(other, pcb.Obj2):
				self.weight = max(self.weight, other.weight)
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError
		return self	
	for obj in objList:
		obj._SR_max_ = _SR_max_

	def _SR_min_(self, other):
		if isinstance(self, (float, int, long)):
			if isinstance(other, (float, int, long)):
				self = min(self, other)
			elif isinstance(other, pcb.Obj1):
				self = min(self, other.weight)
			elif isinstance(other, pcb.Obj2):
				self = min(self, other.weight)
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj1):
			if isinstance(other, (float, int, long)):
				self.weight = min(self.weight, other)
			elif isinstance(other, pcb.Obj1):
				self.weight = min(self.weight, other.weight)
			elif isinstance(other, pcb.Obj2):
				self.weight = min(self.weight, other.weight)
			else:
				raise NotImplementedError
		elif isinstance(self, pcb.Obj2):
			if isinstance(other, (float, int, long)):
				self.weight = min(self.weight, other)
			elif isinstance(other, pcb.Obj1):
				self.weight = min(self.weight, other.weight)
			elif isinstance(other, pcb.Obj2):
				self.weight = min(self.weight, other.weight)
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError
		return self	
	for obj in objList:
		obj._SR_min_ = _SR_min_

	
#
#	Methods below here are used by KDT unittests
#
	@staticmethod
	def maxInitObj1():
		ret = pcb.Obj1()
		ret.weight = -1.8e308
		return ret
	@staticmethod
	def maxInitObj2():
		ret = pcb.Obj2()
		ret.weight = -1.8e308
		return ret
	for obj in objList:
		if obj == pcb.Obj1:
			obj.maxInit = maxInitObj1
		elif obj == pcb.Obj2:
			obj.maxInit = maxInitObj2

	@staticmethod
	def minInitObj1():
		ret = pcb.Obj1()
		ret.weight = 1.8e308
		return ret
	@staticmethod
	def minInitObj2():
		ret = pcb.Obj2()
		ret.weight = 1.8e308
		return ret
	for obj in objList:
		if obj == pcb.Obj1:
			obj.minInit = minInitObj1
		elif obj == pcb.Obj2:
			obj.minInit = minInitObj2
	
		
	@staticmethod
	def ge0lt5(self):
		if isinstance(self, (float, int, long)):
			return self >= 0 and self < 5
		else:
			return self.weight >= 0 and self.weight < 5
	for obj in objList:
		obj.ge0lt5 = ge0lt5

	@staticmethod
	def geM2lt4(self):
		if isinstance(self, (float, int, long)):
			return self >= -2 and self < 4
		else:
			return self.weight >= -2 and self.weight < 4
	for obj in objList:
		obj.geM2lt4 = geM2lt4
