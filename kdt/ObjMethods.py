import pyCombBLAS as pcb

#FIX:  add some doc here
#NOTE:  ObjX fields do not have all the standard operators (e.g., +=) defined
#	on them, and will give obscure errors if you use them

def defUserCallbacks(objList):
	def __abs__(self):
		self.weight = abs(self.weight)
		return self
	for obj in objList:
		obj.__abs__ = __abs__
	
	def __iadd__(self, other):
		# self must be an Obj;  float.op(Obj) case handled by __radd__
		if isinstance(other, (float, int, long)):
			self.weight += other
		else:
			self.weight += other.weight
		return self
	for obj in objList:
		obj.__iadd__ = __iadd__
	
	def __iand__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = int(self.weight) & int(other)
		else:
			self.weight = int(self.weight) & int(other.weight)
		return self
	for obj in objList:
		obj.__iand__ = __iand__
	
	def __div__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight /= other
		else:
			self.weight /= other.weight
		return self
	for obj in objList:
		obj.__div__ = __div__
	
	def __eq__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = self.weight == other
		else:
			self.weight = self.weight == other.weight
		return self
	for obj in objList:
		obj.__eqPy__ = __eq__
	
	def __ge__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = self.weight >= other
		else:
			self.weight = self.weight >= other.weight
		return self
	for obj in objList:
		obj.__ge__ = __ge__
	
	def __gt__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = self.weight > other
		else:
			self.weight = self.weight > other.weight
		return self
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
		self.weight = ~int(self.weight)
		return self
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
			self.weight = self.weight <= other
		else:
			self.weight = self.weight <= other.weight
		return self
	for obj in objList:
		obj.__le__ = __le__
	
	# HACK:  currently using a different name from __lt__ in the object,
	#	because there's a built-in __lt__ used for sort() that we
	#	should not overwrite, but it only returns a boolean, not an
	#	Obj?.  
	def __lt__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = self.weight < other
		else:
			self.weight = self.weight < other.weight
		return self
	for obj in objList:
		obj.__ltPy__ = __lt__
	
	def __mod__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight %= other
		else:
			self.weight %= other.weight
		return self
	for obj in objList:
		obj.__mod__ = __mod__
	
	def __mul__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight *= other
		else:
			self.weight *= other.weight
		return self
	for obj in objList:
		obj.__mul__ = __mul__
	
	def __ne__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = self.weight != other
		else:
			self.weight = self.weight != other.weight
		return self
	for obj in objList:
		obj.__nePy__ = __ne__
	
	def __neg__(self):
		self.weight = -self.weight
		return self
	for obj in objList:
		obj.__neg__ = __neg__
	
	def __or__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = int(self.weight) | int(other)
		else:
			self.weight = int(self.weight) | int(other.weight)
		return self
	for obj in objList:
		obj.__or__ = __or__
	
	def __radd__(self, other):
		# other must be a float/int/long;  float.op(Obj) case handled here
		if isinstance(other, (float, int, long)):
			other += self.weight
		return other
	for obj in objList:
		obj.__radd__ = __radd__
	
	def __rand__(self, other):
		if isinstance(other, (float, int, long)):
			other = int(self.weight) & int(other)
		return other
	for obj in objList:
		obj.__rand__ = __rand__
	
	def __rsub__(self, other):
		if isinstance(other, (float, int, long)):
			other -= self.weight
		else:
			other.weight = other.weight - self.weight
		return other
	for obj in objList:
		obj.__rsub__ = __rsub__
	
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
		if isinstance(other, (float, int, long)):
			self.weight -= other
		else:
			self.weight = self.weight - other.weight
		return self
	for obj in objList:
		obj.__sub__ = __sub__
	
	def __xor__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = int(self.weight) ^ int(other)
		else:
			self.weight = int(self.weight) ^ int(other.weight)
		return self
	for obj in objList:
		obj.__xor__ = __xor__
	
	def _true_(self, other):
		return True
	for obj in objList:
		obj._true_ = _true_
	
	def all(self, other):
		#print "self=", self, "other=", other
		self.weight = (self.weight!=0) & (other.weight!=0)
		return self
	for obj in objList:
		obj.all = all
	
	def any(self, other):
		#print "self=", self, "other=", other
		self.weight = (self.weight!=0) | (other.weight!=0)
		return self
	for obj in objList:
		obj.any = any
	
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

	def count(x, y):
		# used by DiGraph.nedge, DiGraph.degree, Vec.nnn
		# by definition, x and y are of same type
		# x is the addend;  y is the running sum
		if isinstance(x, (float, int, long)):
			y += 1
		elif x.weight != 0 or x.category != 0:
			y.weight = y.weight + 1
		return y
	for obj in objList:
		obj.count = count

	def objLogicalAnd(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = bool(self.weight) and bool(other)
		else:
			self.weight = bool(self.weight) and bool(other.weight)
		return self
	for obj in objList:
		obj.logicalAnd = objLogicalAnd
	
	def objLogicalOr(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = bool(self.weight) or bool(other)
		else:
			self.weight = bool(self.weight) or bool(other.weight)
		return self
	for obj in objList:
		obj.logicalOr = objLogicalOr
	
	def objLogicalXor(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = (bool(self.weight) or bool(other)) - (bool(self.weight) and bool(other))
		else:
			self.weight = (bool(self.weight) or bool(other.weight)) - (bool(self.weight) and bool(other.weight))
		return self
	for obj in objList:
		obj.logicalXor = objLogicalXor
	
	def objMax(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = max(self.weight, other)
		else:
			self.weight = max(self.weight, other.weight)
		return self
	for obj in objList:
		obj.max = objMax
	
	def objMin(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = min(self.weight, other)
		else:
			self.weight = min(self.weight, other.weight)
		return self
	for obj in objList:
		obj.min = objMin
	
	def ones(self):
		if isinstance(self, (float, int, long)):
			self = 1
		else:
			self.weight = 1
		return self
	for obj in objList:
		obj.ones = ones
	
	def prune(self):
		if isinstance(self, (pcb.Obj1)):
			return self.weight==0 and self.category==0
		elif isinstance(self, (pcb.Obj2)):
			return self.weight==0 and self.category==0
		else:
			raise NotImplementedError
	for obj in objList:
		obj.prune = prune

	def spOnes(self):
		if isinstance(self, (float, int, long)):
			self = 1
		else:
			self.weight = 1
		return self
	for obj in objList:
		obj.spOnes = spOnes
	
	def spRange(self, other):
	# only called when self is an Obj
		self.weight = other
		return self
	for obj in objList:
		obj.spRange = spRange

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
