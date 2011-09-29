#import pyCombBLAS as pcb

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
			self.weight -= other.weight
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
	
	def __setitem__(self, key, value):
		if key is 'weight':
			self.weight = value
		elif key is 'type':
			self.type = value
		else:
			raise KeyError
		return self
	for obj in objList:
		obj.__setitem__ = __setitem__
	
	def __sub__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight -= other
		else:
			self.weight -= other.weight
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
		self.weight = max(self.weight, other.weight)
		return self
	for obj in objList:
		obj.max = objMax
	
	def objMin(self, other):
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
	
	def spOnes(self):
		if isinstance(self, (float, int, long)):
			self = 1
		else:
			self.weight = 1
		return self
	for obj in objList:
		obj.spOnes = spOnes
	
	
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
