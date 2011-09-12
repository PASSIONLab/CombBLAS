#import pyCombBLAS as pcb

def defUserCallbacks(objList):
	def __abs__(self):
		self.weight = abs(self.weight)
		return self
	for obj in objList:
		obj.__abs__ = __abs__
	
	def __add__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight += other
		else:
			self.weight += other.weight
		return self
	for obj in objList:
		obj.__add__ = __add__
	
	def __and__(self, other):
		if isinstance(other, (float, int, long)):
			self.weight = int(self.weight) & int(other)
		else:
			self.weight = int(self.weight) & int(other.weight)
		return self
	for obj in objList:
		obj.__and__ = __and__
	
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
			self.weight = bool(self.weight and other)
		else:
			self.weight = bool(self.weight and other.weight)
		return self
	for obj in objList:
		obj.logical_and = objLogicalAnd	# for NumPy compatibility
		obj.logicalAnd = objLogicalAnd
	
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
	
