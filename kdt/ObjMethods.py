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
		ret.setInt(abs(self.weight))
		return ret
	for obj in objList:
		obj.__abs__ = __abs__
	
	def __add__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(self.weight + other)
		else:
			ret.setInt(self.weight + other.weight)
		#print "self:",self," other:",other,"returning:",ret
		return ret
	for obj in objList:
		obj.__add__ = __add__
	
	def __and__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(int(self.weight) & int(other))
		else:
			ret.setInt(int(self.weight) & int(other.weight))
		return ret
	for obj in objList:
		obj.__and__ = __and__
	
	def __div__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(self.weight/ other)
		else:
			ret.setInt(self.weight / other.weight)
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
		ret.setInt(~int(self.weight))
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
			ret.setInt(self.weight % other)
		else:
			ret.setInt(self.weight % other.weight)
		return ret
	for obj in objList:
		obj.__mod__ = __mod__
	
	def __mul__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(self.weight * other)
		else:
			ret.setInt(self.weight * other.weight)
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
		ret.setInt(-self.weight)
		return ret
	for obj in objList:
		obj.__neg__ = __neg__
	
	def __or__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(int(self.weight) | int(other))
		else:
			ret.setInt(int(self.weight) | int(other.weight))
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
			ret.setInt(self.weight - other)
		else:
			ret.setInt(self.weight - other.weight)
		return ret
	for obj in objList:
		obj.__sub__ = __sub__
	
	def __xor__(self, other):
		ret = self.__copy__()
		if isinstance(other, (float, int, long)):
			ret.setInt(int(self.weight) ^ int(other))
		else:
			ret.setInt(int(self.weight) ^ int(other.weight))
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
		ret.setInt((self.weight!=0) & (other.weight!=0))
		return ret
	for obj in objList:
		obj.all = all
	
	def any(self, other):
		ret = self.__copy__()
		#print "self=", self, "other=", other
		ret.setInt((self.weight!=0) | (other.weight!=0))
		return ret
	for obj in objList:
		obj.any = any
	
	def set(self, val):
		if isinstance(val, (float, int, long)):
			self.setInt(val)
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
					self = other.latest
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj1):
				if isinstance(other, (float, int, long)):
					self.weight = other
				elif isinstance(other, pcb.Obj1):
					self.weight = other.weight
				elif isinstance(other, pcb.Obj2):
					self.weight = other.latest
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj2):
				if isinstance(other, (float, int, long)):
					self.latest = other
				elif isinstance(other, pcb.Obj1):
					self.latest = other.weight
				elif isinstance(other, pcb.Obj2):
					self.latest = other.latest
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
					other.latest = self
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj1):
				if isinstance(other, (float, int, long)):
					other = self.weight
				elif isinstance(other, pcb.Obj1):
					other.weight = self.weight
				elif isinstance(other, pcb.Obj2):
					other.weight = self.latest
				else:
					raise NotImplementedError
			elif isinstance(self, pcb.Obj2):
				if isinstance(other, (float, int, long)):
					other = self.latest
				elif isinstance(other, pcb.Obj1):
					other.weight = self.latest
				elif isinstance(other, pcb.Obj2):
					other.latest = self.latest
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
			return other	
	for obj in objList:
		obj.coerce = coerce

# a helper function to help navigate the issue of different structures between Obj1 and Obj2
#	def getInt(self):
#		if isinstance(self, pcb.Obj1):
#			return self.weight
#		elif isinstance(self, pcb.Obj2):
#			return self.latest
#		else:
#			raise NotImplementedError
#	for obj in objList:
#		obj.getInt = getInt
	def setInt(self, value):
		if isinstance(self, pcb.Obj1):
			self.weight = value
		elif isinstance(self, pcb.Obj2):
			self.latest = value
		else:
			raise NotImplementedError
	for obj in objList:
		obj.setInt = setInt

# provide a fake 'weight' attribute to keep compatibility with Obj1 for tests.
	def getweight(self):
		#print "getting"
		return self.latest
	def setweight(self, value):
		#print "setting to",value
		self.latest = value
    
	pcb.Obj2.weight = property(getweight, setweight)

# provide a fake 'category' attribute to keep compatibility with Obj1 for tests.
	def getcat(self):
		#print "getting"
		return self.count
	def setcat(self, value):
		#print "setting to",value
		self.count = value
    
	pcb.Obj2.category = property(getcat, setcat)

#
#	Methods below here are used by KDT unittests
#
	@staticmethod
	def maxInitObj1():
		ret = pcb.Obj1()
		ret.setInt(-1.8e308)
		return ret
	@staticmethod
	def maxInitObj2():
		ret = pcb.Obj2()
		ret.setInt(-1.8e308)
		return ret
	for obj in objList:
		if obj == pcb.Obj1:
			obj.maxInit = maxInitObj1
		elif obj == pcb.Obj2:
			obj.maxInit = maxInitObj2

	@staticmethod
	def minInitObj1():
		ret = pcb.Obj1()
		ret.setInt(1.8e308)
		return ret
	@staticmethod
	def minInitObj2():
		ret = pcb.Obj2()
		ret.setInt(1.8e308)
		return ret
	for obj in objList:
		if obj == pcb.Obj1:
			obj.minInit = minInitObj1
		elif obj == pcb.Obj2:
			obj.minInit = minInitObj2
	