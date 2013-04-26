import kdt
from ctypes import *

kdt.set_verbosity(kdt.DEBUG)

class point(Structure):
	_fields_ = [("x", c_int),("y", c_int)]
	
	def __repr__(self):
		return "(%d, %d)"%(self.x, self.y)

	@staticmethod
	def get_c():
		return "typedef struct { int x; int y; } point;"
	

p = point(1, 1)
p2 = point(2, 2)
p3 = point(3, 3)

v = kdt.Vec(10, element=p, sparse=True)

#kdt.p("empty v:")
#kdt.p(v)

v[0] = p
v[3] = p2
v[6] = p3

kdt.p("v with elements:")
kdt.p(v)

class yTimes2(kdt.KDTUnaryFunction):
	def __call__(self, p):
		return p

v.apply(yTimes2())

kdt.p("v after apply:")
kdt.p(v)

##############################

class isy2(kdt.KDTUnaryPredicate):
	def __call__(self, p):
		if (p.y == 2):
			return True
		else:
			return False

v.addFilter(isy2())

v2 = v.copy() # this executes the SEJITS predicate, a straight print will be Python-only
kdt.p(v2)
