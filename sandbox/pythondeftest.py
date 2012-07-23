import kdt
from ctypes import *

class point(Structure):
	_fields_ = [("x", c_int),("y", c_int)]
	
	def __repr__(self):
		return "(%d, %d)"%(self.x, self.y)

p = point(1, 1)
p2 = point(2, 2)
p3 = point(3, 3)

v = kdt.Vec(10, element=p, sparse=False)

print "empty v:"
print v

v[0] = p
v[3] = p2
v[6] = p3

print "v with elements:"
print v

def yTimes2(p):
	p.y *= 2
	return p
	
v.apply(yTimes2)

print "v after apply:"
print v

print "find, findInds for p.x != 1:"
print v.find(lambda p: p.x != 1)
print v.findInds(lambda p: p.x != 1)

v2 = kdt.Vec(10, element=p)
v2[0] = p
v2[2] = p3
v2[6] = p2

print "v2:"
print v2

def pplus(e1, e2):
	ret = point(e1.x+e2.x, e1.y+e2.y)
	#print "got:",e1,e2,"returning:",ret
	return ret

v3 = v.eWiseApply(v2, op=pplus, inPlace=False)

print "result of eWiseApply:"
print v3

def appIndFunc(p, i):
	p.y = int(i)
	return p
v2.applyInd(appIndFunc)
print "v2 after applyInd:"
print v2

# Vec count is a bit broken
#print v.count(lambda p: p.x != 1)

v_red = v.reduce(pplus, init=point(0,100000))
v2_red = v2.reduce(pplus)
print "v plus reduction with init:",v_red,"   v2 plus reduction:",v2_red

##############################
# matrix

i = kdt.Vec(4, sparse=False)
i[0] = 2
i[1] = 4
i[2] = 0
i[3] = 1

j = kdt.Vec(4, sparse=False)
j[0] = 3
j[1] = 2
j[2] = 1
j[3] = 3

v = kdt.Vec(4, element=p, sparse=False)
v[0] = point(0,0)
v[1] = point(1,1)
v[2] = point(2,2)
v[3] = point(3,3)


m = kdt.Mat(i, j, v, 5, 5)

print m

m_red = m.reduce(kdt.Mat.Column, pplus)
print "column reduction:",m_red

yrange = m_red
yrange.applyInd(appIndFunc)
m.scale(yrange, lambda me, ve: ve, kdt.Mat.Column)
print "after scale:"
print m


def srm(a, b):
	return point(a.x*b.x, a.y*b.y)
def sra(a, b):
	return point(a.x+b.x, a.y+b.y)

sr = kdt.sr(sra, srm)

print "yrange:"
print yrange
yrange = yrange.sparse()
print "spmv vector:"
print yrange
mvret = m.SpMV(yrange, sr)
print "spmv result:"
print mvret