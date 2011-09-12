import unittest
import math
#from kdt import *
from SpVecObj import *
from kdt import pyCombBLAS as pcb
import numpy as np

class SpVecTests(unittest.TestCase):
    def initializeSpVecObj(self, length, i, v=1):
        """
        Initialize a ParVec instance with values equal to one or the input value.
        """
	if type(v) is tuple:
		vTuple = True
        ret = SpVecObj(length)
        for ind in range(len(i)):
	    obj = pcb.Obj1()
	    if type(v) != int and type(v) != float:
	        obj.weight = v[0][ind]
	        obj.type   = v[1][ind]
            else:
                obj.weight = v
                obj.type   = v
	    ret[i[ind]] = obj

        return ret
 
    def initializeParVec(self, length, i, v=1):
        """
        Initialize a ParVec instance with values equal to one or the input value.
        """
        ret = ParVec(length, 0)
        for ind in range(len(i)):
	    if type(v) != int and type(v) != float:
		ret[i[ind]] = v[ind]
            else:
                ret[i[ind]] = v

        return ret

class ConstructorTests(SpVecTests):
    def test_SpVec_simple(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeSpVecObj(sz, i)
        expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(len(i)):
            self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
            self.assertEqual(expI[i[ind]],vec[i[ind]].type)
        for ind in range(sz):
            self.assertAlmostEqual(expI[ind],vec[ind].weight)

    def test_SpVec_null(self):
	sz = 25
        i = []
        vec = self.initializeSpVecObj(sz, i)
        expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(len(i)):
            self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
            self.assertEqual(expI[i[ind]],vec[i[ind]].type)
        for ind in range(sz):
            self.assertAlmostEqual(expI[ind],vec[ind].weight)

    def test_SpVec_object(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, 4, 16, 36, 64, 100]
	type = [2, 2, 2, 5, 5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
        expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
        expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(len(i)):
            self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
            self.assertEqual(expT[i[ind]],vec[i[ind]].type)
        for ind in range(sz):
            self.assertAlmostEqual(expW[ind],vec[ind].weight)

    def test_SpVec_object_all_weight_zero(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, 0, 0, 0, 0, 0]
	type =   [2, 2, 2, 5, 5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
        expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
        expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(len(i)):
            self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
            self.assertEqual(expT[i[ind]],vec[i[ind]].type)
        for ind in range(sz):
            self.assertAlmostEqual(expW[ind],vec[ind].weight)

    def test_SpVec_object_all_weight_all_type_zero(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, 0, 0, 0, 0, 0]
	type =   [0, 0, 0, 0, 0, 0]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
        expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
        expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(len(i)):
            self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
            self.assertEqual(expT[i[ind]],vec[i[ind]].type)
        for ind in range(sz):
            self.assertAlmostEqual(expW[ind],vec[ind].weight)

#    def test_SpVec_zeros(self):
#	sz = 25
#        vec = SpVec(sz)
#        expI = 0
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(expI,vec[ind])
#
#    def test_SpVec_ones(self):
#	sz = 25
#        vec = SpVec.ones(sz)
#        expI = 1
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(expI,vec[ind])
#
#    def test_SpVec_range_simple(self):
#	sz = 25
#        vec = SpVec.range(sz)
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(ind,vec[ind])
#
#    def test_SpVec_range_offset(self):
#	sz = 25
#	offset = -13
#        vec = SpVec.range(offset, sz+offset)
#        expI = 1
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(ind,vec[ind]-offset)
#
#    def test_SpVec_set37(self):
#	sz = 25
#        vec = SpVec.ones(sz)
#        i = [0, 2,  4,  6,  8, 10]
#        v = [4, 8, 12, 16, 20, 24]
#        vec = self.initializeSpVec(sz, i, v)
#	scalar = 37
#	vec.set(scalar)
#	expVec = [37, 0, 37, 0, 37, 0, 37, 0, 37, 0, 37, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(expVec[ind],vec[ind])
#
#    def test_SpVec_spOnes(self):
#	sz = 25
#        i = [0, 2,  4,  6,  8, 10]
#        v = [4, 8, 12, 16, 20, 24]
#        vec = self.initializeSpVec(sz, i, v)
#        vec.spOnes()
#	expVec = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(expVec[ind],vec[ind])
#
#    def test_SpVec_toBool(self):
#	sz = 25
#        i = [0, 2,  4,  6,  8, 10]
#        v = [4, 8, 12, 16, 20, 24]
#        vec = self.initializeSpVec(sz, i, v)
#        vec.toBool()
#	expVec = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec))
#        for ind in range(sz):
#            self.assertEqual(expVec[ind],vec[ind])

class BuiltInTests(SpVecTests):
    def test_len(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = len(vec)
	self.assertEqual(sz, res)

    def test_nnn(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [777, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.nnn()
	expRes = 6
	self.assertEqual(expRes, res)

    def test_nnn_no_elem0(self):
	sz = 25
        i = [1, 2, 4, 6, 8, 10]
	weight = [777, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.nnn()
	expRes = 6
	self.assertEqual(expRes, res)

    def test_add_scalar(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
        expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
        expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	res = vec + 3.07
	#self.assertEqual(len(i), vec2.nnn())
	self.assertEqual(sz, len(res))
        for ind in range(sz):
            self.assertAlmostEqual(expW[ind],res[ind].weight)
            #self.assertEqual(expT[ind],res[ind].type)

    def test_add_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 216, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
        vec3 = vec1 + vec2
	#print "vec1=", vec1
	#print "vec2=", vec2
	#print "vec3=", vec3
        expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
		0, 0, 5832, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind].weight)

#    def test_iadd_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10]
#        vec = self.initializeSpVec(sz, i)
#        vec += 3.07
#        expI = [4.07, 0, 4.07, 0, 4.07, 0, 4.07, 0, 4.07, 0, 
#		4.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#		0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec))
#	self.assertEqual(len(i), vec.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec[ind])
#
#    def test_iadd_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10]
#        v1 = [0, 4, 16, 36, 64, 100]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 3, 6, 9, 12, 15, 18]
#        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#        vec1 += vec2
#        expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
#		0, 0, 5832, 0, 0, 0, 0, 0, 0,]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_sub_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10]
#        vec = self.initializeSpVec(sz, i)
#        vec2 = vec - 3.07
#        expI = [-2.07, 0, -2.07, 0, -2.07, 0, -2.07, 0, -2.07,
#		0, -2.07, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec2))
#	self.assertEqual(vec.nnn(), len(i))
#	self.assertEqual(vec.nnn(), vec2.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec2[ind])
#
#    def test_sub_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10]
#        v1 = [0, 4, 16, 36, 64, 100]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 3, 6, 9, 12, 15, 18]
#        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#        vec3 = vec1 - vec2
#        expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0,
#		-3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
#	self.assertEqual(sz, len(vec3))
#	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_isub_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10]
#        vec = self.initializeSpVec(sz, i)
#        vec -= 3.07
#        expI = [-2.07, 0, -2.07, 0, -2.07, 0, -2.07, 0, -2.07,
#		0, -2.07, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec))
#	self.assertEqual(vec.nnn(), len(i))
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec[ind])
#
#    def test_isub_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10]
#        v1 = [0, 4, 16, 36, 64, 100]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 3, 6, 9, 12, 15, 18]
#        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#        vec1 -= vec2
#        expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0,
#		-3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_negate(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10]
#        v1 = [0, 4, 16, 36, 64, 100]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        vec3 = -vec1
#        expI = [0, 0, -4, 0, -16, 0, -36, 0, -64, 0, -100, 0, 0, 0, 0,
#		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
#	self.assertEqual(sz, len(vec3))
#	self.assertEqual(vec1.nnn(),vec3.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_mul_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10]
#        vec = self.initializeSpVec(sz, i)
#        vec2 = vec * 3.07
#        expI = [3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 
#		3.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#		0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec2))
#	self.assertEqual(vec.nnn(), vec2.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec2[ind])
#
#    def test_mul_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10, 12]
#        v1 = [1, 4, 16, 36, 64, 100, 144]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#	vec2 = ParVec(sz)
#	vec2[0] = 1
#	vec2[3] = 3
#	vec2[6] = 6
#	vec2[9] = 9
#	vec2[12] = 12
#	vec2[15] = 15
#	vec2[18] = 18
#        vec3 = vec1 * vec2
#        expI = [1, 0, 0, 0, 0, 0, 216, 0, 0, 0, 0, 0, 1728, 0, 0, 0, 0, 0, 0, 0,
#		0, 0, 0, 0, 0,]
#	self.assertEqual(sz, len(vec3))
#	self.assertEqual(vec3.nnn(), 3)
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_div_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10, 12]
#        v1 = [1, 4, 16, 36, 64, 100, 144]
#        vec = self.initializeSpVec(sz, i, v1)
#        vec2 = vec / .50
#        expI = [2, 0, 8, 0, 32, 0, 72, 0, 128, 0, 
#		200, 0, 288, 0, 0, 0, 0, 0, 0, 
#		0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec2))
#	self.assertEqual(vec.nnn(), vec2.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec2[ind])
#
#    def test_div_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10, 12]
#        v1 = [1, 4, 16, 36, 64, 100, 144]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#		19, 20, 21, 22, 23, 24]
#        v2 = [1, 1, 1, 3, 1, 1, 6, 1, 1, 9, 1, 1, 12, 1, 1, 15, 
#		1, 1, 18, 1, 1, 1, 1, 1, 1]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#	self.assertRaises(NotImplementedError,SpVec.__div__,vec1,vec2)
#        #expI = [1, 0, 4, 0, 16, 0, 6, 0, 64, 0, 100, 0, 12, 0, 0, 0, 0, 0, 0, 0,
#	#	0, 0, 0, 0, 0,]
#	#self.assertEqual(sz, len(vec3))
#	#self.assertEqual(vec.nnn(), vec3.nnn())
#        #for ind in range(sz):
#	#    self.assertAlmostEqual(expI[ind], vec3[ind])
#
#    def test_mod_constant(self):
#	sz = 25
#        i = [0, 2, 4, 6, 8, 10, 12]
#        v1 = [1, 4, 16, 36, 64, 100, 144]
#        vec = self.initializeSpVec(sz, i, v1)
#        vec2 = vec % 5
#        expI = [1, 0, 4, 0, 1, 0, 1, 0, 4, 0, 
#		0, 0, 4, 0, 0, 0, 0, 0, 0, 
#		0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec2))
#	self.assertEqual(vec.nnn(), vec2.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec2[ind])
#
#    def test_mod_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10, 12]
#        v1 = [1, 4, 16, 36, 64, 100, 144]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#		19, 20, 21, 22, 23, 24]
#        v2 = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 
#		3, 2, 3, 2, 3]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#	self.assertRaises(NotImplementedError,SpVec.__mod__,vec1,vec2)
#        #expI = [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#	#	0, 0, 0, 0, 0,]
#	#self.assertEqual(sz, len(vec3))
#	#self.assertEqual(vec.nnn(), vec3.nnn())
#        #for ind in range(sz):
#	#    self.assertAlmostEqual(expI[ind], vec3[ind])
#
    def test_bitwise_and_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 61, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 217, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
        vec3 = vec1 & vec2
        expI = [0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind].weight)

    def test_logical_and_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 37, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 217, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
        vec3 = vec1.logical_and(vec2)
        expI = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind].weight)

#
#    def test_xor_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
#        v1 = [0, 1, 1, 1, 1,  1,  1,  1,  1,  1]
#        vec1 = self.initializeSpVec(sz, i1, v1)
#        i2 = [0, 3, 6, 9, 12, 15, 18]
#        v2 = [1, 1, 1, 1,  1,  1,  1]
#        vec2 = self.initializeSpVec(sz, i2, v2)
#        vec3 = vec1 ^ vec2
#        expI = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
#		0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec3))
#	self.assertEqual(3,vec3.nnn())
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_indexing_RHS_SpVec_scalar(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndx = 9
#        value = vec1[ndx]
#	self.assertEqual(-3, value)
#
#    def test_indexing_RHS_SpVec_scalar_outofbounds(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndx = 7777
#	self.assertRaises(IndexError, SpVec.__getitem__, vec1, ndx)
#        #value = vec1[ndx]
#	#self.assertEqual(-3, value)
#
#    def test_indexing_RHS_SpVec_scalar_outofbounds2(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndx = -333
#	self.assertRaises(IndexError, SpVec.__getitem__, vec1, ndx)
#        #value = vec1[ndx]
#	#self.assertEqual(-3, value)
#
#    def test_indexing_RHS_SpVec_SpVec(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 4
#        ndxI = [0, 1, 2,  3]
#        ndxV = [1, 4, 9, 16]
#        ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#	self.assertRaises(KeyError, SpVec.__getitem__, vec1, ndx)
#        #expI = [-11, -8, -3, 4]
#	#self.assertEqual(ndxLen, len(vec3))
#        #for ind in range(ndxLen):
#	#    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_indexing_RHS_SpVec_ParVec(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 4
#	ndx = ParVec(4)
#	ndx[0] = 1
#	ndx[1] = 4
#	ndx[2] = 9
#	ndx[3] = 16
#        vec3 = vec1[ndx]
#        expI = [-11, -8, -3, 4]
#	self.assertEqual(ndxLen, len(vec3))
#        for ind in range(ndxLen):
#	    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_indexing_RHS_SpVec_booleanSpVec(self):
#	sz = 25
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 25
#	ndxTrue = 4
#        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#		18, 19, 20, 21, 22, 23, 24]
#        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#		0, 0, 0, 0, 0]
#        ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#	self.assertRaises(KeyError, SpVec.__getitem__, vec1, ndx)
#        #vec3 = vec1[ndx]
#        #expI = [1,16,81,256]
#	#self.assertEqual(ndxTrue, len(vec3))
#        #for ind in range(ndxTrue):
#	#    self.assertEqual(expI[ind], vec3[ind])
#
#    def test_indexing_LHS_SpVec_booleanParVec_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 18
#	ndx = ParVec(sz)
#	ndx[1] = 1
#	ndx[4] = 1
#	ndx[9] = 1
#	ndx[16] = 1
#        vec1[ndx] = 77
#        expI = [-9, 77, -7, -6, 77, -4, -3, -2, -1, 77, 1, 2, 3, 4, 5, 6, 77, 8]
#	self.assertEqual(sz, len(vec1))
#        for ind in range(ndxLen):
#	    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_indexing_LHS_SpVec_nonbooleanParVec_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 18
#	ndx = ParVec(sz)
#	ndx[1] = 7
#	ndx[4] = 11
#	ndx[9] = 1
#	ndx[16] = 5
#	self.assertRaises(KeyError, SpVec.__setitem__, vec1, ndx, 77)
#        #vec1[ndx] = 77
#        #expI = [-9, 77, -7, -6, 77, -4, -3, -2, -1, 77, 1, 2, 3, 4, 5, 6, 77, 8]
#	#self.assertEqual(sz, len(vec1))
#        #for ind in range(ndxLen):
#	#    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_indexing_LHS_SpVec_scalar_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#        ndx = 11
#        value = 777
#        vec1[ndx] = value
#        expI = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 777, 3, 4, 5, 6, 7, 8]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(expI), len(vec1))
#        for ind in range(sz):
#	    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_indexing_LHS_SpVec_booleanSpVec_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 18
#	ndxTrue = 4
#        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#        ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#        value = 777
#	self.assertRaises(KeyError,SpVec.__setitem__,vec1,ndx, value)
#        #vec1[ndx] = value
#        #expI = [0, 777, 4, 9, 777, 25, 36, 49, 64, 777, 100, 121, 144, 169, 
#	#	196, 225, 777, 289]
#	#self.assertEqual(ndxLen, len(vec1))
#        #for ind in range(ndxLen):
#	#    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_indexing_LHS_SpVec_booleanSpVec_SpVec(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 18
#	ndxTrue = 4
#        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#        ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#        valueV = [0, 1, 7, 7, 0.25, 0, 0, 0, 0, 0.111, 0, 0, 0, 0, 0, 0, 0.0625,
#		0, 0, 0, 0, 7, 0, 0, 0]
#        value = self.initializeSpVec(ndxLen, ndxI, valueV)
#	self.assertRaises(KeyError,SpVec.__setitem__,vec1,ndx, value)
#        #vec1[ndx] = value
#        #expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#	#	196, 225, 0.0625, 289]
#	#self.assertEqual(ndxLen, len(vec1))
#        #for ind in range(ndxLen):
#	#    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_indexing_LHS_SpVec_ParVec(self):
#	sz = 18
#        vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#	ndxLen = 4
#        ndxI = [0, 1, 2, 3]
#        ndxV = [1, 4, 9, 16]
#        ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#        valueV = [1, 0.25, 0.111, 0.0625]
#        value = self.initializeSpVec(ndxLen, ndxI, valueV)
#	self.assertRaises(IndexError,SpVec.__setitem__,vec1,ndx, value)
#        #vec1[ndx] = value
#        #expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#	#	196, 225, 0.0625, 289]
#	#self.assertEqual(ndxLen, len(vec1))
#        #for ind in range(ndxLen):
#	#    self.assertEqual(expI[ind], vec1[ind])
#
#    def test_eq_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        scalar = 8
#	eq8 = vec1 == scalar
#        expV = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(eq8), len(vec1))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], eq8[ind])
#
#    def test_eq_vector(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        vec2 = SpVec.ones(sz)*4
#	eq4 = vec1 == vec2
#        expV = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(eq4), len(vec1))
#	self.assertEqual(len(eq4), len(vec2))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], eq4[ind])
#
#    def test_ne_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        scalar = 4
#	ne4 = vec1 != scalar
#        expV = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(ne4), len(vec1))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], ne4[ind])
#
#    def test_ne_vector(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        vec2 = SpVec.ones(sz)*4
#	ne4 = vec1 != vec2
#        expV = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(ne4), len(vec1))
#	self.assertEqual(len(ne4), len(vec2))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], ne4[ind])

    def test_ge_scalar(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
	scalar = 4
	geY = vec1 >= scalar
        expV = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(geY), len(vec1))
        for ind in range(sz):
	    self.assertEqual(expV[ind], geY[ind].weight)

    def test_ge_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 216, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
	geY = vec1 >= vec2
        expV = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(geY), len(vec1))
	self.assertEqual(len(geY), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], geY[ind].weight)

    def test_gt_scalar(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
	scalar = 4
	gtY = vec1 > scalar
        expV = [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(gtY), len(vec1))
        for ind in range(sz):
	    self.assertEqual(expV[ind], gtY[ind].weight)

    def test_gt_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 216, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
	gtY = vec1 > vec2
        expV = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(gtY), len(vec1))
	self.assertEqual(len(gtY), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], gtY[ind].weight)

#
#    def test_le_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        scalar = 4
#	le4 = vec1 <= scalar
#        expV = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(le4), len(vec1))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], le4[ind])
#
#    def test_le_vector(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        vec2 = SpVec.ones(sz)*4
#	le4 = vec1 <= vec2
#        expV = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(le4), len(vec1))
#	self.assertEqual(len(le4), len(vec2))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], le4[ind])
#
#    def test_lt_scalar(self):
#	sz = 18
#        vec1 = SpVec.range(sz)
#        scalar = 4
#	lt4 = vec1 < scalar
#        expV = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(lt4), len(vec1))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], lt4[ind])
#
#    def test_lt_vector(self):
#	sz = 25
#        i1 = [0, 2, 4, 6, 8, 10]
#        w1 = [0, 4, 16, 36, 64, 100]
#        t1 = [1, 1,  1,  2,  2,   2]
#        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
#        i2 = [0, 3, 6, 9, 12, 15, 18]
#        w2 = [1, 27, 216, 729, 1728, 3375, 5832]
#        t2 = [1, 1,  1,  2,  2, 2, 2]
#        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
#	geY = vec1 < vec2
#        expV = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
#		0, 0, 0, 0, 0, 0]
#	self.assertEqual(sz, len(vec1))
#	self.assertEqual(len(geY), len(vec1))
#	self.assertEqual(len(geY), len(vec2))
#        for ind in range(sz):
#	    self.assertEqual(expV[ind], geY[ind].weight)

    def test_lt_scalar(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
	scalar = 4
	ltY = vec1 < scalar
        expV = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ltY), len(vec1))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ltY[ind].weight)

    def test_lt_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        w1 = [0, 4, 16, 36, 64, 100]
        t1 = [1, 1,  1,  2,  2,   2]
        vec1 = self.initializeSpVecObj(sz, i1, (w1, t1))
        i2 = [0, 3, 6, 9, 12, 15, 18]
        w2 = [1, 27, 216, 729, 1728, 3375, 5832]
        t2 = [1, 1,  1,  2,  2, 2, 2]
        vec2 = self.initializeSpVecObj(sz, i2, (w2, t2))
	ltY = vec1 < vec2
        expV = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ltY), len(vec1))
	self.assertEqual(len(ltY), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ltY[ind].weight)


class GeneralPurposeTests(SpVecTests):
#    def test_all_all_nonnull_all_true(self):
#	sz = 10
#	vec = SpVec.ones(sz)
#	res = vec.all()
#	self.assertEqual(True, res)
#
#    def test_all_all_nonnull_one_false(self):
#	sz = 10
#	vec = SpVec.ones(sz)
#	vec[4] = False
#	res = vec.all()
#	self.assertEqual(False, res)
#
#    def test_all_all_nonnull_one_true(self):
#	sz = 10
#	vec = SpVec.ones(sz)-1
#	vec[4] = True
#	res = vec.all()
#	self.assertEqual(False, res)
#
#    def test_all_all_nonnull_all_false(self):
#	sz = 10
#	vec = SpVec.ones(sz)-1
#	res = vec.all()
#	self.assertEqual(False, res)
#
#    def test_all_one_nonnull_one_false(self):
#	sz = 10
#	vec = SpVec(sz)
#	vec[4] = 0
#	res = vec.all()
#	self.assertEqual(False, res)
#
#    def test_all_one_nonnull_one_true(self):
#	sz = 10
#	vec = SpVec(sz)
#	vec[4] = 1
#	res = vec.all()
#	self.assertEqual(True, res)
#
#    def test_all_three_nonnull_one_true(self):
#	sz = 10
#	vec = SpVec(sz)
#	vec[0] = 0
#	vec[2] = 0
#	vec[4] = 1
#	res = vec.all()
#	self.assertEqual(False, res)
#
#    def test_all_three_nonnull_three_true(self):
#	sz = 10
#	vec = SpVec(sz)
#	vec[0] = 1
#	vec[2] = 1
#	vec[4] = 1
#	res = vec.all()
#	self.assertEqual(True, res)
#
#    def test_all_all_nonnull_all_true(self):
#	sz = 6
#        i = [0, 1, 2, 3, 4, 5]
#	weight = [777, -4, 16, -36, -64, 100]
#	type = [2, -2, 2, 5, -5, 5]
#        vec = self.initializeSpVecObj(sz, i, (weight, type))
#	res = vec.all()
#	expRes = True
#	self.assertEqual(expRes, res)

#    def test_all_no_elem0(self):
#	sz = 6
#        i = [1, 2, 3, 4, 5]
#	weight = [-4, 16, 0, -64, 100]
#	type = [2, 5, 5, -5, 5]
#        vec = self.initializeSpVecObj(sz, i, (weight, type))
#	res = vec.all()
#	expRes = True
#	self.assertEqual(expRes, res)

#    def test_all_no_elem3(self):
#	sz = 6
#	i = [1, 2, 4, 5]
#	weight = [-1, -4, 16, -64, 100]
#	type = [2, 5, 5, -5, 5]
#	vec = self.initializeSpVecObj(sz, i, (weight, type))
#	res = vec.all()
#	expRes = True
#	self.assertEqual(expRes, res)

    def test_all_evens_nonnull_elem2_false(self):
	sz = 6
        i = [0, 2, 4]
	weight = [1, 0, -64]
	type = [2, 2, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.all()
	expRes = False
	self.assertEqual(expRes, res)

    def test_any_all_nonnull_all_true(self):
	sz = 6
        i = [0, 1, 2, 3, 4, 5]
	weight = [777, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.any()
	expRes = True
	self.assertEqual(expRes, res)

    def test_any_all_nonnull_elem0_false(self):
	sz = 6
        i = [0, 1, 2, 3, 4, 5]
	weight = [0, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.any()
	expRes = True
	self.assertEqual(expRes, res)

    def test_any_all_nonnull_elem3_false(self):
	sz = 6
        i = [0, 1, 2, 3, 4, 5]
	weight = [1, -4, 16, 0, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.any()
	expRes = True
	self.assertEqual(expRes, res)

    def test_any_evens_nonnull_elem2_false(self):
	sz = 6
        i = [0, 2, 4]
	weight = [1, 0, -64]
	type = [2, 2, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.any()
	expRes = True
	self.assertEqual(expRes, res)

#    def test_any_all_true(self):
#	sz = 10
#	vec = SpVec.ones(sz)
#	res = vec.any()
#	self.assertEqual(True, res)
#
#    def test_any_one_false(self):
#	sz = 10
#	vec = SpVec.ones(sz)
#	vec[4] = False
#	res = vec.any()
#	self.assertEqual(True, res)
#
#    def test_any_one_true(self):
#	sz = 10
#	vec = SpVec.ones(sz)-1
#	vec[4] = True
#	res = vec.any()
#	self.assertEqual(True, res)
#
#    def test_any_all_false(self):
#	sz = 10
#	vec = SpVec.ones(sz)-1
#	res = vec.any()
#	self.assertEqual(False, res)
#
#    def test_sum_nulls(self):
#	sz = 10
#	vec = SpVec(sz)
#	res = vec.sum()
#	self.assertEqual(0.0, res)
#
#    def test_sum_ones(self):
#	sz = 10
#	vec = SpVec.ones(sz)
#	res = vec.sum()
#	self.assertEqual(sz, res)
#
#    def test_sum_range(self):
#	sz = 10
#	vec = SpVec.range(sz)
#	res = vec.sum()
#	self.assertEqual((sz*(sz-1))/2, res)
#
#    def test_sum_range2(self):
#	sz = 11
#	vec = SpVec.range(-(sz/2), (sz/2)+1)
#	res = vec.sum()
#	self.assertEqual(0, res)
#
#    def test_sum_fixed(self):
#	sz = 11
#	vec = SpVec(sz)
#	vec[2] = 23
#	vec[4] = 9
#	vec[5] = -32
#	res = vec.sum()
#	self.assertEqual(0, res)

    def test_abs(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
        expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
        expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	res = abs(vec)
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertAlmostEqual(expW[ind],res[ind].weight)
            #self.assertEqual(expT[ind],res[ind].type)

    def test_max_some_nonnull_maxElem0(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [123, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.max()
	expRes = 123
	self.assertEqual(expRes, res.weight)

    def test_max_all_nonnull_maxElem4(self):
	sz = 6
        i = [0, 1, 2, 3, 4, 5]
	weight = [0, -4, 16, 136, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.max()
	expRes = 136
	self.assertEqual(expRes, res.weight)

    def test_max_some_nonnull_maxElem6(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, -4, 16, 136, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.max()
	expRes = 136
	self.assertEqual(expRes, res.weight)

    def test_min_some_nonnull_minElem0(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [-123, -4, 16, -36, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.min()
	expRes = -123
	self.assertEqual(expRes, res.weight)

    def test_min_all_nonnull_minElem4(self):
	sz = 6
        i = [0, 1, 2, 3, 4, 5]
	weight = [0, -4, 16, 136, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.min()
	expRes = -64
	self.assertEqual(expRes, res.weight)

    def test_min_some_nonnull_minElem8(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
	weight = [0, -4, 16, 136, -64, 100]
	type = [2, -2, 2, 5, -5, 5]
        vec = self.initializeSpVecObj(sz, i, (weight, type))
	res = vec.min()
	expRes = -64
	self.assertEqual(expRes, res.weight)

class MixedDenseSparseVecTests(SpVecTests):
    def test_add_sparse_dense(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        i2 = [0, 1,   3,  5, 7]
        v2 = [0,-2, 0.1,777, 0]
	vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec + vec2
        expI = [0, -2, 4, 0.1, 8, 777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
	# odd behavior here; element 0 in vec1 and element 7 in vec2 become
	# nulls in vec3 (!)
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_add_dense_sparse(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        i2 = [0, 1,   3,  5, 7]
        v2 = [0,-2, 0.1,777, 0]
	vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec2 + vec
        expI = [0, -2, 4, 0.1, 8, 777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
	#self.assertEqual(len(i)+len(i2)-1, vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_subtract_sparse_dense(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        i2 = [0, 1,   3,  5, 7]
        v2 = [0,-2, 0.1,777, 0]
	vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec - vec2
        expI = [0, 2, 4, -0.1, 8, -777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
	# odd behavior here; element 0 in vec1 and element 7 in vec2 become
	# nulls in vec3 (!)
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_isubtract_sparse_dense(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        i2 = [0, 1,   3,  5, 7]
        v2 = [0,-2, 0.1,777, 0]
	vec2 = self.initializeParVec(sz, i2, v2)
	vec3 = vec.copy()
        vec3 -= vec2
        expI = [0, 2, 4, -0.1, 8, -777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
	# odd behavior here; element 0 in vec1 and element 7 in vec2 become
	# nulls in vec3 (!)
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_subtract_dense_sparse(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        i2 = [0, 1,   3,  5, 7]
        v2 = [0,-2, 0.1,777, 0]
	vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec2 - vec
        expI = [0, -2, -4, 0.1, -8, 777, -12, 0, -16, 0, -20, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
	#self.assertEqual(len(i)+len(i2)-1, vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

class ApplyReduceTests(SpVecTests):
    def test_apply(self):
	def ge0lt5(x):
		return x>=0 and x<5
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        vec._apply(ge0lt5)
        vecExpected = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
	    self.assertEqual(vecExpected[ind], vec[ind])

    def test_apply_pcbabs(self):
	sz = 25
        i = [0, 2,  4,   6, 8, 10]
        v = [0, -4, 8, -12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        vec._apply(pcb.abs())
        vecExpected = [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
	    self.assertEqual(vecExpected[ind], vec[ind])

    def test_count(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        ct = vec.count()
        ctExpected = 6
	self.assertEqual(ctExpected, ct)

    def test_reduce_default_op(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        red = vec._reduce(SpVec.op_add)
        redExpected = 60
	self.assertEqual(redExpected, red)

    def test_reduce_max(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [0, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        red = vec._reduce(SpVec.op_max)
        redExpected = 20
	self.assertEqual(redExpected, red)

    def test_reduce_min(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        v = [2, 4, 8,12,16, 20]
        vec = self.initializeSpVec(sz, i, v)
        red = vec._reduce(SpVec.op_min)
        redExpected = 2
	self.assertEqual(redExpected, red)

	

def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
#    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MixedDenseSparseVecTests))
#    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ApplyReduceTests))
    return suite

if __name__ == '__main__':
    runTests()
