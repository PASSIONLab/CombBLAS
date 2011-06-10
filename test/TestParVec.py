import unittest
from kdt import *
from kdt import pyCombBLAS as pcb

class ParVecTests(unittest.TestCase):
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

class ConstructorTests(ParVecTests):
    def test_ParVec_simple(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeParVec(sz, i)
        expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI[ind],vec[ind])

    def test_ParVec_zeros(self):
	sz = 25
        vec = ParVec.zeros(sz)
        expI = 0
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI,vec[ind])

    def test_ParVec_ones(self):
	sz = 25
        vec = ParVec.ones(sz)
        expI = 1
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI,vec[ind])

    def test_ParVec_range_simple(self):
	sz = 25
        vec = ParVec.range(sz)
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(ind,vec[ind])

    def test_ParVec_range_offset(self):
	sz = 25
	offset = -13
        vec = ParVec.range(offset, sz+offset)
        expI = 1
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(ind,vec[ind]-offset)

    def test_ParVec_toSpParVec(self):
	sz = 25
	offset = -13
        vec = (ParVec.range(offset, sz+offset) % 3).toSpParVec()
        expV = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 
		2, 0, 1, 2]
	self.assertEqual(sz, len(vec))
	self.assertEqual(17, vec.nnn())
        for ind in range(sz):
            self.assertEqual(expV[ind], vec[ind])

class BuiltInTests(ParVecTests):
    def test_add_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeParVec(sz, i)
        vec2 = vec + 3.07
        expI = [4.07, 3.07, 4.07, 3.07, 4.07, 3.07, 4.07, 3.07, 4.07, 3.07, 
		4.07, 3.07, 3.07, 3.07, 3.07, 3.07, 3.07, 3.07, 3.07, 3.07, 
		3.07, 3.07, 3.07, 3.07, 3.07, 3.07]
	self.assertEqual(sz, len(vec2))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_add_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        v1 = [0, 4, 16, 36, 64, 100]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 3, 6, 9, 12, 15, 18]
        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 + vec2
        expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
		0, 0, 5832, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_subtract_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeParVec(sz, i)
        vec2 = vec - 3.07
        expI = [-2.07, -3.07, -2.07, -3.07, -2.07, -3.07, -2.07, -3.07, -2.07,
		-3.07, -2.07, -3.07, -3.07, -3.07, -3.07, -3.07, -3.07, -3.07,
		-3.07, -3.07, -3.07, -3.07, -3.07, -3.07, -3.07, -3.07]
	self.assertEqual(sz, len(vec2))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_subtract_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        v1 = [0, 4, 16, 36, 64, 100]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 3, 6, 9, 12, 15, 18]
        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 - vec2
        expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0,
		-3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_negate(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        v1 = [0, -4, 16, -36, 64, 100]
        vec1 = self.initializeParVec(sz, i1, v1)
        vec3 = -vec1
        expI = [0, 0, 4, 0, -16, 0, 36, 0, -64, 0, -100, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_multiply_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeParVec(sz, i)
        vec2 = vec * 3.07
        expI = [3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 
		3.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_multiply_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 3, 6, 9, 12, 15, 18]
        v2 = [1, 3, 6, 9, 12, 15, 18]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 * vec2
        expI = [1, 0, 0, 0, 0, 0, 216, 0, 0, 0, 0, 0, 1728, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_divide_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec = self.initializeParVec(sz, i, v1)
        vec2 = vec / .50
        expI = [2, 0, 8, 0, 32, 0, 72, 0, 128, 0, 
		200, 0, 288, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_divide_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24]
        v2 = [1, 1, 1, 3, 1, 1, 6, 1, 1, 9, 1, 1, 12, 1, 1, 15, 
		1, 1, 18, 1, 1, 1, 1, 1, 1]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 / vec2
        expI = [1, 0, 4, 0, 16, 0, 6, 0, 64, 0, 100, 0, 12, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertAlmostEqual(expI[ind], vec3[ind])

    def test_modulus_constant(self):
	sz = 25
        i =  [0,   2,  4,  6,     8,  10,  12]
        v1 = [1, 4.1, 16, 36, -64.3, 100, 144]
        vec = self.initializeParVec(sz, i, v1)
        vec2 = vec % 5
        expI = [1, 0, 4.1, 0, 1, 0, 1, 0, 0.7, 0, 
		0, 0, 4, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
        for ind in range(sz):
	    self.assertAlmostEqual(expI[ind], vec2[ind])

    def test_modulus_vector(self):
	sz = 25
        i1 = [   0,  2,  4,  6,  8,     10,  12]
        v1 = [1.04, -4, 16,-36, 64, -100.1, 144]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24]
        v2 = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 
		3, 2, 3, 2, 3]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 % vec2
        expI = [1.04, 0, 2, 0, 1, 0, 0, 0, 1, 0, 1.9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertAlmostEqual(expI[ind], vec3[ind])

    def test_indexing_RHS_ParVec_scalar(self):
	sz = 25
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndx = 9
        value = vec1[ndx]
	self.assertEqual(ndx*ndx, value)

    def test_indexing_RHS_ParVec_scalar_scalar_negative(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
        ndx = -4
        value = 777
	self.assertRaises(IndexError,ParVec.__getitem__,vec1, ndx)

    def test_indexing_RHS_ParVec_scalar_scalar_tooBig(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
        ndx = 2**36
        value = 777
	self.assertRaises(IndexError,ParVec.__getitem__,vec1, ndx)

    def test_indexing_RHS_ParVec_ParVec(self):
	sz = 25
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 4
        ndxI = [0, 1, 2,  3]
        ndxV = [1, 4, 9, 16]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        vec3 = vec1[ndx]
        expI = [1,16,81,256]
	self.assertEqual(ndxLen, len(vec3))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_indexing_RHS_ParVec_booleanParVec(self):
	sz = 25
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 25
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        vec3 = vec1[ndx]
        expI = [1,16,81,256]
	self.assertEqual(ndxTrue, len(vec3))
        for ind in range(ndxTrue):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_indexing_RHS_ParVec_ParVec_scalar_negative(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndx = ParVec(1)
        ndx[0] = -4
	self.assertRaises(IndexError,ParVec.__getitem__,vec1, ndx)

    def test_indexing_RHS_ParVec_ParVec_scalar_tooBig(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndx = ParVec(1)
        ndx[0] = 2**36
	self.assertRaises(IndexError,ParVec.__getitem__,vec1, ndx)

    def test_indexing_LHS_ParVec_ParVec(self):
	# Disabled because underlying ParVec method not implemented
	return	
	sz = 15
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 4
        ndxI = [0, 1, 2, 3]
        ndxV = [3, 5, 7, 9]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        vec1[ndx] = ndx
        expI = [0, 1, 4, 3, 16, 5, 36, 7, 64, 9, 100, 121, 144, 169, 196]
	self.assertEqual(sz, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_scalar_scalar(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
        ndx = 11
        value = 777
        vec1[ndx] = value
        expI = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 777, 144, 169, 
		196, 225, 256, 289]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(expI), len(vec1))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_scalar_scalar_negative(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
        ndx = -4
        value = 777
	self.assertRaises(IndexError,ParVec.__setitem__,vec1, ndx, value)

    def test_indexing_LHS_ParVec_scalar_scalar_tooBig(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
        ndx = 2**36
        value = 777
	self.assertRaises(IndexError,ParVec.__setitem__,vec1, ndx, value)

    def test_indexing_LHS_ParVec_booleanParVec_scalar(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 18
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        value = 777
        vec1[ndx] = value
        expI = [0, 777, 4, 9, 777, 25, 36, 49, 64, 777, 100, 121, 144, 169, 
		196, 225, 777, 289]
	self.assertEqual(ndxLen, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_booleanParVec_ParVec(self):
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 18
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        valueV = [0, 1, 7, 7, 0.25, 0, 0, 0, 0, 0.111, 0, 0, 0, 0, 0, 0, 0.0625,
		0, 0, 0, 0, 7, 0, 0, 0]
        value = self.initializeParVec(ndxLen, ndxI, valueV)
        vec1[ndx] = value
        expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
		196, 225, 0.0625, 289]
	self.assertEqual(ndxLen, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_ParVec(self):
	# Disabled because underlying ParVec method not implemented
	return	
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 4
        ndxI = [0, 1, 2, 3]
        ndxV = [1, 4, 9, 16]
        ndx = self.initializeParVec(ndxLen, ndxI, ndxV)
        valueV = [1, 0.25, 0.111, 0.0625]
        value = self.initializeParVec(ndxLen, ndxI, valueV)
        vec1[ndx] = value
        expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
		196, 225, 0.0625, 289]
	self.assertEqual(ndxLen, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_eq_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	eq4 = vec1 == vec2
        expV = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(eq4), len(vec1))
	self.assertEqual(len(eq4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], eq4[ind])

    def test_ne_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	ne4 = vec1 != vec2
        expV = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ne4), len(vec1))
	self.assertEqual(len(ne4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ne4[ind])

    def test_ge_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	ge4 = vec1 >= vec2
        expV = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ge4), len(vec1))
	self.assertEqual(len(ge4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ge4[ind])

    def test_gt_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	gt4 = vec1 > vec2
        expV = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(gt4), len(vec1))
	self.assertEqual(len(gt4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], gt4[ind])

    def test_le_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	le4 = vec1 <= vec2
        expV = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(le4), len(vec1))
	self.assertEqual(len(le4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], le4[ind])

    def test_lt_vector(self):
	sz = 18
        vec1 = ParVec.range(sz)
        vec2 = ParVec(sz, 4)
	lt4 = vec1 < vec2
        expV = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(lt4), len(vec1))
	self.assertEqual(len(lt4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], lt4[ind])

    def test_and_vector(self):
	sz = 18
        i1 = [0, 2, 4, 6, 8, 10, 12, 14]
        v1 = [1, 1, 1, 1, 1,  1,  1,  1]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        v2 = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0,  0,  1,  0,  0,  1,  0,  0,  1]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 & vec2
        expI = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_or_vector(self):
	sz = 18
        i1 = [0, 2, 4, 6, 8, 10, 12, 14]
        v1 = [1, 1, 1, 1, 1,  1,  1,  1]
        vec1 = self.initializeParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        v2 = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0,  0,  1,  0,  0,  1,  0,  0,  1]
        vec2 = self.initializeParVec(sz, i2, v2)
        vec3 = vec1 | vec2
        expI = [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
	self.assertEqual(sz, len(vec3))
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])


class GeneralPurposeTests(ParVecTests):
    def test_all_all_true(self):
	sz = 10
	vec = ParVec.ones(sz)
	res = vec.all()
	self.assertEqual(True, res)

    def test_all_one_false(self):
	sz = 10
	vec = ParVec.ones(sz)
	vec[4] = False
	res = vec.all()
	self.assertEqual(False, res)

    def test_all_one_true(self):
	sz = 10
	vec = ParVec.zeros(sz)
	vec[4] = True
	res = vec.all()
	self.assertEqual(False, res)

    def test_all_all_false(self):
	sz = 10
	vec = ParVec.zeros(sz)
	res = vec.all()
	self.assertEqual(False, res)

    def test_any_all_true(self):
	sz = 10
	vec = ParVec.ones(sz)
	res = vec.any()
	self.assertEqual(True, res)

    def test_any_one_false(self):
	sz = 10
	vec = ParVec.ones(sz)
	vec[4] = False
	res = vec.any()
	self.assertEqual(True, res)

    def test_any_one_true(self):
	sz = 10
	vec = ParVec.zeros(sz)
	vec[4] = True
	res = vec.any()
	self.assertEqual(True, res)

    def test_any_all_false(self):
	sz = 10
	vec = ParVec.zeros(sz)
	res = vec.any()
	self.assertEqual(False, res)

    def test_sum_zeros(self):
	sz = 10
	vec = ParVec(sz)
	res = vec.sum()
	self.assertEqual(0, res)

    def test_sum_ones(self):
	sz = 10
	vec = ParVec.ones(sz)
	res = vec.sum()
	self.assertEqual(sz, res)

    def test_sum_range(self):
	sz = 10
	vec = ParVec.range(sz)
	res = vec.sum()
	self.assertEqual((sz*(sz-1))/2, res)

    def test_sum_range2(self):
	sz = 11
	vec = ParVec.range(-(sz/2), (sz/2)+1)
	res = vec.sum()
	self.assertEqual(0, res)

    def test_sum_fixed(self):
	sz = 11
	vec = ParVec(sz)
	vec[2] = 23
	vec[4] = 9
	vec[5] = -32
	res = vec.sum()
	self.assertEqual(0, res)

    def test_floor_vector(self):
	sz = 11
	vec = ParVec(sz)
	vec[0] = 0.01
	vec[1] = 0.99999
	vec[2] = 3
	vec[3] = -0.01
	vec[4] = -0.99999
	vec[5] = 9.01
	vec[6] = 9.99999
	vec[7] = -9.01
	vec[8] = -9.99999
	vec2 = vec.floor()
	self.assertEqual(sz, len(vec))
	self.assertEqual(sz, len(vec2))
        expI = [0, 0, 3, -1, -1, 9, 9, -10, -10, 0, 0]
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_ceil_vector(self):
	sz = 11
	vec = ParVec(sz)
	vec[0] = 0.01
	vec[1] = 0.99999
	vec[2] = 3
	vec[3] = -0.01
	vec[4] = -0.99999
	vec[5] = 9.01
	vec[6] = 9.99999
	vec[7] = -9.01
	vec[8] = -9.99999
	vec2 = vec.ceil()
	self.assertEqual(sz, len(vec))
	self.assertEqual(sz, len(vec2))
        expI = [1, 1, 3, 0, 0, 10, 10, -9, -9, 0, 0]
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_round_vector(self):
	sz = 14
	vec = ParVec(sz)
	vec[0] = 0.01
	vec[1] = 0.99999
	vec[2] = 3
	vec[3] = -0.01
	vec[4] = -0.99999
	vec[5] = 9.01
	vec[6] = 9.99999
	vec[7] = -9.01
	vec[8] = -9.99999
	vec[9] = -9.5
	vec[10] = -10.5
	vec[11] = 9.5
	vec[12] = 10.5
	vec2 = vec.round()
	self.assertEqual(sz, len(vec))
	self.assertEqual(sz, len(vec2))
        expI = [0, 1, 3, 0, -1, 9, 10, -9, -10, -10, -10, 10, 10, 0]
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_find(self):
	sz = 14
	vec = (ParVec.range(sz)*3) > 8
	vec2 = vec.find()
	self.assertEqual(sz, len(vec))
	self.assertEqual(sz, len(vec2))
        expI = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_findInds(self):
	sz = 14
	vec = (ParVec.range(sz)*3) > 8
	vec2 = vec.findInds()
	expSz = 11
	self.assertEqual(sz, len(vec))
	self.assertEqual(expSz, len(vec2))
        expI = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        for ind in range(expSz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_mean_simple(self):
	sz = 11
	vec = ParVec.range(sz)
	res = vec.mean()
	self.assertAlmostEqual(((sz*(sz-1))/2)/sz, res)

    def test_mean_fixed(self):
	sz = 9
	vec = ParVec(sz)
	vec[0] = -4.777
	vec[1] = -3.222
	vec[2] = -2.789
	vec[3] = -0.999
	vec[4] = 0
	vec[5] = 0.999
	vec[6] = 2.789
	vec[7] = 3.222
	vec[8] = 4.777
	res = vec.mean()
	self.assertAlmostEqual(0, res)

    def test_std_fixed(self):
	sz = 8
	vec = ParVec(sz)
	vec[0] = 2
	vec[1] = 4
	vec[2] = 4
	vec[3] = 4
	vec[4] = 5
	vec[5] = 5
	vec[6] = 7
	vec[7] = 9
	res = vec.std()
	self.assertAlmostEqual(2.0, res)

    def test_std_fixed2(self):
	sz = 9
	vec = ParVec(sz)
	vec[0] = -4.777
	vec[1] = -3.222
	vec[2] = -2.789
	vec[3] = -0.999
	vec[4] = 0
	vec[5] = 0.999
	vec[6] = 2.789
	vec[7] = 3.222
	vec[8] = 4.777
	res = vec.std()
	self.assertAlmostEqual(3.0542333098686338, res)

    def test_sort(self):
	sz = 9
	vec = ParVec.range(-sz/2, sz/2).abs()
	vec.sort()
	self.assertEqual(sz, len(vec))
	expV = [0, 1, 1, 2, 2, 3, 3, 4, 5]
        for ind in range(sz):
	    self.assertEqual(expV[ind], vec[ind])

    def test_sorted(self):
	sz = 9
	vec = ParVec.range(-sz/2, sz/2).abs()
	[sortedVec, permVec] = vec.sorted()
	self.assertEqual(sz, len(sortedVec))
	self.assertEqual(sz, len(permVec))
	expV = [0, 1, 1, 2, 2, 3, 3, 4, 5]
	expPerm = [ 5, 4, 6, 3, 7, 2, 8, 1, 0]
        for ind in range(sz):
	    self.assertEqual(expV[ind], sortedVec[ind])
	    self.assertEqual(expPerm[ind], permVec[ind])

    def test_hist(self):
	sz = 14
	vec = ParVec.range(sz) % 3
	actualV = vec.hist()
	expLen = 3
	self.assertEqual(sz, len(vec))
	expV = [5, 5, 4]
        for ind in range(expLen):
	    self.assertEqual(expV[ind], actualV[ind])

    def test_argmax_simple(self):
	sz = 11
	vec = ParVec.range(sz)
	res = vec.argmax()
	self.assertEqual(sz-1, res)

    def test_argmax_simple2(self):
	sz = 11
	vec = -ParVec.range(sz)
	res = vec.argmax()
	self.assertEqual(0, res)

    def test_argmax_simple3(self):
	sz = 11
	vec = -(ParVec.range(-(sz/2), sz/2).abs()) + 10
	res = vec.argmax()
	self.assertEqual(5, res)

    def test_argmax_fixed(self):
	sz = 9
	vec = ParVec(sz)
	vec[0] = -4.777
	vec[1] = -3.222
	vec[2] = -2.789
	vec[3] = -0.999
	vec[4] = 0
	vec[5] = 0.999
	vec[6] = 7.789
	vec[7] = 3.222
	vec[8] = 4.777
	res = vec.argmax()
	self.assertEqual(6, res)

    def test_argmin_simple(self):
	sz = 11
	vec = ParVec.range(sz)
	res = vec.argmin()
	self.assertEqual(0, res)

    def test_argmin_simple2(self):
	sz = 11
	vec = -ParVec.range(sz)
	res = vec.argmin()
	self.assertEqual(sz-1, res)

    def test_argmin_simple3(self):
	sz = 11
	vec = -(ParVec.range(-(sz/2), sz/2).abs()) + 10
	res = vec.argmin()
	self.assertEqual(0, res)

    def test_argmin_fixed(self):
	sz = 9
	vec = ParVec(sz)
	vec[0] = -4.777
	vec[1] = -3.222
	vec[2] = -7.789
	vec[3] = -0.999
	vec[4] = 0
	vec[5] = 0.999
	vec[6] = 7.789
	vec[7] = 3.222
	vec[8] = 4.777
	res = vec.argmin()
	self.assertEqual(2, res)

def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
    return suite

if __name__ == '__main__':
    runTests()
