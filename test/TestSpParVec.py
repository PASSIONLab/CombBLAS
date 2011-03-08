import unittest
from kdt import *
from kdt import pyCombBLAS as pcb

class SpParVecTests(unittest.TestCase):
    def initializeSpParVec(self, length, i, v=1):
        """
        Initialize a ParVec instance with values equal to one or the input value.
        """
        ret = SpParVec(length)
        for ind in range(len(i)):
	    if type(v) != int and type(v) != float:
		ret[i[ind]] = v[ind]
            else:
                ret[i[ind]] = v

        return ret

class ConstructorTests(SpParVecTests):
    def test_SpParVec_simple(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeSpParVec(sz, i)
        expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI[ind],vec[ind])

    def test_SpParVec_zeros(self):
	sz = 25
        vec = SpParVec(sz)
        expI = 0
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI,vec[ind])

    def test_SpParVec_ones(self):
	sz = 25
        vec = SpParVec.ones(sz)
        expI = 1
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(expI,vec[ind])

    def test_SpParVec_range_simple(self):
	sz = 25
        vec = SpParVec.range(sz)
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(ind,vec[ind])

    def test_SpParVec_range_offset(self):
	sz = 25
	offset = -13
        vec = SpParVec.range(offset, sz+offset)
        expI = 1
	self.assertEqual(sz, len(vec))
        for ind in range(sz):
            self.assertEqual(ind,vec[ind]-offset)

class BuiltInTests(SpParVecTests):
    def test_add_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeSpParVec(sz, i)
        vec2 = vec + 3.07
        expI = [4.07, 0, 4.07, 0, 4.07, 0, 4.07, 0, 4.07, 0, 
		4.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
	self.assertEqual(len(i), vec2.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_add_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        v1 = [0, 4, 16, 36, 64, 100]
        vec1 = self.initializeSpParVec(sz, i1, v1)
        i2 = [0, 3, 6, 9, 12, 15, 18]
        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
        vec2 = self.initializeSpParVec(sz, i2, v2)
        vec3 = vec1 + vec2
        expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
		0, 0, 5832, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_subtract_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeSpParVec(sz, i)
        vec2 = vec - 3.07
        expI = [-2.07, 0, -2.07, 0, -2.07, 0, -2.07, 0, -2.07,
		0, -2.07, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
	self.assertEqual(vec.nnn(), len(i))
	self.assertEqual(vec.nnn(), vec2.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_subtract_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10]
        v1 = [0, 4, 16, 36, 64, 100]
        vec1 = self.initializeSpParVec(sz, i1, v1)
        i2 = [0, 3, 6, 9, 12, 15, 18]
        v2 = [1, 27, 216, 729, 1728, 3375, 5832]
        vec2 = self.initializeSpParVec(sz, i2, v2)
        vec3 = vec1 - vec2
        expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0,
		-3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_multiply_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10]
        vec = self.initializeSpParVec(sz, i)
        vec2 = vec * 3.07
        expI = [3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 3.07, 0, 
		3.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
	self.assertEqual(vec.nnn(), vec2.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_multiply_vector(self):
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec1 = self.initializeSpParVec(sz, i1, v1)
	vec2 = ParVec(sz)
	vec2[0] = 1
	vec2[3] = 3
	vec2[6] = 6
	vec2[9] = 9
	vec2[12] = 12
	vec2[15] = 15
	vec2[18] = 18
        vec3 = vec1 * vec2
        expI = [1, 0, 0, 0, 0, 0, 216, 0, 0, 0, 0, 0, 1728, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(vec3.nnn(), 3)
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_divide_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec = self.initializeSpParVec(sz, i, v1)
        vec2 = vec / .50
        expI = [2, 0, 8, 0, 32, 0, 72, 0, 128, 0, 
		200, 0, 288, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
	self.assertEqual(vec.nnn(), vec2.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_divide_vector(self):
	return
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec1 = self.initializeSpParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24]
        v2 = [1, 1, 1, 3, 1, 1, 6, 1, 1, 9, 1, 1, 12, 1, 1, 15, 
		1, 1, 18, 1, 1, 1, 1, 1, 1]
        vec2 = self.initializeSpParVec(sz, i2, v2)
        vec3 = vec1 / vec2
        expI = [1, 0, 4, 0, 16, 0, 6, 0, 64, 0, 100, 0, 12, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(vec.nnn(), vec3.nnn())
        for ind in range(sz):
	    self.assertAlmostEqual(expI[ind], vec3[ind])

    def test_modulus_constant(self):
	sz = 25
        i = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec = self.initializeSpParVec(sz, i, v1)
        vec2 = vec % 5
        expI = [1, 0, 4, 0, 1, 0, 1, 0, 4, 0, 
		0, 0, 4, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec2))
	self.assertEqual(vec.nnn(), vec2.nnn())
        for ind in range(sz):
	    self.assertEqual(expI[ind], vec2[ind])

    def test_modulus_vector(self):
	return
	sz = 25
        i1 = [0, 2, 4, 6, 8, 10, 12]
        v1 = [1, 4, 16, 36, 64, 100, 144]
        vec1 = self.initializeSpParVec(sz, i1, v1)
        i2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24]
        v2 = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 
		3, 2, 3, 2, 3]
        vec2 = self.initializeSpParVec(sz, i2, v2)
        vec3 = vec1 % vec2
        expI = [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,]
	self.assertEqual(sz, len(vec3))
	self.assertEqual(vec.nnn(), vec3.nnn())
        for ind in range(sz):
	    self.assertAlmostEqual(expI[ind], vec3[ind])

    def test_indexing_RHS_SpParVec_scalar(self):
	# Disabled for now
	return	
	sz = 25
        vec1 = SpParVec.range(sz)*SpParVec.range(sz)
	ndx = 9
        value = vec1[ndx]
	self.assertEqual(ndx*ndx, value)

    def test_indexing_RHS_SpParVec_SpParVec(self):
	# Disabled for now
	return	
	sz = 25
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 4
        ndxI = [0, 1, 2,  3]
        ndxV = [1, 4, 9, 16]
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        vec3 = vec1[ndx]
        expI = [1,16,81,256]
	self.assertEqual(ndxLen, len(vec3))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_indexing_RHS_SpParVec_booleanSpParVec(self):
	# Disabled for now
	return	
	sz = 25
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 25
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0]
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        vec3 = vec1[ndx]
        expI = [1,16,81,256]
	self.assertEqual(ndxTrue, len(vec3))
        for ind in range(ndxTrue):
	    self.assertEqual(expI[ind], vec3[ind])

    def test_indexing_LHS_ParVec_ParVec(self):
	# Disabled because underlying ParVec method not implemented
	return	
	sz = 15
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 4
        ndxI = [0, 1, 2, 3]
        ndxV = [3, 5, 7, 9]
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        vec1[ndx] = ndx
        expI = [0, 1, 4, 3, 16, 5, 36, 7, 64, 9, 100, 121, 144, 169, 196]
	self.assertEqual(sz, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_scalar_scalar(self):
	# Disabled for now
	return	
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

    def test_indexing_LHS_ParVec_booleanParVec_scalar(self):
	# Disabled for now
	return	
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 18
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        value = 777
        vec1[ndx] = value
        expI = [0, 777, 4, 9, 777, 25, 36, 49, 64, 777, 100, 121, 144, 169, 
		196, 225, 777, 289]
	self.assertEqual(ndxLen, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_indexing_LHS_ParVec_booleanParVec_ParVec(self):
	# Disabled for now
	return	
	sz = 18
        vec1 = ParVec.range(sz)*ParVec.range(sz)
	ndxLen = 18
	ndxTrue = 4
        ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        valueV = [0, 1, 7, 7, 0.25, 0, 0, 0, 0, 0.111, 0, 0, 0, 0, 0, 0, 0.0625,
		0, 0, 0, 0, 7, 0, 0, 0]
        value = self.initializeSpParVec(ndxLen, ndxI, valueV)
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
        ndx = self.initializeSpParVec(ndxLen, ndxI, ndxV)
        valueV = [1, 0.25, 0.111, 0.0625]
        value = self.initializeSpParVec(ndxLen, ndxI, valueV)
        vec1[ndx] = value
        expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
		196, 225, 0.0625, 289]
	self.assertEqual(ndxLen, len(vec1))
        for ind in range(ndxLen):
	    self.assertEqual(expI[ind], vec1[ind])

    def test_eq_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	eq4 = vec1 == vec2
        expV = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(eq4), len(vec1))
	self.assertEqual(len(eq4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], eq4[ind])

    def test_ne_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	ne4 = vec1 != vec2
        expV = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ne4), len(vec1))
	self.assertEqual(len(ne4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ne4[ind])

    def test_ge_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	ge4 = vec1 >= vec2
        expV = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(ge4), len(vec1))
	self.assertEqual(len(ge4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], ge4[ind])

    def test_gt_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	gt4 = vec1 > vec2
        expV = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(gt4), len(vec1))
	self.assertEqual(len(gt4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], gt4[ind])

    def test_le_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	le4 = vec1 <= vec2
        expV = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(le4), len(vec1))
	self.assertEqual(len(le4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], le4[ind])

    def test_lt_vector(self):
	sz = 18
        vec1 = SpParVec.range(sz)
        vec2 = SpParVec.ones(sz)*4
	lt4 = vec1 < vec2
        expV = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	self.assertEqual(sz, len(vec1))
	self.assertEqual(len(lt4), len(vec1))
	self.assertEqual(len(lt4), len(vec2))
        for ind in range(sz):
	    self.assertEqual(expV[ind], lt4[ind])

class GeneralPurposeTests(SpParVecTests):
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


def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
    return suite

if __name__ == '__main__':
    runTests()
