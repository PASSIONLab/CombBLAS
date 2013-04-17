import unittest
import math
from kdt import *

def ge0lt5(self):
	if isinstance(self, (float, int, long)):
		return self >= 0 and self < 5
	else:
		return self.weight >= 0 and self.weight < 5

def geM2lt4(self):
	if isinstance(self, (float, int, long)):
		return self >= -2 and self < 4
	else:
		return self.weight >= -2 and self.weight < 4

# AL note: disabled tests for all/any on object vectors because I'm not sure that function
# makes any sense for objects. search for "disableObjAll_" to re-enable 

class SpVecTests(unittest.TestCase):
	def fillVec(self, ret, i, v, element):
		"""
		Initialize a Vec instance with values equal to one or the input value.
		"""
		for ind in range(len(i)):
			if isinstance(element, (float, int, long)):
				if type(v) != int and type(v) != float:
					val = v[ind]
				else:
					val = v
			elif isinstance(element, Obj1):
				val = pcb.Obj1()
				if type(v) == tuple:
					val.weight = v[0][ind]
					val.category   = v[1][ind]
				else:
					val.weight = v
					val.category   = v
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				if type(v) == tuple:
					val.latest = v[0][ind]
					val.count   = v[1][ind]
				else:
					val.latest = v
					val.count   = v
			ret[i[ind]] = val
	
	def fillVecFilter(self, ret, i, v, element):
		filteredValues = [-8000, 8000, 3, 2, 5, 4]
		filteredInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		tmp = filteredInds[:]
		for ind in tmp:
			if ind >= len(ret):
				filteredInds.remove(ind)
				
		for ind in range(len(i)):
			# make sure we don't override existing elements
			try:
				filteredInds.remove(i[ind])
			except ValueError:
				pass
				
			if isinstance(element, (float, int, long)):
				if type(v) != int and type(v) != float:
					val = v[ind]
				else:
					val = v
				ret[i[ind]] = val
				
				# make sure we don't filter out actual values
				if filteredValues.count(val) > 0:
					filteredValues.remove(val)
			elif isinstance(element, Obj1):
				val = pcb.Obj1()
				if type(v) == tuple:
					val.weight = v[0][ind]
					val.category = v[1][ind]
				else:
					val.weight = v
					val.category = v
				ret[i[ind]] = val

				# make sure we don't filter out actual values
				if filteredValues.count(val.weight) > 0:
					filteredValues.remove(val.weight)
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				if type(v) == tuple:
					val.latest = v[0][ind]
					val.count = v[1][ind]
				else:
					val.latest = v
					val.count = v
				ret[i[ind]] = val

				# make sure we don't filter out actual values
				if filteredValues.count(val.latest) > 0:
					filteredValues.remove(val.latest)
		# make sure we don't override existing elements
		if len(filteredInds) == 0:
			if ret.nnn() == len(ret):
				# the test uses a completely full vector, so skip filtering
				return
			else:
				# if this fails then fix the test so at least one filtered value gets added
				self.assertTrue(len(filteredInds) > 0)
		# add extra elements
		for ind in range(len(filteredInds)):
			fv = filteredValues[ind % len(filteredValues) ]
			if isinstance(element, Obj1):
				val = pcb.Obj1()
				val.weight = fv
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				val.latest = fv
			else:
				val = fv
			ret[filteredInds[ind]] = val
		# add the filter to take out the extra elements we just added,
		# so tests should pass as if we didn't do anything.
		if ret.isObj():
			ret.addFilter(lambda e: filteredValues.count(e.weight) == 0)
		else:
			ret.addFilter(lambda e: filteredValues.count(e) == 0)

	testFilter = False
	def initializeSpVec(self, length, i, v=1, element=0):
		ret = Vec(length, element=element, sparse=True)
		self.fillVec(ret, i, v, element)
		return ret
 
	def initializeVec(self, length, i, v=1, element=0):
		ret = Vec(length, element=element, sparse=False)
		self.fillVec(ret, i, v, element)
		return ret

	def assertEqualVec(self, vec, expV, expT=None, nn=()):
		self.assertEqual(len(vec), len(expV))
		for i in range(len(vec)):
			if vec[i] is None and expV[i] == 0 and i not in nn:
				pass
			else:
				if vec.isObj():
					self.assertEqual(vec[i].weight, expV[i])
					if expT is not None:
						self.assertEqual(vec[i].category, expT[i])
				else:
					self.assertEqual(vec[i], expV[i])
	
	def assertAlmostEqualVec(self, vec, expV, expT=None, nn=()):
		self.assertEqual(len(vec), len(expV))
		for i in range(len(vec)):
			if vec[i] is None and expV[i] == 0 and i not in nn:
				pass
			else:
				if vec.isObj():
					self.assertAlmostEqual(vec[i].weight, expV[i])
					if expT is not None:
						self.assertAlmostEqual(vec[i].category, expT[i])
				else:
					self.assertAlmostEqual(vec[i], expV[i])

class ConstructorTests(SpVecTests):
	def test_SpVecDint_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		vec = self.initializeSpVec(sz, i, element=0)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqualVec(vec, expI, nn=i)

	def test_SpVecObj1_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, element=element)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expI, expI, nn=i)

	def test_SpVecObj2_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, element=element)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expI, nn=i)

	def test_SpVecDint_null(self):
		sz = 25
		i = []
		element = 0.0
		vec = self.initializeSpVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		#self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expI, expI)

	def test_SpVecObj1_null(self):
		sz = 25
		i = []
		element = Obj1()
		vec = self.initializeSpVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		#self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expI, expI)

	def test_SpVecObj2_null(self):
		sz = 25
		i = []
		element = Obj2()
		vec = self.initializeSpVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		#self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expI, expI)

	def test_SpVecDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = 0.0
		vec = self.initializeSpVec(sz, i, weight, element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj2(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj1_all_weight_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [2, 2, 2, 5, 5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecDint_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		element = 0.0
		vec = self.initializeSpVec(sz, i, weight, element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj1_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj2_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecDint_spOnes(self):
		sz = 25
		i = [0, 2, 4, 6, 9, 10]
		weight = [1, 4, 8, 12, 18, 20]
		category =   [0, 0, 0, 0, 0, 0]
		element = 0.0
		vec = self.initializeSpVec(sz, i, weight, element=element)
		vec.spOnes()
		expW = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj1_spOnes(self):
		sz = 25
		i = [0, 2, 4, 6, 9, 10]
		weight = [1, 4, 8, 12, 18, 20]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		vec.spOnes()
		expW = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

	def test_SpVecObj2_spOnes(self):
		sz = 25
		i = [0, 2, 4, 6, 9, 10]
		weight = [1, 4, 8, 12, 18, 20]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		vec.spOnes()
		expW = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertAlmostEqualVec(vec, expW, expT)

	def test_SpVecDint_spRange(self):
		sz = 25
		i = [0, 2, 4, 6, 9, 10]
		weight = [1, 4, 8, 12, 8, 2]
		vec = self.initializeSpVec(sz, i, weight)
		vec.spRange()
		expW = [0, 0, 2, 0, 4, 0, 6, 0, 0, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertAlmostEqualVec(vec, expW)

	def test_SpVecObj1_spRange(self):
		sz = 25
		i = [0, 2, 4, 6, 9, 10]
		weight = [1, 4, 8, 12, 8, 2]
		category =   [2, 2, 2, 3, 5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		vec.spRange()
		expW = [0, 0, 2, 0, 4, 0, 6, 0, 0, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 3, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqualVec(vec, expW, expT, nn=i)

##	def test_SpVec_zeros(self):
##		sz = 25
##		vec = SpVec(sz)
##		expI = 0
##		self.assertEqual(sz, len(vec))
##		for ind in range(sz):
##			self.assertEqual(expI,vec[ind])
##
#	def test_SpVec_ones_simple(self):
#		sz = 25
#		vec = Vec.ones(sz, sparse=True)
#		expI = 1
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expI,vec[ind])
#
#	def test_SpVec_ones_Obj1(self):
#		sz = 25
#		vec = SpVec.ones(sz, element=pcb.Obj1())
#		expI = 1
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expI,vec[ind].weight)

	def test_SpVecDint_range_simple(self):
		sz = 25
		vec = Vec.range(sz, sparse=True)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind])

	def test_SpVecObj1_range(self):
		sz = 25
		vec = Vec.range(sz, element=pcb.Obj1(), sparse=True)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind].weight)

	def test_SpVecDint_range_offset_simple(self):
		sz = 25
		offset = 7
		vec = Vec.range(offset, stop=sz+offset, sparse=True)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind+offset,vec[ind])

	def test_SpVecObj1_range_offset(self):
		sz = 25
		offset = 7
		vec = Vec.range(offset, stop=sz+offset, element=pcb.Obj1(), sparse=True)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind+offset,vec[ind].weight)

	def test_SpVec_range_offset_simple(self):
		sz = 25
		offset = -13
		vec = Vec.range(offset, stop=sz+offset, sparse=True)
		expI = 1
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind]-offset)
			#self.assertEqual(ind,vec[ind].weight-offset)

	def test_SpVecObj_range_offset(self):
		sz = 25
		offset = -13
		vec = Vec.range(offset, stop=sz+offset, element=pcb.Obj1(), sparse=True)
		expI = 1
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind].weight-offset)

#	def test_SpVec_toBool(self):
#		sz = 25
#		i = [0,	  2,		4,  6, 8, 10, 12]
#		w = [1, .00001, -.000001, -1, 0, 24, 3.1415962]
#		t = [2, 2,  2,  5,  5,  5, 5]
#		vec = self.initializeSpVec(sz, i, (w, t))
#		vec.toBool()
#		expVec = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#				0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expVec[ind],vec[ind].weight)

	def disabled_for_builtin_types___test_copy_elementArg_Obj1toDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element1 = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element1)
		element2 = 0.0
		vec2 = vec.copy(element=element2)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element2),type(vec2[0]))
		self.assertEqualVec(vec2, expW, nn=i)

	def disabled_for_builtin_types___test_copy_elementArg_Obj2toDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element1 = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element1)
		element2 = 0.0
		vec2 = vec.copy(element=element2)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element2),type(vec2[0]))
		self.assertEqualVec(vec2, expW, nn=i)

class xxxTests(SpVecTests):
		pass

class BuiltInTests(SpVecTests):
	def test_len_vectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		res = len(vec)
		self.assertEqual(sz, res)

	def test_len_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = len(vec)
		self.assertEqual(sz, res)

	def test_nnn_vectorDint_noZeros(self):
		sz = 25
		i = [1, 2, 4, 6, 8, 10]
		weight = [777, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		res = vec.nnn()
		expRes = 6
		self.assertEqual(expRes, res)

	def test_nnn_vectorDint_elem3Zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [777, -4, 16, 0, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		res = vec.nnn()
		expRes = 6
		self.assertEqual(expRes, res)

	def disabled_for_builtin_types___test_nnn_vectorObj1_elem3Zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [777, -4, 16, 0, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.nnn()
		expRes = 6
		self.assertEqual(expRes, res)

	def test_nnn_vectorDint_noElem0(self):
		sz = 25
		i = [1, 2, 4, 6, 8, 10]
		weight = [777, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		res = vec.nnn()
		expRes = 6
		self.assertEqual(expRes, res)

	def disabled_for_builtin_types___test_nnn_vectorObj1_noElem0(self):
		sz = 25
		i = [1, 2, 4, 6, 8, 10]
		weight = [777, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.nnn()
		expRes = 6
		self.assertEqual(expRes, res)

	def test_add_vectorDint_vectorDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		vec1 = self.initializeSpVec(sz, i1, w1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		vec2 = self.initializeSpVec(sz, i2, w2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def test_add_vectorDint_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		res = vec + 3.07
		self.assertAlmostEqualVec(res, expW)

	def test_add_vectorObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		res = vec + 3.07
		self.assertAlmostEqualVec(res, expW, expT)

	def test_add_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0,  3,   6,   9,   12,   15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element1 = Obj1()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element1)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def disabled_for_builtin_types___test_add_vectorObj1_vectorObj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj2()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def test_add_vectorObj2_vectorObj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj2()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj2()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def disabled_for_builtin_types___test_add_vectorObj1_vectorDint(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [2, 2,  3,  3,  7,   7]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0,  3,   6,   9,   12,   15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		vec2 = self.initializeSpVec(sz, i2, w2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def disabled_for_builtin_types___test_add_vectorDint_vectorObj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		vec1 = self.initializeSpVec(sz, i1, w1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [2, 2, 3, 3,  7,  7, 23]
		element2 = Obj2()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)
		
	def test_add_vectorObj1Filt_vectorObj1Filt(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		#for ind in range(sz):
		#	self.assertEqual(vecExpected[ind], vec3[ind].weight)
		self.assertEqualVec(vec3, vecExpected)

	def test_add_vectorObj1Filt_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
		# ---- commented----  vec2.addFilter(geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [-2, 0, 1, 0, 3, 0, 5, 0, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		self.assertAlmostEqualVec(vec3, vecExpected)

	def test_add_vectorObj1_vectorObj1Filt(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
		# ----- commented out--- vec1.addFilter(ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		self.assertAlmostEqualVec(vec3, vecExpected)

	def test_bitwiseAnd_vectorDint_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		vec1 = self.initializeSpVec(sz, i1, w1)
		scalar = 0x7
		vec3 = vec1 & scalar
		expI = [0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqualVec(vec3, expI)
		
	def test_bitwiseAnd_vectorDint_vectorDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F]
		vec1 = self.initializeSpVec(sz, i1, w1)
		i2 = [0, 2, 4, 6, 8, 10, 18]
		w2 = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
		vec2 = self.initializeSpVec(sz, i2, w2)
		vec3 = vec1 & vec2
		expI = [0x1, 0, 0x2, 0, 0x4, 0, 0x8, 0, 0x10, 0, 0x20, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqualVec(vec3, expI)

	def disabled_for_builtin_types___test_bitwiseAnd_vectorObj1_vectorObj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F]
		c1 = [2, 2,  2,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, c1), element=element1 )
		i2 = [0, 2, 4, 6, 8, 10, 18]
		w2 = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
		c2 = [2, 2, 2, 2,  2,  2,  2]
		element2 = Obj2()
		vec2 = self.initializeSpVec(sz, i2, (w2, c2), element=element2 )
		vec3 = vec1 & vec2
		expI = [0x1, 0, 0x2, 0, 0x4, 0, 0x8, 0, 0x10, 0, 0x20, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(max(len(i1), len(i2)),vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def test_abs_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertAlmostEqualVec(res, expW)

	def test_iadd_vectorDint_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec += 3.07
		self.assertEqual(sz, len(vec))
		self.assertAlmostEqualVec(vec, expW)

	def test_iadd_vectorObj_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		vec += 3.07
		self.assertEqual(sz, len(vec))
		self.assertAlmostEqualVec(vec, expW)

	def test_iadd_vectorDint_vectorDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		vec1 = self.initializeSpVec(sz, i1, w1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		vec2 = self.initializeSpVec(sz, i2, w2)
		vec1 += vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
		self.assertEqualVec(vec1, expI)

	def test_invert_vectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [-1, 0, 3, 0, -17, 0, 35, 0, 63, 0, -101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		res = ~vec
		self.assertEqualVec(res, expW)

	def test_invert_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [-1, 0, 3, 0, -17, 0, 35, 0, 63, 0, -101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		res = ~vec
		self.assertEqualVec(res, expW)

	def test_sub_vectorDint_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [-3.07, 0, -7.07, 0, 12.93, 0, -39.07, 0, -67.07, 0, 96.93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		vec2 = vec - 3.07
		self.assertEqualVec(vec2, expW)

	def test_sub_vectorDint_vectorDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj1()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 - vec2
		expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0, -3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expI)

	def test_sub_vectorObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [-3.07, 0, -7.07, 0, 12.93, 0, -39.07, 0, -67.07, 0, 96.93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		vec2 = vec - 3.07
		self.assertEqualVec(vec2, expW, expT)

	def test_sub_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj1()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 - vec2
		expW = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0, -3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		self.assertEqualVec(vec3, expW)

class BuiltInTests_disabled(SpVecTests):
	def test_iadd_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element = Obj1()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element)
		vec1 += vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_isub_vectorObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeSpVec(sz, i, (weight, category))
		expW = [-3.07, 0, -7.07, 0, 12.93, 0, -39.07, 0, -67.07, 0, 96.93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec -= 3.07
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_isub_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec1 -= vec2
		expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0, -3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_mul_vectorObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeSpVec(sz, i, (weight, category))
		expW = [0, 0, -13, 0, 52, 0, -117, 0, -208, 0, 325, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec *= 3.25
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_mul_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec1 *= vec2
		expI = [0, 0, 0, 0, 0, 0, 36*216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(i1)+len(i2)-2,vec1.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_div_scalarObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeSpVec(sz, i, (weight, category))
		expW = [0.0, 0, -1.3333333, 0, 5.3333333, 0, -12.0, 0, -21.33333333, 0, 
				33.3333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = vec / 3
		#self.assertEqual(len(i), vec2.nnn())
		self.assertEqual(sz, len(res))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_div_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		self.assertRaises(NotImplementedError, vec1.__div__, vec1, vec2)
		return
		#NOTE:  dead code below; numbers not yet arranged to work properly
		vec3 = vec1 / vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_mod_vectorObj1_scalarDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4.25, 16.5, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeSpVec(sz, i, (weight, category))
		expW = [0.0, 0, 1.75, 0, 1.5, 0, 0, 0, 2, 0, 
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = vec % 3
		#self.assertEqual(len(i), vec2.nnn())
		self.assertEqual(sz, len(res))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_mod_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		self.assertRaises(NotImplementedError, vec1.__mod__, vec1, vec2)
		return
		#NOTE:  dead code below; numbers not yet arranged to work properly
		vec3 = vec1 % vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseAnd_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj1()
		vec2 = self.initializeSpVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 & vec2
		expI = [0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(type(element1),type(vec3[0]))
		self.assertEqual(sz, len(vec3))
		#print "test_BW_and: vec3.nnn=", vec3.nnn()
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseOr_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec3 = vec1 | vec2
		expI = [1, 0, 4, 27, 16, 0, 253, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseXor_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec3 = vec1 ^ vec2
		expI = [1, 0, 4, 27, 16, 0, 228, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_logicalOr_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec3 = vec1.logicalOr(vec2)
		expI = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
				0, 0, 1, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_logicalXor_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		vec3 = vec1.logicalXor(vec2)
		expI = [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
				0, 0, 1, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		self.assertEqual(len(i1)+len(i2)-2,vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

#	def test_indexing_RHS_SpVec_scalar(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 9
#		value = vec1[ndx]
#		self.assertEqual(-3, value)
#
#	def test_indexing_RHS_SpVec_scalar_outofbounds(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 7777
#		self.assertRaises(IndexError, SpVec.__getitem__, vec1, ndx)
#		#value = vec1[ndx]
#		#self.assertEqual(-3, value)
#
#	def test_indexing_RHS_SpVec_scalar_outofbounds2(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = -333
#		self.assertRaises(IndexError, SpVec.__getitem__, vec1, ndx)
#		#value = vec1[ndx]
#		#self.assertEqual(-3, value)
#
#	def test_indexing_RHS_SpVec_SpVec(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 4
#		ndxI = [0, 1, 2,  3]
#		ndxV = [1, 4, 9, 16]
#		ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#		self.assertRaises(KeyError, SpVec.__getitem__, vec1, ndx)
#		#expI = [-11, -8, -3, 4]
#		#self.assertEqual(ndxLen, len(vec3))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec3[ind])
#
#	def test_indexing_RHS_SpVec_Vec(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 4
#		ndx = Vec(4)
#		ndx[0] = 1
#		ndx[1] = 4
#		ndx[2] = 9
#		ndx[3] = 16
#		vec3 = vec1[ndx]
#		expI = [-11, -8, -3, 4]
#		self.assertEqual(ndxLen, len(vec3))
#		for ind in range(ndxLen):
#			self.assertEqual(expI[ind], vec3[ind])
#
#	def test_indexing_RHS_SpVec_booleanSpVec(self):
#		sz = 25
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 25
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#				18, 19, 20, 21, 22, 23, 24]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#				0, 0, 0, 0, 0]
#		ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#		self.assertRaises(KeyError, SpVec.__getitem__, vec1, ndx)
#		#vec3 = vec1[ndx]
#		#expI = [1,16,81,256]
#		#self.assertEqual(ndxTrue, len(vec3))
#		#for ind in range(ndxTrue):
#		#	self.assertEqual(expI[ind], vec3[ind])
#
#	def test_indexing_LHS_SpVec_booleanVec_scalar(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndx = Vec(sz)
#		ndx[1] = 1
#		ndx[4] = 1
#		ndx[9] = 1
#		ndx[16] = 1
#		vec1[ndx] = 77
#		expI = [-9, 77, -7, -6, 77, -4, -3, -2, -1, 77, 1, 2, 3, 4, 5, 6, 77, 8]
#		self.assertEqual(sz, len(vec1))
#		for ind in range(ndxLen):
#			self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_SpVec_nonbooleanVec_scalar(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndx = Vec(sz)
#		ndx[1] = 7
#		ndx[4] = 11
#		ndx[9] = 1
#		ndx[16] = 5
#		self.assertRaises(KeyError, SpVec.__setitem__, vec1, ndx, 77)
#		#vec1[ndx] = 77
#		#expI = [-9, 77, -7, -6, 77, -4, -3, -2, -1, 77, 1, 2, 3, 4, 5, 6, 77, 8]
#		#self.assertEqual(sz, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_SpVec_scalar_scalar(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 11
#		value = 777
#		vec1[ndx] = value
#		expI = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 777, 3, 4, 5, 6, 7, 8]
#		self.assertEqual(sz, len(vec1))
#		self.assertEqual(len(expI), len(vec1))
#		for ind in range(sz):
#			self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_SpVec_booleanSpVec_scalar(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#		ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#		value = 777
#		self.assertRaises(KeyError,SpVec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 777, 4, 9, 777, 25, 36, 49, 64, 777, 100, 121, 144, 169, 
#		#		196, 225, 777, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_SpVec_booleanSpVec_SpVec(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#		ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#		valueV = [0, 1, 7, 7, 0.25, 0, 0, 0, 0, 0.111, 0, 0, 0, 0, 0, 0, 0.0625,
#				0, 0, 0, 0, 7, 0, 0, 0]
#		value = self.initializeSpVec(ndxLen, ndxI, valueV)
#		self.assertRaises(KeyError,SpVec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#		#		196, 225, 0.0625, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_SpVec_Vec(self):
#		sz = 18
#		vec1 = SpVec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 4
#		ndxI = [0, 1, 2, 3]
#		ndxV = [1, 4, 9, 16]
#		ndx = self.initializeSpVec(ndxLen, ndxI, ndxV)
#		valueV = [1, 0.25, 0.111, 0.0625]
#		value = self.initializeSpVec(ndxLen, ndxI, valueV)
#		self.assertRaises(IndexError,SpVec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#		#		196, 225, 0.0625, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
	def test_eq_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		eqY = vec1 == scalar
		expV = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(eqY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], eqY[ind].weight)

	def test_eq_vectorObj1_vectorDint(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8,  10,  12]
		w1 = [0, 4, 16, 36, 64, 100, 144]
		t1 = [1, 1,  1,  2,  2,   2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 2,   6,   9,  12,   15,   18]
		w2 = [1, 4, 216, 729, 144, 3375, 5832]
		t2 = [1, 1,   1,   2,   2,	2,	2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		eqY = vec1 == vec2
		expV = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(eqY), len(vec1))
		self.assertEqual(len(eqY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], eqY[ind].weight)

	def test_ne_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		neY = vec1 != scalar
		expV = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(neY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], neY[ind].weight)

	def test_ne_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8,  10,  12]
		w1 = [0, 4, 16, 36, 64, 100, 144]
		t1 = [1, 1,  1,  2,  2,   2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 2,   6,   9,  12,   15,   18]
		w2 = [1, 4, 216, 729, 144, 3375, 5832]
		t2 = [1, 1,   1,   2,   2,	2,	2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		neY = vec1 != vec2
		expV = [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(neY), len(vec1))
		self.assertEqual(len(neY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], neY[ind].weight)

	def test_ge_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		geY = vec1 >= scalar
		expV = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(geY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], geY[ind].weight)

	def test_ge_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		geY = vec1 >= vec2
		expV = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(geY), len(vec1))
		self.assertEqual(len(geY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], geY[ind].weight)

	def test_gt_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		gtY = vec1 > scalar
		expV = [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(gtY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], gtY[ind].weight)

	def test_gt_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		gtY = vec1 > vec2
		expV = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(gtY), len(vec1))
		self.assertEqual(len(gtY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], gtY[ind].weight)

	def test_le_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		leY = vec1 <= scalar
		expV = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(leY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], leY[ind].weight)

	def test_le_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		leY = vec1 <= vec2
		expV = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(leY), len(vec1))
		self.assertEqual(len(leY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], leY[ind].weight)

	def test_lt_vectorObj1_scalarDint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		scalar = 4
		ltY = vec1 < scalar
		expV = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(ltY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], ltY[ind].weight)

	def test_lt_vectorObj1_vectorObj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeSpVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeSpVec(sz, i2, (w2, t2))
		ltY = vec1 < vec2
		expV = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(ltY), len(vec1))
		self.assertEqual(len(ltY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], ltY[ind].weight)



class GeneralPurposeTests(SpVecTests):
	def test_abs_vectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqualVec(res, expW)

	def test_abs_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertEqualVec(res, expW, expT)

	def test_abs_vectorObj2(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertEqualVec(res, expW, expT)

	def test_abs_vectorDint_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [4, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)

		filter = lambda x: abs(x) >= 0 and abs(x) < 5
		vec.addFilter(filter)
		expW = [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		vec.delFilter(filter)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqualVec(res, expW)

	def test_abs_vectorDint_filtered_inplace(self):
		sz = 25
		i = [0, 1, 2, 4, 6, 8, 10]
		weight = [0, 4, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		filter = lambda x: abs(x) >= 0 and abs(x) < 5
		vec.addFilter(filter)
		expW = [0, 4, 4, 0, 16, 0, -36, 0, -64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec.apply(lambda x: abs(x))
		vec.delFilter(filter)
		self.assertEqual(type(vec[0]), type(vec[0]))
		self.assertEqualVec(vec, expW)

	def test_abs_vectorObj1_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -2, 1, 3, -64, 100]
		category = [2, 2, 2, 2, 2, 2]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		vec.addFilter(geM2lt4)
		expW = [0, 0, 2, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqualVec(res, expW)

	def test_abs_vectorObj2_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 64, 1, 3, -2, -100]
		category = [2, 2, 2, 2, 2, 2]
		element = Obj2()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		vec.addFilter(geM2lt4)
		expW = [0, 0, 0, 0, 1, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqualVec(res, expW)

	def test_all_vectorDint_all_nonnull_all_true(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)
		res = vec.all()
		self.assertEqual(True, res)

	def test_all_vectorDint_all_nonnull_one_false(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)
		vec[4] = False
		res = vec.all()
		self.assertEqual(False, res)

	def test_all_vectorDint_all_nonnull_one_true(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)-1
		vec[4] = True
		res = vec.all()
		self.assertEqual(False, res)

	def test_all_vectorDint_all_nonnull_all_false(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)-1
		res = vec.all()
		self.assertEqual(False, res)

	def test_all_vectorDint_one_nonnull_one_false(self):
		sz = 10
		vec = Vec(sz, sparse=True)
		vec[4] = 0
		res = vec.all()
		self.assertEqual(False, res)

	def test_all_vectorDint_one_nonnull_one_true(self):
		sz = 10
		vec = Vec(sz, sparse=True)
		vec[4] = 1
		res = vec.all()
		self.assertEqual(True, res)

	def test_all_vectorDint_three_nonnull_one_true(self):
		sz = 10
		vec = Vec(sz, sparse=True)
		vec[0] = 0
		vec[2] = 0
		vec[4] = 1
		res = vec.all()
		self.assertEqual(False, res)

	def test_all_vectorDint_three_nonnull_three_true(self):
		sz = 10
		vec = Vec(sz, sparse=True)
		vec[0] = 1
		vec[2] = 1
		vec[4] = 1
		res = vec.all()
		self.assertEqual(True, res)

	def disabled_test_all_vectorObj1_all_nonnull_all_true(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [777, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.all()
		expRes = True
		self.assertEqual(expRes, res)

	def disableObjAll_test_all_vectorObj1_no_elem0(self):
		sz = 6
		i = [1, 2, 3, 4, 5]
		weight = [-4, 16, 0, -64, 100]
		category = [2, 5, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.all()
		expRes = False
		self.assertEqual(expRes, res)

	def disableObjAll_test_all_vectorObj1_no_elem3(self):
		sz = 6
		i = [0, 1, 2, 4, 5]
		weight = [-1, 1, -4, -64, 100]
		category = [2, 2, 5,  -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.all()
		expRes = True
		self.assertEqual(expRes, res)

	def disableObjAll_test_all_vectorObj1_evens_nonnull_elem2_false(self):
		sz = 6
		i = [0, 2, 4]
		weight = [1, 0, -64]
		category = [2, 2, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.all()
		expRes = False
		self.assertEqual(expRes, res)

	def disableObjAll_test_any_vectorObj1_all_nonnull_all_true(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [777, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def disableObjAll_test_any_vectorObj1_all_nonnull_elem0_false(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def disableObjAll_test_any_vectorObj1_all_nonnull_elem3_false(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [1, -4, 16, 0, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def disableObjAll_test_any_vectorObj1_evens_nonnull_elem2_false(self):
		sz = 6
		i = [0, 2, 4]
		weight = [1, 0, -64]
		category = [2, 2, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def test_any_vectorObj1_all_true(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)
		res = vec.any()
		self.assertEqual(True, res)

	def test_any_vectorObj1_one_false(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)
		vec[4] = False
		res = vec.any()
		self.assertEqual(True, res)

	def test_any_vectorObj1_one_true(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)-1
		vec[4] = True
		res = vec.any()
		self.assertEqual(True, res)

	def test_any_vectorObj1_all_false(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)-1
		res = vec.any()
		self.assertEqual(False, res)

	def test_sum_vectorDint_nulls(self):
		sz = 10
		vec = Vec(sz, sparse=True)
		res = vec.sum()
		self.assertEqual(0.0, res)

	def test_sum_vectorDint_ones(self):
		sz = 10
		vec = Vec.ones(sz, sparse=True)
		res = vec.sum()
		self.assertEqual(sz, res)

	def test_sum_vectorDint_range(self):
		sz = 10
		vec = Vec.range(sz, sparse=True)
		res = vec.sum()
		self.assertEqual((sz*(sz-1))/2, res)

	def test_sum_vectorDint_range2(self):
		sz = 11
		vec = Vec.range(-(sz/2), stop=(sz/2)+1, sparse=True)
		res = vec.sum()
		self.assertEqual(0, res)

	def test_sum_vectorDint_fixed(self):
		sz = 11
		vec = Vec(sz, sparse=True)
		vec[2] = 23
		vec[4] = 9
		vec[5] = -32
		res = vec.sum()
		self.assertEqual(0, res)

	def test_set_vectorObj1_scalarObj1(self):
		sz = 25
		k = 3.7
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		
		def f(x):
			x.weight=k
			return x
		vec.apply(f)
		self.assertEqualVec(vec, expW, expT)

	def test_negate_vectorDint(self):
		#print "\n\n\t\t***in test_negate_dint\n\n"
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [0, 0, 4, 0, -16, 0, 36, 0, 64, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = -vec
		self.assertEqualVec(res, expW)

	def test_negate_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, -16, 0, 36, 0, 64, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		res = -vec
		self.assertEqualVec(res, expW)

	def test_max_vectorObj1_some_nonnull_maxElem0(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [123, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.max(Obj1.maxInit())
		expRes = 123
		self.assertEqual(expRes, res.weight)

	def test_max_vectorObj1_all_nonnull_maxElem4(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.max(Obj1.maxInit())
		expRes = 136
		self.assertEqual(expRes, res.weight)

	def test_max_vectorObj1_some_nonnull_maxElem6(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.max(Obj1.maxInit())
		expRes = 136
		self.assertEqual(expRes, res.weight)

	def test_min_vectorObj1_some_nonnull_minElem0(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [-123, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.min(Obj1.minInit())
		expRes = -123
		self.assertEqual(expRes, res.weight)

	def test_min_vectorObj1_all_nonnull_minElem4(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.min(Obj1.minInit())
		expRes = -64
		self.assertEqual(expRes, res.weight)

	def test_min_vectorObj1_some_nonnull_minElem8(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		res = vec.min(Obj1.minInit())
		expRes = -64
		self.assertEqual(expRes, res.weight)

	def test_sort_vectorObj1(self):
		if SpVecTests.testFilter:
			return
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [-64, 0, -36, 0, -4, 0, 0, 0, 16, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [-5, 0, 5, 0, -2, 0, 2, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		vec.sort()
		self.assertEqualVec(vec, expW, expT)

	def disabled_test_sorted_vector(self):
		if SpVecTests.testFilter:
			return
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [-64, 0, -36, 0, -4, 0, 0, 0, 16, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expNdx = [8, 0, 6, 0, 2, 0, 0, 0, 4, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		(vec2, vec3) = vec.sorted()
		self.assertEqualVec(vec2, expW)
		self.assertEqualVec(vec3, expNdx)

	def disabled_test_sorted_vectorObj1(self):
		if SpVecTests.testFilter:
			return
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [-64, 0, -36, 0, -4, 0, 0, 0, 16, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expT = [-5, 0, 5, 0, -2, 0, 2, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		expNdx = [8, 0, 6, 0, 2, 0, 0, 0, 4, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0]
		(vec2, vec3) = vec.sorted()
		self.assertEqualVec(vec2, expW, expT)
		self.assertEqualVec(vec3, expNdx)

	def disabled_test_topK_vectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeSpVec(sz, i, weight)
		expW = [100, 16, 0]
		topKSz = 3
		vec2 = vec.topK(topKSz)
		self.assertEqual(topKSz, len(vec2))
		for ind in range(topKSz):
			self.assertAlmostEqual(expW[ind],vec2[ind])

class GeneralPurposeTests_disabled(SpVecTests):
	def test_load(self):
		vec = Vec.load('UFget/Pajek/CSphd/CSphd.mtx')
		return

	def test_topK_vectorObj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeSpVec(sz, i, (weight, category), element=element)
		expW = [100, 16, 0]
		expT = [5, 2, 2]
		topKSz = 3
		vec2 = vec.topK(topKSz)
		self.assertEqual(topKSz, len(vec2))
		for ind in range(topKSz):
			self.assertAlmostEqual(expW[ind],vec2[ind].weight)
			self.assertEqual(expT[ind], vec2[ind].category)


class MixedDenseSparseVecTests(SpVecTests):
		pass

class MixedDenseSparseVecTests_disabled(SpVecTests):
	def test_add_sparseDint_denseDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-2, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec + vec2
		expI = [0, -2, 4, 0.1, 8, 777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]

		self.assertEqual(sz, len(vec3))
		# odd behavior here; element 0 in vec1 and element 7 in vec2 become
		# nulls in vec3 (!)
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_add_denseDint_sparseDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-2, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec2 + vec
		expI = [0, -2, 4, 0.1, 8, 777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec3))
		#self.assertEqual(len(i)+len(i2)-1, vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_subtract_sparseDint_denseDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-2, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec - vec2
		expI = [0, 2, 4, -0.1, 8, -777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec3))
		# odd behavior here; element 0 in vec1 and element 7 in vec2 become
		# nulls in vec3 (!)
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_isubtract_sparseDint_denseDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-2, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec.copy()
		vec3 -= vec2
		expI = [0, 2, 4, -0.1, 8, -777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec3))
		# odd behavior here; element 0 in vec1 and element 7 in vec2 become
		# nulls in vec3 (!)
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_subtract_denseDint_sparseDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-2, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec2 - vec
		expI = [0, -2, -4, 0.1, -8, 777, -12, 0, -16, 0, -20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec3))
		#self.assertEqual(len(i)+len(i2)-1, vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

class ApplyReduceTests(SpVecTests):
	def test_vectorDint_apply(self):
		def ge0lt5(x):
				return x>=0 and x<5
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		vec.apply(ge0lt5)
		vecExpected = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqualVec(vec, vecExpected)

	def test_apply_vectorDint_pcbabs(self):
		sz = 25
		i = [0, 1, 2,  4,   6, 8, 10]
		v = [0, 4, -4, 8, -12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		vec.apply(op_abs)
		vecExpected = [0, 4, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqualVec(vec, vecExpected)

	def test_countvectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		ct = vec.count()
		ctExpected = 5
		self.assertEqual(ctExpected, ct)

	def test_countvectorDintPred(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		ct = vec.count(lambda x: x > 0 and x < 10)
		ctExpected = 2
		self.assertEqual(ctExpected, ct)

	def test_reduce_vectorDint_default_op(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		red = vec.reduce(op_add)
		redExpected = 60
		self.assertEqual(redExpected, red)

	def test_reduce_vectorDint_max(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		red = vec.reduce(op_max)
		redExpected = 20
		self.assertEqual(redExpected, red)

	def test_reduce_vectorDint_min(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [2, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		# disabled_test : used to use reduce directly, but without init attribute (somewhat incorrect)
		#red = vec.reduce(op_min)
		red = vec.min()
		redExpected = 2
		self.assertEqual(redExpected, red)

class ApplyReduceTests_disabled(SpVecTests):
	def test_apply_vectorDint(self):
		def ge0lt5(x):
				return x>=0 and x<5
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		vec._apply(ge0lt5)
		vecExpected = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqualVec(vec, vecExpected)

	def test_apply_vectorDint_pcbabs(self):
		sz = 25
		i = [0, 2,  4,   6, 8, 10]
		v = [0, -4, 8, -12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		vec._apply(pcb.abs())
		vecExpected = [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqualVec(vec, vecExpected)

	def test_count_vectorDint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		ct = vec.count()
		ctExpected = 6
		self.assertEqual(ctExpected, ct)

	def test_reduce_vectorDint_default_op(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		red = vec._reduce(SpVec.op_add)
		redExpected = 60
		self.assertEqual(redExpected, red)

	def test_reduce_vectorDint_max(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		red = vec._reduce(SpVec.op_max)
		redExpected = 20
		self.assertEqual(redExpected, red)

	def test_reduce_vectorDint_min(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [2, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		red = vec._reduce(SpVec.op_min)
		redExpected = 2
		self.assertEqual(redExpected, red)

		
class FilterTests(SpVecTests):
#	def test_apply_filter(self):
#		def add5(x):
#				if isinstance(x, (int, long, float)):
#						return x+5
#				else:
#						x.weight += 5
#						x.category *= 2
#						return x
#		sz = 25
#		i = [0, 2, 4, 6, 8, 10]
#		v = [1, 4, 8,12,16, 20]
#		c = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec = self.initializeSpVec(sz, i, (v,v), element=element)
#		vec.addFilter(ge0lt5)
#		vec._apply(add5)
#		vecExpected = [6, 0, 9, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec[ind].weight)
#
#	def test_apply_filters(self):
#		def add3p14(x):
#				if isinstance(x, (int, long, float)):
#						return x+3.14159
#				else:
#						x.weight += 3.14159
#						x.category *= 2
#						return x
#		sz = 25
#		i = [ 0, 2, 4, 6, 8, 10]
#		v = [-3, 2, 3, 4, 5,  6]
#		c = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec = self.initializeSpVec(sz, i, (v,v), element=element)
#		vec.addFilter(ge0lt5)
#		vec.addFilter(geM2lt4)
#		vec._apply(add3p14)
#		vecExpected = [-3, 0, 5.14159, 0, 6.14159, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec[ind].weight)
#
#	def test_delete_all_filters(self):
#		# add filters, then delete before _apply
#		def add3p14(x):
#				if isinstance(x, (int, long, float)):
#						return x+3.14159
#				else:
#						x.weight += 3.14159
#						x.category *= 2
#						return x
#		sz = 25
#		i = [ 0, 2, 4, 6, 8, 10]
#		v = [-3, 2, 3, 4, 5,  6]
#		c = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec = self.initializeSpVec(sz, i, (v,v), element=element)
#		vec.addFilter(ge0lt5)
#		vec.addFilter(geM2lt4)
#		vec.delFilter()
#		vec._apply(add3p14)
#		vecExpected = [0.14159, 0, 5.14159, 0, 6.14159, 0, 7.14159, 0, 8.14159, 0, 9.14159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertAlmostEqual(vecExpected[ind], vec[ind].weight)
#
#	def test_apply2_deleteLast_filter(self):
#		def add5(x):
#				if isinstance(x, (int, long, float)):
#						return x+5
#				else:
#						x.weight += 5
#						x.category *= 2
#						return x
#		sz = 25
#		i = [0, 2, 4, 6, 8, 10]
#		v = [1, 4, 8,12,16, 20]
#		c = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec = self.initializeSpVec(sz, i, (v,v), element=element)
#		vec.addFilter(ge0lt5)
#		vec.addFilter(geM2lt4)
#		vec.delFilter(geM2lt4)
#		vec._apply(add5)
#		vecExpected = [6, 0, 9, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec[ind].weight)
#
#	def test_apply2_deleteFirst_filter(self):
#		def add5(x):
#				if isinstance(x, (int, long, float)):
#						return x+5
#				else:
#						x.weight += 5
#						x.category *= 2
#						return x
#		sz = 25
#		i = [0, 2, 4, 6, 8, 10]
#		v = [1, 4, 8,12,16, 20]
#		c = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec = self.initializeSpVec(sz, i, (v,v), element=element)
#		vec.addFilter(ge0lt5)
#		vec.addFilter(geM2lt4)
#		vec.delFilter(ge0lt5)
#		vec._apply(add5)
#		vecExpected = [6, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec[ind].weight)
#
#	def test_eWiseApply_filterBoth(self):
#		sz = 25
#		i1 = [0, 2, 4, 6, 8, 10]
#		v1 = [1, 2, 3, 4, 5,  4]
#		c1 = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
#		vec1.addFilter(ge0lt5)
#		i2 = [ 0, 2, 4, 6, 8, 10]
#		v2 = [-3,-1, 0, 1, 2,  5]
#		c2 = [ 2, 2, 7, 7, 3,  3]
#		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
#		vec2.addFilter(geM2lt4)
#		vec3 = vec1._eWiseApply(vec2, Obj1.__iadd__, True, True)
#		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec1))
#		self.assertEqual(sz, len(vec2))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec3[ind].weight)
#
#	def test_eWiseApply_filterFirst(self):
#		sz = 25
#		i1 = [0, 2, 4, 6, 8, 10]
#		v1 = [1, 2, 3, 4, 5,  4]
#		c1 = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
#		vec1.addFilter(ge0lt5)
#		i2 = [ 0, 2, 4, 6, 8, 10]
#		v2 = [-3,-1, 0, 1, 2,  5]
#		c2 = [ 2, 2, 7, 7, 3,  3]
#		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
#		# ---- commented----  vec2.addFilter(geM2lt4)
#		vec3 = vec1._eWiseApply(vec2, Obj1.__iadd__, True, True)
#		vecExpected = [-2, 0, 1, 0, 3, 0, 5, 0, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec1))
#		self.assertEqual(sz, len(vec2))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec3[ind].weight)
#
#	def test_eWiseApply_filterSecond(self):
#		sz = 25
#		i1 = [0, 2, 4, 6, 8, 10]
#		v1 = [1, 2, 3, 4, 5,  4]
#		c1 = [2, 2, 7, 7, 3,  3]
#		element = Obj1()
#		vec1 = self.initializeSpVec(sz, i1, (v1,v1), element=element)
#		# ----- commented out--- vec1.addFilter(ge0lt5)
#		i2 = [ 0, 2, 4, 6, 8, 10]
#		v2 = [-3,-1, 0, 1, 2,  5]
#		c2 = [ 2, 2, 7, 7, 3,  3]
#		vec2 = self.initializeSpVec(sz, i2, (v2,v2), element=element)
#		vec2.addFilter(geM2lt4)
#		vec3 = vec1._eWiseApply(vec2, Obj1.__iadd__, True, True)
#		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec1))
#		self.assertEqual(sz, len(vec2))
#		for ind in range(sz):
#			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_applyInd_vectorDint(self):
		def set_ind_indpInd(x,y):
				# x is the Obj1 instance;  y is the index of the element
				if isinstance(x, (int, long, float)):
						return y
				else:
						x.weight = y + y/100
						x.category = int(y)
						return x
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [1, 4, 8,12,16, 20]
		c1 = [2, 2, 7, 7, 3,  3]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1,c1), element=element1)
		vec1.applyInd(set_ind_indpInd)
		expW = [0, 0, 2.02, 0, 4.04, 0, 6.06, 0, 8.08, 0, 10.10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expC = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertAlmostEqualVec(vec1, expW, expC)

	def test_applyInd_vectorDintFilt(self):
		def set_ind_indpInd(x,y):
			# x is the Obj1 instance;  y is the index of the element
			if isinstance(x, (int, long, float)):
				return y
			else:
				x.weight = y + y/100
				x.category = int(y)
				return x
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [1, 4, 8,12,16, 20]
		c1 = [2, 2, 7, 7, 3,  3]
		element1 = Obj1()
		vec1 = self.initializeSpVec(sz, i1, (w1,c1), element=element1)
		vec1.addFilter(ge0lt5)
		vec1.applyInd(set_ind_indpInd)
		expW = [0, 0, 2.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expC = [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertAlmostEqualVec(vec1, expW, expC)


def runTests(verbosity = 1):
	testSuite = suite()
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

	print "running again using filtered data: (sort tests disabled because filtered sorting not supported yet)"
	SpVecTests.fillVec = SpVecTests.fillVecFilter
	SpVecTests.testFilter = True
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
	suite = unittest.TestSuite()
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MixedDenseSparseVecTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ApplyReduceTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(xxxTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(FilterTests))
	return suite

if __name__ == '__main__':
	runTests()
