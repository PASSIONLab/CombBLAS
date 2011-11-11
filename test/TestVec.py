import unittest
import math
from kdt import *
import kdt.ObjMethods

class VecTests(unittest.TestCase):
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
					val.weight = v[0][ind]
					val.category   = v[1][ind]
				else:
					val.weight = v
					val.category   = v
			ret[i[ind]] = val
	
	def fillVecFilter(self, ret, i, v, element):
		filteredValues = [-8000, 8000, 3, 2, 5, 4]
		filteredInds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
					val.weight = v[0][ind]
					val.category = v[1][ind]
				else:
					val.weight = v
					val.category = v
				ret[i[ind]] = val

				# make sure we don't filter out actual values
				if filteredValues.count(val.weight) > 0:
					filteredValues.remove(val.weight)
		# make sure we don't override existing elements
		self.assertTrue(len(filteredInds) > 0)
		# add extra elements
		for ind in range(len(filteredInds)):
			fv = filteredValues[ind % len(filteredValues) ]
			if isinstance(element, Obj1):
				val = pcb.Obj1()
				val.weight = fv
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				val.weight = fv
			else:
				val = fv
			ret[filteredInds[ind]] = val
		# add the filter to take out the extra elements we just added,
		# so tests should pass as if we didn't do anything.
		if ret.isObj():
			ret.addFilter(lambda e: filteredValues.count(e.weight) == 0)
		else:
			ret.addFilter(lambda e: filteredValues.count(e) == 0)

	def initializeSpVec(self, length, i, v=1, element=0):
		ret = Vec(length, element=element, sparse=True)
		self.fillVec(ret, i, v, element)
		return ret

#	def initializeSpVecFilter(self, length, i, v=1, element=0):
#		self.assertTrue(False) # implement this!
#		raise NotImplementedError, "todo sparse"
#		return self.initializeSpVec(self, length, i, v, element)
 
	def initializeVec(self, length, i, v=1, element=0):
		ret = Vec(length, element=element, sparse=False)
		self.fillVec(ret, i, v, element)
		return ret

	def asdfasdf_initializeVec(self, length, i, v=1, element=0):
		"""
		Initialize a Vec instance with values equal to one or the input value.
		"""
		ret = Vec(length, element=element, sparse=False)
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
					val.category = v[1][ind]
				else:
					val.weight = v
					val.category = v
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				if type(v) == tuple:
					val.weight = v[0][ind]
					val.category = v[1][ind]
				else:
					val.weight = v
					val.category = v
			ret[i[ind]] = val
		return ret

	def asfdasdf_initializeVecFilter(self, length, i, v=1, element=0):
		"""
		Initialize a Vec instance with values equal to one or the input value.
		"""
		ret = Vec(length, element=element, sparse=False)
		filteredValues = [-8000, 8000, 3, 2, 5, 4]
		filteredInds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
					val.weight = v[0][ind]
					val.category = v[1][ind]
				else:
					val.weight = v
					val.category = v
				ret[i[ind]] = val

				# make sure we don't filter out actual values
				if filteredValues.count(val.weight) > 0:
					filteredValues.remove(val.weight)
		# make sure we don't override existing elements
		self.assertTrue(len(filteredInds) > 0)
		# add extra elements
		for ind in range(len(filteredInds)):
			fv = filteredValues[ind % len(filteredValues) ]
			if isinstance(element, Obj1):
				val = pcb.Obj1()
				val.weight = fv
			elif isinstance(element, Obj2):
				val = pcb.Obj2()
				val.weight = fv
			else:
				val = fv
			ret[filteredInds[ind]] = val
		# add the filter to take out the extra elements we just added,
		# so tests should pass as if we didn't do anything.
		if ret.isObj():
			ret.addFilter(lambda e: filteredValues.count(e.weight) == 0)
		else:
			ret.addFilter(lambda e: filteredValues.count(e) == 0)
		return ret

class ConstructorTests(VecTests):
	def test_Vec_dint_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		vec = self.initializeVec(sz, i, element=0)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]])
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind])

	def test_Vec_Obj1_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		element = Obj1()
		vec = self.initializeVec(sz, i, element=element)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertEqual(type(element),type(vec[0]))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expI[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind].weight)

	def test_Vec_Obj2_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		element = Obj2()
		vec = self.initializeVec(sz, i, element=element)
		expI = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertEqual(type(element),type(vec[0]))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expI[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind].weight)

	def test_Vec_dint_null(self):
		sz = 25
		i = []
		element = 0.0
		vec = self.initializeVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertEqual(type(element),type(vec[0]))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]])
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind])
		return

	def test_Vec_Obj1_null(self):
		sz = 25
		i = []
		element = Obj1()
		vec = self.initializeVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertEqual(type(element),type(vec[0]))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expI[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind].weight)
		return

	def test_Vec_Obj2_null(self):
		sz = 25
		i = []
		element = Obj2()
		vec = self.initializeVec(sz, i, element=element)
		expI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		self.assertEqual(type(element),type(vec[0]))
		for ind in range(len(i)):
			self.assertAlmostEqual(expI[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expI[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expI[ind],vec[ind].weight)
		return

	def test_Vec_dint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = 0.0
		vec = self.initializeVec(sz, i, weight, element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]])
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind])
		return

	def test_Vec_Obj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expT[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
		return

	def test_Vec_Obj2(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 4, 16, 36, 64, 100]
		category = [2, 2, 2, 5, 5, 5]
		element = Obj2()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expT[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
		return

	def test_Vec_Obj1_all_weight_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [2, 2, 2, 5, 5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, 2, 0, 2, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expT[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
		return

	def test_Vec_dint_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		element = 0.0
		vec = self.initializeVec(sz, i, weight, element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqual(sz, len(vec))
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]])
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind])
		return

	def test_Vec_Obj1_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqual(sz, len(vec))
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expT[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
		return

	def test_Vec_Obj2_all_weight_all_type_zero(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 0, 0, 0, 0, 0]
		category =   [0, 0, 0, 0, 0, 0]
		element = Obj2()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(len(i)):
			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
			self.assertEqual(expT[i[ind]],vec[i[ind]].category)
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
		return

#	def test_Vec_object_spRange(self):
#		sz = 25
#		i = [0, 2, 4, 6, 9, 10]
#		weight = [1, 4, 8, 12, 18, 20]
#		category =   [0, 0, 0, 0, 0, 0]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
#		vec.spRange()
#		expW = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#				 0, 0, 0, 0, 0, 0]
#		expT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#				 0, 0, 0, 0, 0]
		self.assertEqual(type(element),type(vec[0]))
#		self.assertEqual(sz, len(vec))
#		for ind in range(len(i)):
#			self.assertAlmostEqual(expW[i[ind]],vec[i[ind]].weight)
#			#self.assertEqual(expT[i[ind]],vec[i[ind]].category)
#		for ind in range(sz):
#			self.assertAlmostEqual(expW[ind],vec[ind].weight)
#
##	def test_Vec_zeros(self):
##		sz = 25
##		vec = Vec(sz)
##		expI = 0
##		self.assertEqual(sz, len(vec))
##		for ind in range(sz):
##			self.assertEqual(expI,vec[ind])
##
#	def test_Vec_ones_simple(self):
#		sz = 25
#		vec = Vec.ones(sz)
#		expI = 1
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expI,vec[ind])
#
#	def test_Vec_ones_Obj1(self):
#		sz = 25
#		vec = Vec.ones(sz, element=pcb.Obj1())
#		expI = 1
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expI,vec[ind].weight)

	def test_Vec_range_simple(self):
		sz = 25
		vec = Vec.range(sz)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind])

	def test_Vec_range_Obj1(self):
		sz = 25
		vec = Vec.range(sz, element=pcb.Obj1())
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind].weight)

	def test_Vec_range_offset_simple(self):
		sz = 25
		offset = 7
		vec = Vec.range(offset, stop=sz+offset)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind+offset,vec[ind])

	def test_Vec_range_offset_Obj1(self):
		sz = 25
		offset = 7
		vec = Vec.range(offset, stop=sz+offset, element=pcb.Obj1())
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind+offset,vec[ind].weight)

	def test_Vec_range_offset_simple(self):
		sz = 25
		offset = -13
		vec = Vec.range(offset, stop=sz+offset)
		expI = 1
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind]-offset)
			#self.assertEqual(ind,vec[ind].weight-offset)

	def test_Vec_range_offset_Obj1(self):
		sz = 25
		offset = -13
		vec = Vec.range(offset, stop=sz+offset, element=pcb.Obj1())
		expI = 1
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(ind,vec[ind].weight-offset)

#	def test_Vec_toBool(self):
#		sz = 25
#		i = [0,	  2,		4,  6, 8, 10, 12]
#		w = [1, .00001, -.000001, -1, 0, 24, 3.1415962]
#		t = [2, 2,  2,  5,  5,  5, 5]
#		vec = self.initializeVec(sz, i, (w, t))
#		vec.toBool()
#		expVec = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#				0, 0, 0, 0, 0, 0, 0]
#		self.assertEqual(sz, len(vec))
#		for ind in range(sz):
#			self.assertEqual(expVec[ind],vec[ind].weight)

class xxxTests(VecTests):
	def test_bitwise_abs_vectors_Obj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)
	
	def test_load_save(self):
		from shutil import rmtree
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec.save("tempsave.vec")
		loaded = Vec.load("tempsave.vec", element=0.0)
		#rmtree("tempsave.vec")
		#ind= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
		exp = [0, 0, 4, 0, 8, 0,12, 0,16, 0,20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(loaded))
		for ind in range(sz):
			self.assertAlmostEqual(exp[ind],loaded[ind])

	def test_load_save_Obj1(self):
		from shutil import rmtree
		sz = 25
		i =      [0, 2, 4,  6,  8,  10]
		weight = [0,-4,16,-36,-64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		vec.save("tempsave.vec")
		loaded = Vec.load("tempsave.vec", element=Obj1())
		#rmtree("tempsave.vec")
		expW = [0, 0,-4, 0,16, 0,-36, 0,-64, 0,100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(loaded))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],loaded[ind].weight)
	
	def test_randperm_sort(self):
		sz = 25
		vec = Vec.range(sz, sparse=False)
		permed = vec.copy()
		permed.randPerm()
		sorted = permed.copy()
		sorted.sort()
		for ind in range(sz):
			self.assertAlmostEqual(vec[ind],sorted[ind])

class BuiltInTests(VecTests):
	def test_len_simple(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeVec(sz, i, weight)
		res = len(vec)
		self.assertEqual(sz, res)

	def test_len_Obj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		res = len(vec)
		self.assertEqual(sz, res)

	def test_add_vector_dint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		vec1 = self.initializeVec(sz, i1, w1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		vec2 = self.initializeVec(sz, i2, w2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_add_vectors_Obj1_Obj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element1 = Obj1()
		vec2 = self.initializeVec(sz, i2, (w2, t2), element=element1)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_add_vectors_Obj1_Obj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj2()
		vec2 = self.initializeVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_add_vectors_Obj2_Obj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj2()
		vec1 = self.initializeVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj2()
		vec2 = self.initializeVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_add_vectors_Obj1_dint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [2, 2,  3,  3,  7,   7]
		element1 = Obj1()
		vec1 = self.initializeVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		vec2 = self.initializeVec(sz, i2, w2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_add_vectors_dint_Obj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		vec1 = self.initializeVec(sz, i1, w1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [2, 2, 3, 3,  7,  7, 23]
		element2 = Obj2()
		vec2 = self.initializeVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 + vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_bitwiseAnd_vector_scalar_dint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		vec1 = self.initializeVec(sz, i1, w1)
		scalar = 0x7
		vec3 = vec1 & scalar
		expI = [0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]		
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_bitwiseAnd_vector_dint(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F]
		vec1 = self.initializeVec(sz, i1, w1)
		i2 = [0, 2, 4, 6, 8, 10, 18]
		w2 = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
		vec2 = self.initializeVec(sz, i2, w2)
		vec3 = vec1 & vec2
		expI = [0x1, 0, 0x2, 0, 0x4, 0, 0x8, 0, 0x10, 0, 0x20, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

	def test_bitwiseAnd_vector_Obj1_Obj2(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F]
		c1 = [2, 2,  2,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeVec(sz, i1, (w1, c1), element=element1 )
		i2 = [0, 2, 4, 6, 8, 10, 18]
		w2 = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
		c2 = [2, 2, 2, 2,  2,  2,  2]
		element2 = Obj2()
		vec2 = self.initializeVec(sz, i2, (w2, c2), element=element2 )
		vec3 = vec1 & vec2
		expI = [0x1, 0, 0x2, 0, 0x4, 0, 0x8, 0, 0x10, 0, 0x20, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)


class BuiltInTests_disabled(VecTests):
		#disabled because no EWiseApply(vec, scalar)
	def test_add_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = vec + 3.07
		self.assertEqual(sz, len(res))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)
			#self.assertEqual(expT[ind],res[ind].category)

	def test_iadd_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [3.07, 0, -.93, 0, 19.07, 0, -32.93, 0, -60.93, 0, 103.07, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec += 3.07
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_iadd_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec1 += vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_invert(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [-1, 0, 3, 0, -17, 0, 35, 0, 63, 0, -101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		res = ~vec
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)
			#self.assertEqual(expT[ind],res[ind].category)

	def test_sub_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [-3.07, 0, -7.07, 0, 12.93, 0, -39.07, 0, -67.07, 0, 96.93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec2 = vec - 3.07
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec2[ind].weight)

	def test_sub_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec3 = vec1 - vec2
		expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0, -3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_isub_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [-3.07, 0, -7.07, 0, 12.93, 0, -39.07, 0, -67.07, 0, 96.93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec -= 3.07
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_isub_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec1 -= vec2
		expI = [-1, 0, 4, -27, 16, 0, -180, 0, 64, -729, 100, 0, -1728, 0, 0, -3375, 0, 0, -5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_mul_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [0, 0, -13, 0, 52, 0, -117, 0, -208, 0, 325, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec *= 3.25
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_mul_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec1 *= vec2
		expI = [0, 0, 0, 0, 0, 0, 36*216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec1))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec1[ind].weight)

	def test_div_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializec(sz, i, (weight, category))
		expW = [0.0, 0, -1.3333333, 0, 5.3333333, 0, -12.0, 0, -21.33333333, 0, 
				33.3333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = vec / 3
		self.assertEqual(sz, len(res))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_div_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		self.assertRaises(NotImplementedError, vec1.__div__, vec1, vec2)
		return
		#NOTE:  dead code below; numbers not yet arranged to work properly
		vec3 = vec1 / vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_mod_scalar(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4.25, 16.5, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [0.0, 0, 1.75, 0, 1.5, 0, 0, 0, 2, 0, 
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = vec % 3
		self.assertEqual(sz, len(res))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_mod_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		self.assertRaises(NotImplementedError, vec1.__mod__, vec1, vec2)
		return
		#NOTE:  dead code below; numbers not yet arranged to work properly
		vec3 = vec1 % vec2
		expI = [1, 0, 4, 27, 16, 0, 252, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseAnd_vectors_Obj1_Obj1(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		element1 = Obj1()
		vec1 = self.initializeVec(sz, i1, (w1, t1), element=element1)
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		element2 = Obj1()
		vec2 = self.initializeVec(sz, i2, (w2, t2), element=element2)
		vec3 = vec1 & vec2
		expI = [0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(type(element1),type(vec3[0]))
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseOr_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec3 = vec1 | vec2
		expI = [1, 0, 4, 27, 16, 0, 253, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_bitwiseXor_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 61, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec3 = vec1 ^ vec2
		expI = [1, 0, 4, 27, 16, 0, 228, 0, 64, 729, 100, 0, 1728, 0, 0, 3375,
				0, 0, 5832, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_logicalOr_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec3 = vec1.logicalOr(vec2)
		expI = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
				0, 0, 1, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

	def test_logicalXor_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 37, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 217, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		vec3 = vec1.logicalXor(vec2)
		expI = [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
				0, 0, 1, 0, 0, 0, 0, 0, 0,]
		self.assertEqual(sz, len(vec3))
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind].weight)

#	def test_indexing_RHS_Vec_scalar(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 9
#		value = vec1[ndx]
#		self.assertEqual(-3, value)
#
#	def test_indexing_RHS_Vec_scalar_outofbounds(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 7777
#		self.assertRaises(IndexError, Vec.__getitem__, vec1, ndx)
#		#value = vec1[ndx]
#		#self.assertEqual(-3, value)
#
#	def test_indexing_RHS_Vec_scalar_outofbounds2(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = -333
#		self.assertRaises(IndexError, Vec.__getitem__, vec1, ndx)
#		#value = vec1[ndx]
#		#self.assertEqual(-3, value)
#
#	def test_indexing_RHS_Vec_Vec(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 4
#		ndxI = [0, 1, 2,  3]
#		ndxV = [1, 4, 9, 16]
#		ndx = self.initializeVec(ndxLen, ndxI, ndxV)
#		self.assertRaises(KeyError, Vec.__getitem__, vec1, ndx)
#		#expI = [-11, -8, -3, 4]
#		#self.assertEqual(ndxLen, len(vec3))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec3[ind])
#
#	def test_indexing_RHS_Vec_Vec(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
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
#	def test_indexing_RHS_Vec_booleanVec(self):
#		sz = 25
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 25
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
#				18, 19, 20, 21, 22, 23, 24]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#				0, 0, 0, 0, 0]
#		ndx = self.initializeVec(ndxLen, ndxI, ndxV)
#		self.assertRaises(KeyError, Vec.__getitem__, vec1, ndx)
#		#vec3 = vec1[ndx]
#		#expI = [1,16,81,256]
#		#self.assertEqual(ndxTrue, len(vec3))
#		#for ind in range(ndxTrue):
#		#	self.assertEqual(expI[ind], vec3[ind])
#
#	def test_indexing_LHS_Vec_booleanVec_scalar(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
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
#	def test_indexing_LHS_Vec_nonbooleanVec_scalar(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndx = Vec(sz)
#		ndx[1] = 7
#		ndx[4] = 11
#		ndx[9] = 1
#		ndx[16] = 5
#		self.assertRaises(KeyError, Vec.__setitem__, vec1, ndx, 77)
#		#vec1[ndx] = 77
#		#expI = [-9, 77, -7, -6, 77, -4, -3, -2, -1, 77, 1, 2, 3, 4, 5, 6, 77, 8]
#		#self.assertEqual(sz, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_Vec_scalar_scalar(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndx = 11
#		value = 777
#		vec1[ndx] = value
#		expI = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 777, 3, 4, 5, 6, 7, 8]
#		self.assertEqual(sz, len(vec1))
#		self.assertEqual(len(expI), len(vec1))
#		for ind in range(sz):
#			self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_Vec_booleanVec_scalar(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#		ndx = self.initializeVec(ndxLen, ndxI, ndxV)
#		value = 777
#		self.assertRaises(KeyError,Vec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 777, 4, 9, 777, 25, 36, 49, 64, 777, 100, 121, 144, 169, 
#		#		196, 225, 777, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_Vec_booleanVec_Vec(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 18
#		ndxTrue = 4
#		ndxI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#		ndxV = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#		ndx = self.initializeVec(ndxLen, ndxI, ndxV)
#		valueV = [0, 1, 7, 7, 0.25, 0, 0, 0, 0, 0.111, 0, 0, 0, 0, 0, 0, 0.0625,
#				0, 0, 0, 0, 7, 0, 0, 0]
#		value = self.initializeVec(ndxLen, ndxI, valueV)
#		self.assertRaises(KeyError,Vec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#		#		196, 225, 0.0625, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
#	def test_indexing_LHS_Vec_Vec(self):
#		sz = 18
#		vec1 = Vec.range(int(-math.floor(sz/2.0)),int(math.ceil(sz/2.0)))
#		ndxLen = 4
#		ndxI = [0, 1, 2, 3]
#		ndxV = [1, 4, 9, 16]
#		ndx = self.initializec(ndxLen, ndxI, ndxV)
#		valueV = [1, 0.25, 0.111, 0.0625]
#		value = self.initializeVec(ndxLen, ndxI, valueV)
#		self.assertRaises(IndexError,Vec.__setitem__,vec1,ndx, value)
#		#vec1[ndx] = value
#		#expI = [0, 1, 4, 9, 0.25, 25, 36, 49, 64, 0.111, 100, 121, 144, 169, 
#		#		196, 225, 0.0625, 289]
#		#self.assertEqual(ndxLen, len(vec1))
#		#for ind in range(ndxLen):
#		#	self.assertEqual(expI[ind], vec1[ind])
#
	def test_eq_scalar(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		scalar = 4
		eqY = vec1 == scalar
		expV = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(eqY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], eqY[ind].weight)

	def test_eq_vector(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8,  10,  12]
		w1 = [0, 4, 16, 36, 64, 100, 144]
		t1 = [1, 1,  1,  2,  2,   2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 2,   6,   9,  12,   15,   18]
		w2 = [1, 4, 216, 729, 144, 3375, 5832]
		t2 = [1, 1,   1,   2,   2,	2,	2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		eqY = vec1 == vec2
		expV = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(eqY), len(vec1))
		self.assertEqual(len(eqY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], eqY[ind].weight)

	def test_ne_scalar(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		scalar = 4
		neY = vec1 != scalar
		expV = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(neY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], neY[ind].weight)

	def test_ne_vector(self):
		sz = 25
		i1 = [0, 2,  4,  6,  8,  10,  12]
		w1 = [0, 4, 16, 36, 64, 100, 144]
		t1 = [1, 1,  1,  2,  2,   2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 2,   6,   9,  12,   15,   18]
		w2 = [1, 4, 216, 729, 144, 3375, 5832]
		t2 = [1, 1,   1,   2,   2,	2,	2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		neY = vec1 != vec2
		expV = [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(neY), len(vec1))
		self.assertEqual(len(neY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], neY[ind].weight)

	def test_ge_scalar(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
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
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
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
		vec1 = self.initializeVec(sz, i1, (w1, t1))
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
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		gtY = vec1 > vec2
		expV = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(gtY), len(vec1))
		self.assertEqual(len(gtY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], gtY[ind].weight)

	def test_le_scalar(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		scalar = 4
		leY = vec1 <= scalar
		expV = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(leY), len(vec1))
		for ind in range(sz):
			self.assertEqual(expV[ind], leY[ind].weight)

	def test_le_vector(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		leY = vec1 <= vec2
		expV = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(leY), len(vec1))
		self.assertEqual(len(leY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], leY[ind].weight)

	def test_lt_scalar(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		w1 = [0, 4, 16, 36, 64, 100]
		t1 = [1, 1,  1,  2,  2,   2]
		vec1 = self.initializeVec(sz, i1, (w1, t1))
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
		vec1 = self.initializeVec(sz, i1, (w1, t1))
		i2 = [0, 3, 6, 9, 12, 15, 18]
		w2 = [1, 27, 216, 729, 1728, 3375, 5832]
		t2 = [1, 1,  1,  2,  2, 2, 2]
		vec2 = self.initializeVec(sz, i2, (w2, t2))
		ltY = vec1 < vec2
		expV = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
				0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(len(ltY), len(vec1))
		self.assertEqual(len(ltY), len(vec2))
		for ind in range(sz):
			self.assertEqual(expV[ind], ltY[ind].weight)



class GeneralPurposeTests(VecTests):
	def test_abs_vectors_dint(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeVec(sz, i, weight)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind])

	def test_abs_vectors_Obj1(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_abs_vectors_Obj2(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		element = Obj2()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(element), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def disabled_test_abs_vectors_dint_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		vec = self.initializeVec(sz, i, weight)
		element = Obj1()
		self.assertRaises(NotImplementedError, vec.addFilter, Obj1.ge0lt5)
		#dead code
		expW = [0, 0, 4, 0, 16, 0, 36, 0, 64, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind])

	def test_abs_vectors_Obj1_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -2, 1, 3, -64, 100]
		category = [2, 2, 2, 2, 2, 2]
		element = Obj1()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		vec.addFilter(Obj1.geM2lt4)
		expW = [0, 0, 2, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)

	def test_abs_vectors_Obj2_filtered(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, 64, 1, 3, -2, -100]
		category = [2, 2, 2, 2, 2, 2]
		element = Obj2()
		vec = self.initializeVec(sz, i, (weight, category), element=element)
		vec.addFilter(Obj2.geM2lt4)
		expW = [0, 0, 0, 0, 1, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		res = abs(vec)
		self.assertEqual(type(vec[0]), type(res[0]))
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)


class GeneralPurposeTests_disabled(VecTests):
#	def test_all_all_nonnull_all_true(self):
#		sz = 10
#		vec = Vec.ones(sz)
#		res = vec.all()
#		self.assertEqual(True, res)
#
#	def test_all_all_nonnull_one_false(self):
#		sz = 10
#		vec = Vec.ones(sz)
#		vec[4] = False
#		res = vec.all()
#		self.assertEqual(False, res)
#
#	def test_all_all_nonnull_one_true(self):
#		sz = 10
#		vec = Vec.ones(sz)-1
#		vec[4] = True
#		res = vec.all()
#		self.assertEqual(False, res)
#
#	def test_all_all_nonnull_all_false(self):
#		sz = 10
#		vec = Vec.ones(sz)-1
#		res = vec.all()
#		self.assertEqual(False, res)
#
#	def test_all_one_nonnull_one_false(self):
#		sz = 10
#		vec = Vec(sz)
#		vec[4] = 0
#		res = vec.all()
#		self.assertEqual(False, res)
#
#	def test_all_one_nonnull_one_true(self):
#		sz = 10
#		vec = Vec(sz)
#		vec[4] = 1
#		res = vec.all()
#		self.assertEqual(True, res)
#
#	def test_all_three_nonnull_one_true(self):
#		sz = 10
#		vec = Vec(sz)
#		vec[0] = 0
#		vec[2] = 0
#		vec[4] = 1
#		res = vec.all()
#		self.assertEqual(False, res)
#
#	def test_all_three_nonnull_three_true(self):
#		sz = 10
#		vec = Vec(sz)
#		vec[0] = 1
#		vec[2] = 1
#		vec[4] = 1
#		res = vec.all()
#		self.assertEqual(True, res)
#
#	def test_all_all_nonnull_all_true(self):
#		sz = 6
#		i = [0, 1, 2, 3, 4, 5]
#		weight = [777, -4, 16, -36, -64, 100]
#		category = [2, -2, 2, 5, -5, 5]
#		vec = self.initializeVec(sz, i, (weight, category))
#		res = vec.all()
#		expRes = True
#		self.assertEqual(expRes, res)

#	def test_all_no_elem0(self):
#		sz = 6
#		i = [1, 2, 3, 4, 5]
#		weight = [-4, 16, 0, -64, 100]
#		category = [2, 5, 5, -5, 5]
#		vec = self.initializeVec(sz, i, (weight, category))
#		res = vec.all()
#		expRes = True
#		self.assertEqual(expRes, res)

#	def test_all_no_elem3(self):
#		sz = 6
#		i = [1, 2, 4, 5]
#		weight = [-1, -4, 16, -64, 100]
#		category = [2, 5, 5, -5, 5]
#		vec = self.initializeVec(sz, i, (weight, category))
#		res = vec.all()
#		expRes = True
#		self.assertEqual(expRes, res)

	def test_all_evens_nonnull_elem2_false(self):
		sz = 6
		i = [0, 2, 4]
		weight = [1, 0, -64]
		category = [2, 2, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.all()
		expRes = False
		self.assertEqual(expRes, res)

	def test_any_all_nonnull_all_true(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [777, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def test_any_all_nonnull_elem0_false(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def test_any_all_nonnull_elem3_false(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [1, -4, 16, 0, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

	def test_any_evens_nonnull_elem2_false(self):
		sz = 6
		i = [0, 2, 4]
		weight = [1, 0, -64]
		category = [2, 2, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.any()
		expRes = True
		self.assertEqual(expRes, res)

#	def test_any_all_true(self):
#		sz = 10
#		vec = Vec.ones(sz)
#		res = vec.any()
#		self.assertEqual(True, res)
#
#	def test_any_one_false(self):
#		sz = 10
#		vec = Vec.ones(sz)
#		vec[4] = False
#		res = vec.any()
#		self.assertEqual(True, res)
#
#	def test_any_one_true(self):
#		sz = 10
#		vec = Vec.ones(sz)-1
#		vec[4] = True
#		res = vec.any()
#		self.assertEqual(True, res)
#
#	def test_any_all_false(self):
#		sz = 10
#		vec = Vec.ones(sz)-1
#		res = vec.any()
#		self.assertEqual(False, res)
#
	def test_sum_nulls(self):
		sz = 10
		vec = Vec(sz)
		res = vec.sum()
		self.assertEqual(0.0, res.weight)

	def test_sum_ones(self):
		sz = 10
		vec = Vec.ones(sz)
		res = vec.sum()
		self.assertEqual(sz, res.weight)

	def test_sum_range(self):
		sz = 10
		vec = Vec.range(sz)
		res = vec.sum()
		self.assertEqual((sz*(sz-1))/2, res)

	def test_sum_range2(self):
		sz = 11
		vec = Vec.range(-(sz/2), stop=(sz/2)+1)
		res = vec.sum()
		self.assertEqual(0, res)

#	def test_sum_fixed(self):
#		sz = 11
#		vec = Vec(sz)
#		vec[2] = 23
#		vec[4] = 9
#		vec[5] = -32
#		res = vec.sum()
#		self.assertEqual(0, res)

	def test_set(self):
		sz = 25
		k = 3.7
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 3.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec.set(k)
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)

	def test_negate(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [0, 0, 4, 0, -16, 0, 36, 0, 64, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [2, 0, -2, 0, 2, 0, 5, 0, -5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		res = -vec
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],res[ind].weight)
			#self.assertEqual(expT[ind],res[ind].category)

	def test_max_some_nonnull_maxElem0(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [123, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.max()
		expRes = 123
		self.assertEqual(expRes, res.weight)

	def test_max_all_nonnull_maxElem4(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.max()
		expRes = 136
		self.assertEqual(expRes, res.weight)

	def test_max_some_nonnull_maxElem6(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.max()
		expRes = 136
		self.assertEqual(expRes, res.weight)

	def test_min_some_nonnull_minElem0(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [-123, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.min()
		expRes = -123
		self.assertEqual(expRes, res.weight)

	def test_min_all_nonnull_minElem4(self):
		sz = 6
		i = [0, 1, 2, 3, 4, 5]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.min()
		expRes = -64
		self.assertEqual(expRes, res.weight)

	def test_min_some_nonnull_minElem8(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, 136, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		res = vec.min()
		expRes = -64
		self.assertEqual(expRes, res.weight)

	def test_sort(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [-64, 0, -36, 0, -4, 0, 0, 0, 16, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [-5, 0, 5, 0, -2, 0, 2, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		vec.sort()
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec[ind].weight)
			self.assertEqual(expT[ind],vec[ind].category)

	def test_sorted(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [-64, 0, -36, 0, -4, 0, 0, 0, 16, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0]
		expT = [-5, 0, 5, 0, -2, 0, 2, 0, 2, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		expNdx = [8, 0, 6, 0, 2, 0, 0, 0, 4, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0]
		(vec2, vec3) = vec.sorted()
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(expW[ind],vec2[ind].weight)
			self.assertEqual(expT[ind], vec2[ind].category)
			self.assertAlmostEqual(expNdx[ind],vec3[ind])

	def test_topK(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		weight = [0, -4, 16, -36, -64, 100]
		category = [2, -2, 2, 5, -5, 5]
		vec = self.initializeVec(sz, i, (weight, category))
		expW = [100, 16, 0]
		expT = [5, 2, 2]
		topKSz = 3
		vec2 = vec.topK(topKSz)
		self.assertEqual(topKSz, len(vec2))
		for ind in range(topKSz):
			self.assertAlmostEqual(expW[ind],vec2[ind].weight)
			self.assertEqual(expT[ind], vec2[ind].category)

#	def test_load(self):
#		vec = Vec.load('UFget/Pajek/CSphd/CSphd.mtx')
#		return


class MixedDenseSparseVecTests(VecTests):
	def test_add_sparse_dense(self):
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

	def test_add_dense_sparse(self):
		sz = 25
		i = [0,    2,   4,  6, 8, 10]
		v = [0,    4,   8, 12,16, 20]
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

	def test_subtract_sparse_dense(self):
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

	def test_isubtract_sparse_dense(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeSpVec(sz, i, v)
		i2 = [0, 1,   3,  5, 7]
		v2 = [0,-1, 0.1,777, 0]
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec.copy()
		vec3 -= vec2
		expI = [0, 1, 4, -0.1, 8, -777, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0,
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
		vec2 = self.initializeVec(sz, i2, v2)
		vec3 = vec2 - vec
		expI = [0, -2, -4, 0.1, -8, 777, -12, 0, -16, 0, -20, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec3))
		#self.assertEqual(len(i)+len(i2)-1, vec3.nnn())
		for ind in range(sz):
			self.assertEqual(expI[ind], vec3[ind])

class FindTests(VecTests):
	def test_find(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec2 = vec.find(lambda x: x > 0 and x < 10)
		vecExpected = [0, 0, 4, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec2[ind])
	def test_findInds(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec2 = vec.findInds(lambda x: x > 0 and x < 10)
		vecExpected = [2, 4]
		self.assertEqual(len(vecExpected), len(vec2))
		for ind in range(len(vecExpected)):
			self.assertEqual(vecExpected[ind], vec2[ind])


class ApplyReduceTests(VecTests):
	def test_apply(self):
		# AL: i moved the lower bound from 0 to 1 because the default value of 0 meant that the
		# predicate returns True for the unspecified elements, not False like vecExpected expects.
		def ge0lt5(x):
				return x>=1 and x<5
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [1, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec.apply(ge0lt5)
		vecExpected = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind])

	def test_apply_pcbabs(self):
		sz = 25
		i = [0, 2,  4,   6, 8, 10]
		v = [0, -4, 8, -12,16, 20]
		vec = self.initializeVec(sz, i, v)
		if vec._hasFilter():
			# abs(-4) == 4, and 4s are filtered out, so the test fails.
			# the test below will test the case where there is no overlap.
			#print "!"
			return
		vec.apply(op_abs)
		vecExpected = [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind])

	def test_apply_pcbabs_no4problem(self):
		sz = 25
		i = [0, 2,  4,   6, 8, 10, 11]
		v = [0, -4, 8, -12,16, 20, 4]
		vec = self.initializeVec(sz, i, v)
		vec.apply(op_abs)
		vecExpected = [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind])

	def test_count(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		ct = vec.count()
		ctExpected = 5
		self.assertEqual(ctExpected, ct)

	def test_count_pred(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		ct = vec.count(lambda x: x > 10)
		ctExpected = 3
		self.assertEqual(ctExpected, ct)

	def test_reduce_default_op(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		red = vec.reduce(kdt.op_add)
		redExpected = 60
		self.assertEqual(redExpected, red)

	def test_reduce_max(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		red = vec.reduce(kdt.op_max)
		redExpected = 20
		self.assertEqual(redExpected, red)

	def test_reduce_min(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [2, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v, element=100) # the element is required because otherwise all the unspecified elements will become 0, which is smaller than 2.
		red = vec.reduce((lambda x,y: min(x,y)), init=20)
		redExpected = 2
		self.assertEqual(redExpected, red)

class ApplyReduceTests_disabled(VecTests):
	def test_apply(self):
		def ge0lt5(x):
				return x>=0 and x<5
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec.apply(ge0lt5)
		vecExpected = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind])

	def test_apply_pcbabs(self):
		sz = 25
		i = [0, 2,  4,   6, 8, 10]
		v = [0, -4, 8, -12,16, 20]
		vec = self.initializeVec(sz, i, v)
		vec.apply(op_abs)
		vecExpected = [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind])

	def test_count(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		ct = vec.count()
		ctExpected = 6
		self.assertEqual(ctExpected, ct)

	def test_reduce_default_op(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		red = vec.reduce(Vec.op_add)
		redExpected = 60
		self.assertEqual(redExpected, red)

	def test_reduce_max(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [0, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		red = vec.reduce(Vec.op_max)
		redExpected = 20
		self.assertEqual(redExpected, red)

	def test_reduce_min(self):
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [2, 4, 8,12,16, 20]
		vec = self.initializeVec(sz, i, v)
		red = vec.reduce(Vec.op_min)
		redExpected = 2
		self.assertEqual(redExpected, red)

		
class FilterTests(VecTests):

	# the following three functions came out of BuiltinTests
	def test_add_vectors_filterBoth(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(element.geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_add_vectors_filterFirst(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		# ---- commented----  vec2.addFilter(element.geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [-2, 0, 1, 0, 3, 0, 5, 0, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_add_vectors_filterSecond(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		# ----- commented out--- vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(element.geM2lt4)
		vec3 = vec1 + vec2
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_apply_filter(self):
		def add5(x):
				if isinstance(x, (int, long, float)):
						return x+5
				else:
						x.weight += 5
						x.category *= 2
						return x
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [1, 4, 8,12,16, 20]
		c = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec = self.initializeVec(sz, i, (v,v), element=element)
		vec.addFilter(element.ge0lt5)
		vec.apply(add5)
		vec.delFilter(element.ge0lt5)
		vecExpected = [6, 5, 9, 5, 8, 5, 12, 5, 16, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind].weight)

	def test_apply_filters(self):
		def add3p14(x):
				if isinstance(x, (int, long, float)):
						return x+3.14159
				else:
						x.weight += 3.14159
						x.category *= 2
						return x
		sz = 25
		i = [ 0, 2, 4, 6, 8, 10]
		v = [-3, 2, 3, 4, 5,  6]
		c = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec = self.initializeVec(sz, i, (v,v), element=element)
		vec.addFilter(element.ge0lt5)
		vec.addFilter(element.geM2lt4)
		vec.apply(add3p14)
		vec.delFilter(element.ge0lt5)
		vec.delFilter(element.geM2lt4)
		vecExpected = [-3, 3.14159, 5.14159, 3.14159, 6.14159, 3.14159, 4, 3.14159, 5, 3.14159, 6, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind].weight)

	def test_delete_all_filters(self):
		# add filters, then delete before apply
		def add3p14(x):
				if isinstance(x, (int, long, float)):
						return x+3.14159
				else:
						x.weight += 3.14159
						x.category *= 2
						return x
		sz = 25
		i = [ 0, 2, 4, 6, 8, 10]
		v = [-3, 2, 3, 4, 5,  6]
		c = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec = self.initializeVec(sz, i, (v,v), element=element)
		vec.addFilter(element.ge0lt5)
		vec.addFilter(element.geM2lt4)
		vec.delFilter()
		vec.apply(add3p14)
		vecExpected = [0.14159, 3.14159, 5.14159, 3.14159, 6.14159, 3.14159, 7.14159, 3.14159, 8.14159, 3.14159, 9.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertAlmostEqual(vecExpected[ind], vec[ind].weight)

	def test_apply2_deleteLast_filter(self):
		def add5(x):
				if isinstance(x, (int, long, float)):
						return x+5
				else:
						x.weight += 5
						x.category *= 2
						return x
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [1, 4, 8,12,16, 20]
		c = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec = self.initializeVec(sz, i, (v,v), element=element)
		vec.addFilter(element.ge0lt5)
		vec.addFilter(element.geM2lt4)
		vec.delFilter(element.geM2lt4)
		vec.apply(add5)
		vec.delFilter(element.ge0lt5) # must remove this filter or else the [] operation is still filtered
		vecExpected = [6, 5, 9, 5, 8, 5, 12, 5, 16, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind].weight)

	def test_apply2_deleteFirst_filter(self):
		def add5(x):
				if isinstance(x, (int, long, float)):
						return x+5
				else:
						x.weight += 5
						x.category *= 2
						return x
		sz = 25
		i = [0, 2, 4, 6, 8, 10]
		v = [1, 4, 8,12,16, 20]
		c = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec = self.initializeVec(sz, i, (v,v), element=element)
		vec.addFilter(element.ge0lt5)
		vec.addFilter(element.geM2lt4)
		vec.delFilter(element.ge0lt5)
		vec.apply(add5)
		vec.delFilter(element.geM2lt4) # must remove this filter or else the [] operation is still filtered
		vecExpected = [6, 5, 4, 5, 8, 5, 12, 5, 16, 5, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
		self.assertEqual(sz, len(vec))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec[ind].weight)

	def test_eWiseApply_filterBoth(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(element.geM2lt4)
		vec3 = vec1.eWiseApply(vec2, Obj1.__iadd__, True, True)
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_eWiseApply_filterFirst(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		# ---- commented----  vec2.addFilter(element.geM2lt4)
		vec3 = vec1.eWiseApply(vec2, Obj1.__iadd__, True, True)
		vecExpected = [-2, 0, 1, 0, 3, 0, 5, 0, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)

	def test_eWiseApply_filterSecond(self):
		sz = 25
		i1 = [0, 2, 4, 6, 8, 10]
		v1 = [1, 2, 3, 4, 5,  4]
		c1 = [2, 2, 7, 7, 3,  3]
		element = Obj1()
		vec1 = self.initializeVec(sz, i1, (v1,v1), element=element)
		# ----- commented out--- vec1.addFilter(element.ge0lt5)
		i2 = [ 0, 2, 4, 6, 8, 10]
		v2 = [-3,-1, 0, 1, 2,  5]
		c2 = [ 2, 2, 7, 7, 3,  3]
		vec2 = self.initializeVec(sz, i2, (v2,v2), element=element)
		vec2.addFilter(element.geM2lt4)
		vec3 = vec1.eWiseApply(vec2, Obj1.__iadd__, True, True)
		vecExpected = [1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(sz, len(vec1))
		self.assertEqual(sz, len(vec2))
		for ind in range(sz):
			self.assertEqual(vecExpected[ind], vec3[ind].weight)


def runTests(verbosity = 1):
	testSuite = suite()
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

	print "running again using filtered data:"
	
	VecTests.fillVec = VecTests.fillVecFilter
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
	suite = unittest.TestSuite()
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MixedDenseSparseVecTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(FindTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ApplyReduceTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(xxxTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(FilterTests))

	return suite

if __name__ == '__main__':
		runTests()
