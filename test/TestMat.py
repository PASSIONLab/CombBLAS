import unittest
#import sys
#print sys.prefix
#print sys.path
from kdt import *

class MatTests(unittest.TestCase):
	def fillMat_worker(self, nvert, nedge, i, j, v):
		"""
		Initialize a graph with edge weights equal to one or the input value.
		"""
		iInd = Vec(nedge, sparse=False)
		jInd = Vec(nedge, sparse=False)
		if type(v) == int or type(v) == float:
			vInd = Vec(nedge, v, sparse=False)
		else:
			vInd = Vec(nedge, sparse=False)
		for ind in range(nedge):
			iInd[ind] = i[ind]
			jInd[ind] = j[ind]
			if type(v) != int and type(v) != float:
				vInd[ind] = v[ind]

		return Mat(iInd, jInd, vInd, nvert)

	def fillMat(self, nvert, nedge, i, j, v):
		ret = self.fillMat_worker(nvert, nedge, i, j, v)
		return ret

	def fillMatFiltered(self, nvert, nedge, i, j, v):
		ret = self.fillMat_worker(nvert, nedge, i, j, v)
		ret = self.addFilteredElements(ret)
		#ret.materializeFilter()
		return ret

	useFilterFill = False
	def initializeMat(self, nvert, nedge, i, j, v=1, allowFilter=True):
		if MatTests.useFilterFill and allowFilter:
			return self.fillMatFiltered(nvert, nedge, i, j, v)
		else:
			return self.fillMat(nvert, nedge, i, j, v)

	def initializeIJMat(self, nvert, scale=1000):
		"""
		Initialize a graph of size nvert*nvert with values equal to i*scale+j
		"""
		i = Vec.range(nvert*nvert) % nvert
		j = (Vec.range(nvert*nvert) / nvert).floor()
		v = i*scale + j
		return Mat(i, j, v, nvert)

	def assertEqualMat(self, G, expI, expJ, expV):
		self.assertEqual(len(expI), G.nnn())
		self.assertEqual(len(expJ), G.nnn())
		self.assertEqual(len(expV), G.nnn())

		self.assertEqual(G.ncol(), G.nrow())
		exp = self.initializeMat(G.ncol(), len(expI), expI, expJ, expV, allowFilter=False)
		self.assertEqual(G.nnn(), exp.nnn())
		comp = G.eWiseApply(exp, (lambda x,y: 1), doOp=(lambda x,y: x == y))
		self.assertEqual(comp.nnn(), G.nnn())

	def addFilteredElements(self, M):
		if M.isObj():
			print "NF" # make it known that the test wasn't done due to no object filters
			return M
			
		filteredValues = [-8000.1, 8000.1, 33, 66, -55]
		
		# remove elements that already exist in the matrix
		for v in filteredValues:
			if M.count(Mat.All, lambda x: x == v) > 0:
				filteredValues.remove(v)
		
		# add nodes
		offset = 0
		for v in filteredValues:
			n = min(M.ncol(), M.nrow())
			nc = M.ncol()
			rows = Vec.range(n)
			cols = Vec.range(n)
			cols.apply(lambda x: (x+offset)%nc) # move elements right by `offset`, wrapping around if needed
			offset += 1
		
			F = Mat(rows, cols, v, M.ncol(), M.nrow())
			#F = Mat.eye(n, nc, element=v)
			#print F
			def MoveFunc(m, f):
				if m != 0:
					return m
				else:
					return f
			M.eWiseApply(F, op=MoveFunc, allowANulls=True, allowBNulls=True, inPlace=True)
			
			# prune out the intersection between F and M
			#Fp = F.eWiseApply(M, op=(lambda f, m: f), allowANulls=False, allowBNulls=False, inPlace=False)
			#F.eWiseApply(Fp, op=(lambda f, p: f), allowANulls=False, allowBNulls=True, inPlace=True)
			#print F
			#M += F
		
		# add the filter that filters out the added nodes
		if M.isObj():
			M.addFilter(lambda e: filteredValues.count(e.weight) == 0)
		else:
			M.addFilter(lambda e: filteredValues.count(e) == 0)

		return M

class LinearAlgebraTests(MatTests):
	def test_matMul_1row1col(self):
		nvert1 = 16
		nedge1 = 4
		origI1 = [0, 0, 0, 0]
		origJ1 = [1, 3, 4, 12]
		origV1 = [1, 1, 1, 1]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		nvert2 = 16
		nedge2 = 4
		origI2 = [1, 3, 4, 12]
		origJ2 = [0, 0, 0, 0]
		origV2 = [1, 1, 1, 1]
		G2 = self.initializeMat(nvert2, nedge2, origI2, origJ2, origV2)
		G3 = G1.SpGEMM(G2, sr_plustimes)
		self.assertEqual(G1.ncol(), G3.ncol())
		[i3, j3, v3] = G3.toVec()
		expLen = 1
		self.assertEqual(len(i3),expLen)
		self.assertEqual(len(j3),expLen)
		self.assertEqual(len(v3),expLen)
		expectedI = [0]
		expectedJ = [0]
		expectedV = [4]

		for ind in range(len(expectedI)):
				self.assertEqual(i3[ind], expectedI[ind])
				self.assertEqual(j3[ind], expectedJ[ind])
				self.assertEqual(v3[ind], expectedV[ind])

	def disabled_test_matMul_simple(self):
		G = Mat.load('testfiles/small_nonsym_fp.mtx')
		[i, j, v] = G.toVec()
		print ""
		print "G.i:",i
		print "G.j:",j
		print "G.v:",v
		GT = G.copy()
		GT.transpose()
		G2 = G.SpGEMM(GT, sr_plustimes)
		self.assertEqual(G.ncol(),9)
		[i2, j2, v2] = G2.toVec()
		self.assertEqual(len(i2),30)
		self.assertEqual(len(j2),30)
		self.assertEqual(len(v2),30)
		expectedI = [0, 2, 3, 1, 2, 3, 4, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 
				1, 2, 4, 1, 6, 7, 1, 6, 7, 1, 8]
		expectedJ = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 
				4, 4, 4, 6, 6, 6, 7, 7, 7, 8, 8]
		expectedV = [0.0001, 0.0001, 0.0001, 0.001, 0.0001, 0.0001, 0.0001, 
				0.0001, 0.0001, 1.6e+8, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001,
				0.0001, 0.0001, 0.0001, 0.0003, 0.0001, 0.0001, 0.0001, 0.0001,
				0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.6e+8, 2.56e+20]

		for ind in range(len(expectedI)):
				self.assertEqual(i2[ind], expectedI[ind])
				self.assertEqual(j2[ind], expectedJ[ind])
				self.assertAlmostEqual(v2[ind], expectedV[ind], places=3)

	def test_SpGEMM_simple(self):
		G = Mat.load('testfiles/small.mtx')
		self.assertEqual(G.ncol(),4)
		self.assertEqual(G.nrow(),4)
		#[i, j, v] = G.toVec()
		GT = G.copy()
		GT.transpose()
		G2 = G.SpGEMM(GT, sr_plustimes)
		self.assertEqual(G2.ncol(),4)
		self.assertEqual(G2.nrow(),4)
		
		expI = [0,     2,     3,     1,     2,     3,     0,     1,     2,     3,     0,     1,     2,     3]
		expJ = [0,     0,     0,     1,     1,     1,     2,     2,     2,     2,     3,     3,     3,     3]
		expV = [4.0, 6.0,   11.0,  35.25, 4.0,   5.5,   6.0,   4.0,  13.0,  16.5,  11.0,   5.5,  16.5,  31.25]
		
		self.assertEqualMat(G2, expI, expJ, expV)

	def test_SpGEMM_simple_square(self):
		G = Mat.load('testfiles/small.mtx')
		G2 = G.SpGEMM(G, sr_plustimes)
		self.assertEqual(G2.ncol(),4)
		self.assertEqual(G2.nrow(),4)
		
		expI = [0,     2,     3,     1,     2,     3,     0,     1,     2,     3,     0,     1,     2,     3]
		expJ = [0,     0,     0,     1,     1,     1,     2,     2,     2,     2,     3,     3,     3,     3]
		expV = [2.0,  3.0,  5.5,  29.5,  11.0,   3.0,  11.0,   2.0,  18.5,  30.25,  4.0,  11.0,   6.0,  13.0]
		
		self.assertEqualMat(G2, expI, expJ, expV)
				
	def test_SpMV_simple_sparse(self):
		G = Mat.load('testfiles/small.mtx')
		vec = Vec(4, sparse=True)
		vec[1] = 2
		vec[3] = 5
		vec2 = G.SpMV(vec, sr_plustimes)
		expV = [4,    10,    16,    11]
		
		self.assertEqual(4, len(vec2))
		for ind in range(4):
			self.assertEqual(expV[ind], vec2[ind])

	def disabled_test_SpMV_simple_dense(self):
		# segfaults
		G = Mat.load('testfiles/small.mtx')
		vec = Vec(4, sparse=False)
		vec[1] = 2
		vec[3] = 5
		vec2 = G.SpMV(vec, sr_plustimes)
		expV = [4,    10,    16,    11]
		
		self.assertEqual(4, len(vec2))
		for ind in range(4):
			self.assertEqual(expV[ind], vec2[ind])
				
class ReductionTests(MatTests):
	def test_max_out_in(self):
		nvert = 9
		nedge = 19
		i = [0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 6, 8, 7, 8, 1, 1, 1, 1, 1]
		j = [1, 0, 2, 1, 3, 1, 3, 2, 4, 3, 8, 6, 8, 7, 4, 5, 6, 7, 8]
		v = [01, 10, 12, 21, 13, 31, 23, 32, 34, 43, 68, 1.6e10, 78, 87, 14,
				15, 16, 17, 18]
		G = self.initializeMat(nvert, nedge, i, j, v)
		self.assertEqual(G.ncol(), nvert)
		self.assertEqual(G.nrow(), nvert)
		self.assertEqual(G.nnn(), nedge)
		outmax = G.max(dir=Mat.Row)
		inmax = G.max(dir=Mat.Column)
		outmaxExpected = [1, 18, 23, 34, 43, 0, 68, 78, 1.6e10]
		inmaxExpected = [10, 31, 32, 43, 34, 15, 1.6e+10, 87, 78]
		self.assertEqual(len(outmax), len(outmaxExpected))
		self.assertEqual(len(inmax), len(inmaxExpected))

		for ind in range(len(outmax)):
				self.assertEqual(outmax[ind], outmaxExpected[ind])
				self.assertEqual(inmax[ind], inmaxExpected[ind])
		
	def test_min_out_in(self):
		nvert = 9
		nedge = 19
		i = [0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 6, 8, 7, 8, 1, 1, 1, 1, 1]
		j = [1, 0, 2, 1, 3, 1, 3, 2, 4, 3, 8, 6, 8, 7, 4, 5, 6, 7, 8]
		v = [-01, -10, -12, -21, -13, -31, -23, -32, -34, -43, -68, -1.6e10, 
				-78, -87, -14, -15, -16, -17, -18]
		G = self.initializeMat(nvert, nedge, i, j, v)
		self.assertEqual(G.ncol(), nvert)
		self.assertEqual(G.nrow(), nvert)
		self.assertEqual(G.nnn(), nedge)
		outmin = G.min(dir=Mat.Row)
		inmin = G.min(dir=Mat.Column)
		outminExpected = [-1, -18, -23, -34, -43, 0, -68, -78, -1.6e10]
		inminExpected = [-10, -31, -32, -43, -34, -15, -1.6e+10, -87, -78]
		self.assertEqual(len(outmin), len(outminExpected))
		self.assertEqual(len(inmin), len(inminExpected))

		for ind in range(len(outmin)):
				self.assertEqual(outmin[ind], outminExpected[ind])
				self.assertEqual(inmin[ind], inminExpected[ind])
		
	def test_sum_out_in(self):
		nvert = 9
		nedge = 19
		i = [0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 6, 8, 7, 8, 1, 1, 1, 1, 1]
		j = [1, 0, 2, 1, 3, 1, 3, 2, 4, 3, 8, 6, 8, 7, 4, 5, 6, 7, 8]
		v = [-01, -10, -12, -21, -13, -31, -23, -32, -34, -43, -68, -1.6e10, 
				-78, -87, -14, -15, -16, -17, -18]
		G = self.initializeMat(nvert, nedge, i, j, v)

		self.assertEqual(G.ncol(), nvert)
		self.assertEqual(G.nrow(), nvert)
		self.assertEqual(G.nnn(), nedge)
		outsum = G.sum(dir=Mat.Row)
		insum = G.sum(dir=Mat.Column)
		outsumExpected = [-1, -115, -44, -97, -43, 0, -68, -78, 
				-1.6000000087e+10]
		insumExpected = [-10, -53, -44, -79, -48, -15, -1.6000000016e+10, 
				-104, -164]
		print "rowsum:",outsum
		print "expected rowsum:",outsumExpected
		print "colsum:",insum
		print "expected colsum:", insumExpected
		
		self.assertEqual(len(outsum), len(outsumExpected))
		self.assertEqual(len(insum), len(insumExpected))

		for ind in range(len(outsum)):
				self.assertEqual(outsum[ind], outsumExpected[ind])
				self.assertEqual(insum[ind], insumExpected[ind])
				
class GeneralPurposeTests(MatTests):
	def test_multNot(self):
		nvert1 = 9
		nedge1 = 19
		origI1 = [1, 0, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
				17, 87, 8, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		nvert2 = 9
		nedge2 = 10
		origI2 = [7, 0, 4, 8, 5, 2, 7, 8, 1, 7]
		origJ2 = [0, 1, 1, 1, 2, 3, 5, 6, 7, 8]
		origV2 = [70, 1, 41, 81, 52, 23, 75, 8.6, 17, 78]
		G2 = self.initializeMat(nvert2, nedge2, origI2, origJ2, origV2)
		G3 = G1._mulNot(G2)

		expNvert = 9
		expNedge = 13
		expI = [1, 6, 1, 1, 3, 1, 3, 1, 1, 8, 0, 6]
		expJ = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8]
		expV = [10, 61, 12, 13, 33, 14, 34, 15, 1.6, 87, 8, 68]
		self.assertEqualMat(G3, expI, expJ, expV)

	def test_scale_row(self):
		nvert1 = 9
		nedge1 = 19
		origI1 = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
				17, 87, 8, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		vec1 = Vec.zeros(nvert1)
		# vec[0] null, scaling a null column in G1
		vec1[1] = 1
		vec1[2] = 2
		vec1[3] = 3
		vec1[4] = 4
		vec1[5] = 5
		# vec[6] null, scaling a non-null column in G1
		vec1[7] = 7
		vec1[8] = 8
		G1.scale(vec1, dir=Mat.Row)
		[actualI, actualJ, actualV] = G1.toVec()
		expI = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		expJ = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		expV = [0, 1, 164, 0, 12, 260, 13, 46, 99, 14, 102, 15, 1.6, 68.8,
				17, 696, 0, 0, 546]
		
		self.assertEqualMat(G1, expI, expJ, expV)

	def test_scale_column(self):
		nvert1 = 9
		nedge1 = 19
		origI1 = [0,  1,  4,  6,  1,  5,  1,  2,  3,  1,  3,  1,   1,   8,  1,  8, 0,  6, 7]
		origJ1 = [1,  1,  1,  1,  2,  2,  3,  3,  3,  4,  4,  5,   6,   6,  7,  7, 8,  8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6, 17, 87, 8, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		vec1 = Vec(nvert1, sparse=False)
		# vec[0] null, scaling a null column in G1
		vec1[1] = 1
		vec1[2] = 2
		vec1[3] = 3
		vec1[4] = 4
		vec1[5] = 5
		# vec[6] null, scaling a non-null column in G1
		vec1[7] = 7
		vec1[8] = 8
		G1.scale(vec1, dir=Mat.Column)
		[actualI, actualJ, actualV] = G1.toVec()
		expI = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		expJ = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		expV = [10, 1, 41, 61, 24, 104, 39, 69, 99, 56, 136, 75, 0, 0,
				119, 609, 64, 544, 624]
		self.assertEqualMat(G1, expI, expJ, expV)

class BuiltInMethodTests(MatTests):
	def test_add_simple(self):
		# ensure that Mat addition creates the number, source/
		# destination, and value pairs expected when all edges are 
		# in both Mats.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, (1.6e+10)+1,
				18, 88, 19, 69, 79]
		expV = [21, 3, 43, 63, 25, 65, 27, 47, 87, 29, 69, 31, 33, (3.2e+10)+1,
				35, 175, 37, 137, 157]
		G1 = self.initializeMat(nvert, nedge, origI, origJ, origV1)
		G2 = self.initializeMat(nvert, nedge, origI, origJ, origV2)
		G3 = G1+G2
		[actualI, actualJ, actualV] = G3.toVec()
		self.assertEqual(len(origI), len(actualI))
		self.assertEqual(len(origJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(origI)):
				self.assertEqual(origI[ind], actualI[ind])
				self.assertEqual(origJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_add_union(self):
		# ensure that Mat addition creates the number, source/
		# destination, and value pairs expected when some edges are not
		# in both Mats.
		nvert1 = 9
		nedge1 = 19
		origI1 = [1, 0, 2, 4, 1, 3, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 21, 41, 12, 32, 13, 23, 33, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		nvert2 = 9
		nedge2 = 19
		origI2 = [7, 3, 6, 8, 5, 7, 4, 5, 6, 5, 7, 7, 2, 7, 2, 7, 0, 2, 5]
		origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV2 = [70, 31, 61, 81, 52, 72, 43, 53, 63, 54, 74, 75, 26, 1.6e+10,
				27, 77, 8, 28, 58]
		G2 = self.initializeMat(nvert2, nedge2, origI2, origJ2, origV2)
		G3 = G1 + G2
		[actualI, actualJ, actualV] = G3.toVec()
		expNvert = 9
		expNedge = 38
		expI = [1, 7, 0, 2, 3, 4, 6, 8, 1, 3, 5, 7, 1, 2, 3, 4, 5, 6, 1, 3, 5,
				 7, 1, 7, 1, 2, 7, 8, 1, 2, 7, 8, 0, 1, 2, 5, 6, 7]
		expJ = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
				 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
		expV = [10,70, 1,21,31,41,61,81,12,32,52,72,13,23,33,43,53,63,14,34,54,
				74,15,75,16,26,1.6e+10,1.6e+10,17,27,77,87,8,18,28,58,68,78]
		[actualI, actualJ, actualV] = G3.toVec()
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_neg_simple(self):
		# ensure that Mat negation creates the number, source/
		# destination, and value pairs expected when all edges are 
		# in both Mats.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, -1, 21, -31, 12, -32, 13, -23, 43, -14, 34, -15, 16, 
				-1.6e+10, 17, -87, 18, -68, 78]
		expV   = [-10, 1, -21, 31, -12, 32, -13, 23, -43, 14, -34, 15, -16, 
				1.6e+10, -17, 87, -18, 68, -78]
		G1 = self.initializeMat(nvert, nedge, origI, origJ, origV1)
		G3 = -G1
		[actualI, actualJ, actualV] = G3.toVec()
		self.assertEqual(len(origI), len(actualI))
		self.assertEqual(len(origJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(origI)):
				self.assertEqual(origI[ind], actualI[ind])
				self.assertEqual(origJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_mul_simple(self):
		# ensure that Mat multiplication creates the number, source/
		# destination, and value pairs expected when all edges are 
		# in both Mats.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 7.3,
				17, 87, 18, 68, 78]
		origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, 8.3,
				18, 88, 19, 69, 79]
		expV = [110, 2, 462, 992, 156, 1056, 182, 552, 1892, 210, 1190, 240,
				272, 60.59, 306, 7656, 342, 4692, 6162]
		G1 = self.initializeMat(nvert, nedge, origI, origJ, origV1)
		G2 = self.initializeMat(nvert, nedge, origI, origJ, origV2)
		G3 = G1*G2
		[actualI, actualJ, actualV] = G3.toVec()
		self.assertEqual(len(origI), len(actualI))
		self.assertEqual(len(origJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(origI)):
				self.assertEqual(origI[ind], actualI[ind])
				self.assertEqual(origJ[ind], actualJ[ind])
				self.assertAlmostEqual(expV[ind], actualV[ind])
		
	def test_mul_intersection(self):
		# ensure that Mat multiplication creates the number, source/
		# destination, and value pairs expected when some edges are not
		# in both Mats.
		nvert1 = 9
		nedge1 = 19
		origI1 = [1, 0, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 16, 7.7,
				17, 87, 8, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		nvert2 = 9
		nedge2 = 19
		origI2 = [7, 3, 4, 8, 5, 7, 3, 5, 6, 3, 7, 7, 2, 8, 2, 7, 0, 2, 5]
		origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV2 = [70, 31, 41, 81, 52, 72, 33, 53, 63, 34, 74, 75, 26, 7.7,
				27, 77, 8, 28, 58]
		G2 = self.initializeMat(nvert2, nedge2, origI2, origJ2, origV2)
		G3 = G1*G2
		[actualI, actualJ, actualV] = G3.toVec()
		expNvert = 9
		expNedge = 6
		expI = [4, 5, 3, 3, 8, 0]
		expJ = [1, 2, 3, 4, 6, 8]
		expV = [1681, 2704, 1089, 1156, 59.29, 64]
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertAlmostEqual(expV[ind], actualV[ind])

	def test_imul_intersection(self):
		# ensure that Mat multiplication creates the number, source/
		# destination, and value pairs expected when some edges are not
		# in both Mats.
		nvert1 = 9
		nedge1 = 19
		origI1 = [1, 0, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 16, 7.7,
				17, 87, 8, 68, 78]
		G1 = self.initializeMat(nvert1, nedge1, origI1, origJ1, origV1)
		nvert2 = 9
		nedge2 = 19
		origI2 = [7, 3, 4, 8, 5, 7, 3, 5, 6, 3, 7, 7, 2, 8, 2, 7, 0, 2, 5]
		origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV2 = [70, 31, 41, 81, 52, 72, 33, 53, 63, 34, 74, 75, 26, 7.7,
				27, 77, 8, 28, 58]
		G2 = self.initializeMat(nvert2, nedge2, origI2, origJ2, origV2)
		G1 *= G2
		[actualI, actualJ, actualV] = G1.toVec()
		expNvert = 9
		expNedge = 6
		expI = [4, 5, 3, 3, 8, 0]
		expJ = [1, 2, 3, 4, 6, 8]
		expV = [1681, 2704, 1089, 1156, 59.29, 64]
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertAlmostEqual(expV[ind], actualV[ind])

	def test_div_simple(self):
		# ensure that Mat addition creates the number, source/
		# destination, and value pairs expected when all edges are 
		# in both Mats.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, (1.6e+10)+1,
				18, 88, 19, 69, 79]
		expV = [0.9090909091, 0.5, 0.9545454545, 0.96875, 0.92307692, 0.96969696, 
				0.92857142, 0.95833333, 0.97727272, 0.93333333, 0.97142857, 0.93750000, 
				0.94117647, 1, 0.94444444, 0.98863636, 0.94736842, 0.98550724, 0.98734177]
		G1 = self.initializeMat(nvert, nedge, origI, origJ, origV1)
		G2 = self.initializeMat(nvert, nedge, origI, origJ, origV2)
		G3 = G1/G2
		[actualI, actualJ, actualV] = G3.toVec()
		self.assertEqual(len(origI), len(actualI))
		self.assertEqual(len(origJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(origI)):
				self.assertEqual(origI[ind], actualI[ind])
				self.assertEqual(origJ[ind], actualJ[ind])
				self.assertAlmostEqual(expV[ind], actualV[ind])

	def disabled_test_indexing_simple_scalar_scalar(self):
		# ensure that a simple Mat constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeMat(nvert, nedge, origI, origJ, origV)
		ndx = 2
		G2 = G[ndx,ndx]
		[actualI, actualJ, actualV] = G2.toVec()
		expI = [0]
		expJ = [0]
		expV = [21]
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def disabled_test_indexing_simple_scalar_null(self):
		# ensure that a simple Mat constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeMat(nvert, nedge, origI, origJ, origV)
		ndx = 2
		G2 = G[ndx,ndx]
		[actualI, actualJ, actualV] = G2.toVec()
		expI = []
		expJ = []
		expV = []
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_indexing_simple_Veclen1_scalar(self):
		# ensure that a simple Mat constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeMat(nvert, nedge, origI, origJ, origV)
		ndx = Vec(1, sparse=False)
		ndx[0] = 2
		G2 = G[ndx,ndx]
		[actualI, actualJ, actualV] = G2.toVec()
		expI = [0]
		expJ = [0]
		expV = [21]
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_indexing_simple_Veclen1_null(self):
		# ensure that a simple Mat constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeMat(nvert, nedge, origI, origJ, origV)
		ndx = Vec(1, sparse=False)
		ndx[0] = 2
		G2 = G[ndx,ndx]
		[actualI, actualJ, actualV] = G2.toVec()
		expI = []
		expJ = []
		expV = []
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])
		
	def test_indexing_simple_Veclenk(self):
		# ensure that a simple Mat constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeMat(nvert, nedge, origI, origJ, origV)
		ndx = Vec(3, sparse=False)
		ndx[0] = 2
		ndx[1] = 3
		ndx[2] = 4
		G2 = G[ndx,ndx]
		[actualI, actualJ, actualV] = G2.toVec()
		expI = [1, 0, 2, 1]
		expJ = [0, 1, 1, 2]
		expV = [32, 23, 43, 34]
		self.assertEqual(len(expI), len(actualI))
		self.assertEqual(len(expJ), len(actualJ))
		self.assertEqual(len(expV), len(actualV))
		for ind in range(len(expI)):
				self.assertEqual(expI[ind], actualI[ind])
				self.assertEqual(expJ[ind], actualJ[ind])
				self.assertEqual(expV[ind], actualV[ind])



def runTests(verbosity = 1):
	testSuite = suite()
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

	print "running again using filtered data:"
	
	MatTests.useFilterFill = True
	#MatTests.fillMat = MatTests.fillMatFiltered
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
	suite = unittest.TestSuite()
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PageRankTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NormalizeEdgeWeightsTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(DegreeTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CentralityTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BFSTreeTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IsBFSTreeTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NeighborsTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PathsHopTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LoadTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ReductionTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInMethodTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LinearAlgebraTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ContractTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ApplyReduceTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(EdgeStatTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(SemanticGraphTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConnCompTests))
	return suite

if __name__ == '__main__':
	runTests()
