import unittest
#import sys
#print sys.prefix
#print sys.path
from kdt import *
#from kdt import pyCombBLAS as pcb

from TestMat import MatTests as MatTests

class DiGraphTests(unittest.TestCase):
	def initializeGraph(self, nvert, nedge, i, j, v=1):
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

		ret = DiGraph(iInd, jInd, vInd, nvert)
		ret = self.addFilterStuff(ret)
		return ret

	def initializeIJGraph(self, nvert, scale=1000):
		"""
		Initialize a graph of size nvert*nvert with values equal to i*scale+j
		"""
		i = Vec.range(nvert*nvert) % nvert
		j = (Vec.range(nvert*nvert) / nvert).floor()
		v = i*scale + j
		ret = DiGraph(i, j, v, nvert)
		ret = self.addFilterStuff(ret)
		return ret
	
	testFilter = False
	testMaterializingFilter = False
	def addFilterStuff(self, G):
		if self.testFilter:
			G.e = MatTests.addFilteredElements(G.e)
			if self.testMaterializingFilter:
				G.e.materializeFilter()
		return G
	
	def assertEqualMat(self, G, expI, expJ, expV, equalityCheck=None):
		if equalityCheck is None:
			def EQ(x,y):
				if x == y:
					return True
				else:
					print "element",x,"!=",y
					return False
			if expV is not None:
				equalityCheck = EQ
			else:
				equalityCheck = (lambda x,y: True) # just check nonzero structure

		self.assertEqual(len(expI), G.nnn())
		self.assertEqual(len(expJ), G.nnn())
		if expV is not None:
			self.assertEqual(len(expV), G.nnn())
		else:
			expV = 1

		self.assertEqual(G.ncol(), G.nrow())
		exp = MatTests.initializeMat(G.ncol(), len(expI), expI, expJ, expV, allowFilter=False)
		self.assertEqual(G.nnn(), exp.nnn())
		comp = G.eWiseApply(exp, (lambda x,y: G._identity_), doOp=equalityCheck)
		self.assertEqual(comp.nnn(), G.nnn())
	
	def assertEqualGraph(self, G, expI, expJ, expV):
		self.assertEqualMat(G.e, expJ, expI, expV) # I and J transposed intentionally

	def assertEqualVec(self, vec, expV, nn=()):
		self.assertEqual(len(vec), len(expV))
		for i in range(len(vec)):
			if vec[i] is None and expV[i] == 0 and i not in nn:
				pass
			else:
				if vec.isObj():
					val = vec[i].weight
				else:
					val = vec[i]
				if type(expV[i])==tuple:
					self.assertTrue(val in expV[i])
				else:
					self.assertEqual(val, expV[i])

class PageRankTests(DiGraphTests):
	def test_connected(self):
		G = DiGraph.fullyConnected(10)
		pr = G.pageRank()

		for prv in pr:
			self.assertAlmostEqual(0.1, prv, 7)

	def test_simple(self):
		# This test is drawn from the PageRank example at
		# http://en.wikipedia.org/wiki/File:PageRanks-Example.svg.
		nvert = 11
		nedge = 17
		i = [1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10]
		j = [2, 1, 0, 1, 1, 3, 5, 1, 4, 1, 4, 1, 4, 1, 4, 4,  4]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		
		G = self.initializeGraph(nvert, nedge, i, j)
		pr = G.pageRank(0.0001)

		expected = [0.032814, 0.38440, 0.34291, 0.03909, 0.08089, 0.03909, \
					0.01617, 0.01617, 0.01617, 0.01617, 0.01617]
		for ind in range(nvert):
			self.assertAlmostEqual(pr[ind], expected[ind], 4)

	def test_simple_loops(self):
		# This test is drawn from the PageRank example at
		# http://en.wikipedia.org/wiki/File:PageRanks-Example.svg.
		#
		# The difference between this and the previous test is that
		# this test includes several self loops to verify they have no
		# effect on the outcome.		
		nvert = 11
		nedge = 21
		i = [1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10]
		j = [1, 2, 1, 0, 1, 1, 3, 4, 5, 1, 4, 1, 4, 1, 4, 7, 1, 4, 4, 4,  10]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		
		G = self.initializeGraph(nvert, nedge, i, j)
		pr = G.pageRank(0.0001)

		expected = [0.032814, 0.38440, 0.34291, 0.03909, 0.08089, 0.03909, \
					0.01617, 0.01617, 0.01617, 0.01617, 0.01617]
		for ind in range(nvert):
			self.assertAlmostEqual(pr[ind], expected[ind], 4)

class NormalizeEdgeWeightsTests(DiGraphTests):
	def no_edge_graph(self):
		nvert = 4
		nedge = 0
		i = []
		j = []
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)

		return self.initializeGraph(nvert, nedge, i, j)

	def test_no_edges_default(self):
		G = self.no_edge_graph()
		G.normalizeEdgeWeights()
		self.assertEqual(G.nedge(), 0)

	def test_no_edges_out(self):
		G = self.no_edge_graph()
		G.normalizeEdgeWeights(DiGraph.Out)
		self.assertEqual(G.nedge(), 0)

	def test_no_edges_in(self):
		G = self.no_edge_graph()
		G.normalizeEdgeWeights(DiGraph.In)
		self.assertEqual(G.nedge(), 0)

	def small_test_graph(self):
		# 1 0 1 0
		# 0 0 0 1
		# 0 1 0 1
		# 1 0 0 0
		nvert = 4
		nedge = 6
		i = [0, 3, 2, 0, 1, 2]
		j = [0, 0, 1, 2, 3, 3]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)

		return [nvert, nedge, i, j, self.initializeGraph(nvert, nedge, i, j)]
		
	def test_small_default(self):
		[nvert, nedge, i, j, G] = self.small_test_graph()
		G.normalizeEdgeWeights()
		G.reverseEdges() # test was written before implicit transpose
		w = [0.5, 1., 0.5, 0.5, 1., 0.5]

		self.assertEqualMat(G.e, i, j, w)

	def test_small_out(self):
		[nvert, nedge, i, j, G] = self.small_test_graph()
		G.normalizeEdgeWeights(DiGraph.Out)
		G.reverseEdges() # test was written before implicit transpose
		w = [0.5, 1., 0.5, 0.5, 1., 0.5]

		self.assertEqualMat(G.e, i, j, w)

	def test_small_in(self):
		[nvert, nedge, i, j, G] = self.small_test_graph()
		G.normalizeEdgeWeights(DiGraph.In)
		G.reverseEdges() # test was written before implicit transpose
		w = [0.5, 0.5, 1., 1., 0.5, 0.5]

		self.assertEqualMat(G.e, i, j, w)

class DegreeTests(DiGraphTests):
	def test_outdegree_no_edges(self):
		nvert = 4
		nedge = 0
		i = []
		j = []
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)

		G = self.initializeGraph(nvert, nedge, i, j)
		inDeg = G.degree(DiGraph.Out)
		outDeg = G.degree(DiGraph.Out)
		for ind in range(nvert):
			self.assertEqual(inDeg[ind], 0)
			self.assertEqual(outDeg[ind], 0)
			
	def test_indegree_no_edges(self):
		nvert = 4
		nedge = 0
		i = []
		j = []
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)

		G = self.initializeGraph(nvert, nedge, i, j)
		deg = G.degree(DiGraph.In)
		for vdeg in deg:
			self.assertEqual(vdeg, 0)

	def test_simple(self):
		nvert = 11
		nedge = 17
		i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
		j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		
		G = self.initializeGraph(nvert, nedge, i, j)
		inDeg = G.degree(DiGraph.In)
		outDeg = G.degree(DiGraph.Out)
		inExpected = [0, 1, 1, 2, 3, 2, 2, 2, 2, 1, 1]
		outExpected = [1, 7, 1, 1, 6, 1, 0, 0, 0, 0, 0]

		for ind in range(nvert):
			self.assertEqual(inDeg[ind], inExpected[ind])
			self.assertEqual(outDeg[ind], outExpected[ind])

	def test_loop(self):
		nvert = 4
		nedge = 6
		i = [0, 3, 2, 2, 1, 3]
		j = [0, 0, 1, 2, 3, 3]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		
		G = self.initializeGraph(nvert, nedge, i, j)
		inDeg = G.degree(DiGraph.In)
		outDeg = G.degree(DiGraph.Out)
		inExpected = [2, 1, 1, 2]
		outExpected = [1, 1, 2, 2]

		for ind in range(nvert):
			self.assertEqual(inDeg[ind], inExpected[ind])
			self.assertEqual(outDeg[ind], outExpected[ind])

class CentralityTests(DiGraphTests):
	def test_exactBC_generate2DTorus(self):
		n = 16 #4
		G = DiGraph.generate2DTorus(n)
		G = self.addFilterStuff(G)
		nv = G.nvert()
		bc = G.centrality('exactBC',normalize=True)
		bcExpected = 0.0276826
		#bcExpected = 0.080952380952380956 # for generate2DTorus(4)
		for ind in range(nv):
			self.assertAlmostEqual(bc[ind],bcExpected, 6)		

	def dtest_approxBC_generate2DTorus(self):
		n = 16
		G = DiGraph.generate2DTorus(n)
		G = self.addFilterStuff(G)
		nv = G.nvert()
		bc = G.centrality('approxBC',sample=1.0, normalize=True)
		bcExpected = 0.0276826
		for ind in range(nv):
			self.assertAlmostEqual(bc[ind],bcExpected, 6)		

	def test_approxBC_generate2DTorus_sample(self):
		n = 16
		G = DiGraph.generate2DTorus(n)
		G = self.addFilterStuff(G)
		nv = G.nvert()
		bc = G.centrality('approxBC',sample=0.7, normalize=True)
		bcExpected = 0.0276
		for ind in range(nv):
			self.assertAlmostEqual(bc[ind],bcExpected,2)		

class BFSTreeTests(DiGraphTests):
	def test_bfsTree(self):
		nvert = 8
		nedge = 13
		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		parentsExpected = [-1, 1, 1, 4, 1, 2, (3,5), (2,4)]
		
		G = self.initializeGraph(nvert, nedge, i, j, 2)
		parents = G.bfsTree(1)
		self.assertEqualVec(parents, parentsExpected)

	def test_bfsTree_sym(self):
		nvert = 8
		nedge = 20
		i = [2, 4, 1, 5, 7, 4, 6, 7, 1, 3, 7, 2, 6, 7, 3, 5, 2, 3, 4, 5]
		j = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		parentsExpected = [-1, 1, 1, 4, 1, 2, (3,5), (2,4)]
		
		G = self.initializeGraph(nvert, nedge, i, j)
		parents = G.bfsTree(1)
		self.assertEqualVec(parents, parentsExpected)

	def test_bfsTree_Filtered(self):
		def gt0lt5(x):
			return x>0 and x<5
		nvert = 8
		nedge = 13
		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
		v = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		parentsExpected = [-1, 1, 1, 6, 1, 2, (5), (2,4)]
		
		G = self.initializeGraph(nvert, nedge, i, j, v)
		G.addEFilter(gt0lt5) # remove links to 3 from 4 and 7
		parents = G.bfsTree(1)
		self.assertEqualVec(parents, parentsExpected)

class IsBFSTreeTests(DiGraphTests):
	def test_isBfsTree(self):
		nvert = 8
		nedge = 13
		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		parentsExpected = [-1, 1, 1, 4, 1, 2, 5, 4]
		
		G = self.initializeGraph(nvert, nedge, i, j)
		root = 1
		parents = G.bfsTree(root)
		ret = G.isBfsTree(root, parents)
		self.assertTrue(type(ret)==tuple)
		[ret2, levels] = ret
		self.assertTrue(ret2)

class NeighborsTests(DiGraphTests):
	def test_neighbors(self):
		nvert = 8
		nedge = 13
		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		neighborsExpected = [0, 1, 0, 1, 0, 0, 0, 1]
		
		G = self.initializeGraph(nvert, nedge, i, j)
		neighbors = G.neighbors(4)
		self.assertEqualVec(neighbors, neighborsExpected)

	def test_neighbors_2hop(self):
		nvert = 8
		nedge = 12
		i = [1, 1, 2, 2, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 1, 3, 7, 6, 3, 3, 4, 5]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		neighborsExpected = [0, 1, 1, 1, 1, 1, 0, 1]
		
		G = self.initializeGraph(nvert, nedge, i, j)
		neighbors = G.neighbors(4, nhop=2)
		self.assertEqualVec(neighbors, neighborsExpected)

class PathsHopTests(DiGraphTests):
	def test_pathsHop(self):
		nvert = 8
		nedge = 13
		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		neighborsExpected = [-1, 4, -1, 4, -1, 2, -1, (2,4)]
		
		G = self.initializeGraph(nvert, nedge, i, j)
		starts = Vec(8, sparse=False)
		starts[2] = 1
		starts[4] = 1
		neighbors = G.pathsHop(starts)
		self.assertEqualVec(neighbors, neighborsExpected)

class LoadTests(DiGraphTests):
	def test_load(self):
		G = DiGraph.load('testfiles/small_nonsym_fp.mtx')
		self.assertEqual(G.nvert(),9)
		self.assertEqual(G.nedge(),19)
		expectedI = [1,0,2,3,5,5,6,7,8,1,3,1,2,4,3,8,8,6,7]
		expectedJ = [0,1,1,1,1,1,1,1,1,2,2,3,3,3,4,6,7,8,8]
		expectedV = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,1.6e+10,0.01]

		self.assertEqualMat(G.e, expectedI, expectedJ, expectedV)

	def test_load_bad_file(self):
		self.assertRaises(IOError, DiGraph.load, 'not_a_real_file.mtx')

	def test_UFget_simple_unsym(self):
		G = UFget('Pajek/CSphd')
		self.assertEqual(G.nvert(), 1882)
		self.assertEqual(G.nedge(), 1740)

	def test_UFget_simple_sym(self):
		G = UFget('Pajek/dictionary28')
		self.assertEqual(G.nvert(), 52652)
		self.assertEqual(G.nedge(), 89038)
		
class BuiltInMethodTests(DiGraphTests):
	def test_DiGraph_simple(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		self.assertEqualGraph(G, origI, origJ, origV)
				
	def test_subgraph_simple_scalar_scalar(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		ndx = 2
		G2 = G.subgraph(ndx,ndx)
		expI = [0]
		expJ = [0]
		expV = [21]
		self.assertEqualGraph(G2, expI, expJ, expV)
		
	def test_subgraph_simple_scalar_null(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		ndx = 2
		G2 = G.subgraph(ndx,ndx)
		expI = []
		expJ = []
		expV = []
		self.assertEqualGraph(G2, expI, expJ, expV)
		
	def test_subgraph_simple_Veclen1_scalar(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		ndx = Vec(1, sparse=False)
		ndx[0] = 2
		G2 = G.subgraph(ndx,ndx)
		expI = [0]
		expJ = [0]
		expV = [21]
		self.assertEqualGraph(G2, expI, expJ, expV)
		
	def test_subgraph_simple_Veclen1_null(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		ndx = Vec(1, sparse=False)
		ndx[0] = 2
		G2 = G.subgraph(ndx,ndx)
		expI = []
		expJ = []
		expV = []
		self.assertEqualGraph(G2, expI, expJ, expV)
		
	def test_subgraph_simple_Veclenk(self):
		# ensure that a simple DiGraph constructor creates the number, source/
		# destination, and value pairs expected.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		ndx = Vec(3, sparse=False)
		ndx[0] = 2
		ndx[1] = 3
		ndx[2] = 4
		G2 = G.subgraph(ndx,ndx)
		[actualI, actualJ, actualV] = G2.e.toVec()
		expI = [1, 0, 2, 1]
		expJ = [0, 1, 1, 2]
		expV = [32, 23, 43, 34]
		self.assertEqualGraph(G2, expI, expJ, expV)
		
	def test_DiGraph_duplicates(self):
		# ensure that a DiGraph constructor creates the number, source/
		# destination, and value pairs expected when 3 input edges have
		# the same source and destination.
		nvert = 9
		nedge = 19
		origI = [1, 0, 2, 3, 1, 3, 3, 3, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		expI = [1, 0, 2, 3, 1, 3, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
		expJ = [0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		expV = [10, 1, 21, 31, 12, 32, 79, 14, 34, 15, 16, 1.6e+10,
				17, 87, 18, 68, 78]
		G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
		self.assertEqualGraph(G, expI, expJ, expV)
				
class GeneralPurposeTests(DiGraphTests):
	def test_degree_in(self):
		nvert1 = 9
		nedge1 = 19
		origI1 = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
				17, 87, 8, 68, 78]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		deg = G1.degree(dir=DiGraph.In)
		expDeg = [0, 4, 2, 3, 2, 1, 2, 2, 3]
		self.assertEqual(len(expDeg), len(deg))
		for ind in range(len(expDeg)):
			self.assertEqual(expDeg[ind], deg[ind])

	def test_degree_out(self):
		nvert1 = 9
		nedge1 = 19
		origI1 = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
		origJ1 = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
		origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
				17, 87, 8, 68, 78]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		deg = G1.degree(dir=DiGraph.Out)
		expDeg = [2, 7, 1, 2, 1, 1, 2, 1, 2]
		self.assertEqual(len(expDeg), len(deg))
		for ind in range(len(expDeg)):
			self.assertEqual(expDeg[ind], deg[ind])

class ContractTests(DiGraphTests):
	def test_contract_simple(self):
		nvert1 = 5
		nedge1 = 5
		origI1 = [4, 0, 0, 1, 2]
		origJ1 = [0, 1, 2, 3, 4]
		origV1 = [1, 1, 1, 1, 1]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		groups = Vec(5, sparse=False)
		groups[0] = 0
		groups[1] = 0
		groups[2] = 0
		groups[3] = 1
		groups[4] = 1
		smallG = G1.contract(groups)
		self.assertEqual(smallG.nvert(), 2)
		expectedI = [0, 1, 0]
		expectedJ = [0, 0, 1]
		expectedV = None # [4, 3, 3]
		
		self.assertEqualGraph(smallG, expectedI, expectedJ, expectedV)

class EdgeStatTests(DiGraphTests):
	def test_nedge_simple(self):
		nvert1 = 5
		nedge1 = 5
		origI1 = [4, 0, 0, 1, 2]
		origJ1 = [0, 1, 2, 3, 4]
		origV1 = [1, 1, 1, 1, 1]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		ne = G1.nedge()
		expNe = 5
		self.assertEqual(expNe, ne)

	def test_nvert_simple(self):
		nvert1 = 5
		nedge1 = 5
		origI1 = [4, 0, 0, 1, 2]
		origJ1 = [0, 1, 2, 3, 4]
		origV1 = [1, 1, 1, 1, 1]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		nv = G1.nvert()
		expNv = 5
		self.assertEqual(nv,expNv)

	def disabled_test_nedge_vpart_simple(self):
		nvert1 = 5
		nedge1 = 5
		origI1 = [4, 0, 0, 1, 2]
		origJ1 = [0, 1, 2, 3, 4]
		origV1 = [1, 1, 1, 1, 1]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		vpart = Vec(5, sparse=False)
		vpart[0] = 0
		vpart[1] = 0
		vpart[2] = 0
		vpart[3] = 1
		vpart[4] = 1
		ne = G1.nedge(vpart)
		expLen = 2
		self.assertEqual(len(ne),expLen)
		expectedNe = [2, 0]

		for ind in range(len(expectedNe)):
			self.assertEqual(ne[ind], expectedNe[ind])

	def disabled_test_nvert_vpart_simple(self):
		nvert1 = 5
		nedge1 = 5
		origI1 = [4, 0, 0, 1, 2]
		origJ1 = [0, 1, 2, 3, 4]
		origV1 = [1, 1, 1, 1, 1]
		G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
		vpart = Vec(5, sparse=False)
		vpart[0] = 0
		vpart[1] = 0
		vpart[2] = 0
		vpart[3] = 1
		vpart[4] = 1
		nv = G1.nvert(vpart)
		expLen = 2
		self.assertEqual(len(nv),expLen)
		expectedNv = [3, 2]

		for ind in range(len(expectedNv)):
			self.assertEqual(nv[ind], expectedNv[ind])

class ConnCompTests(DiGraphTests):
	def test_connComp(self):
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		res = G.connComp()

		resExpected = [0, 7, 7, 7, 7, 7, 7, 7]
		self.assertEqual(G.nvert(), len(res))
		for ind in range(nvert):
			self.assertEqual(resExpected[ind], res[ind])

class SemanticGraphTests(DiGraphTests):
	def test_addVFilter_copy(self):
		def ge0lt4(x):
				return x>=0 and x<4
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		G.addVFilter(ge0lt4)

		G2 = G.copy(copyFilter=True)
		[actualI, actualJ, ign] = G2.e.toVec()
		#print actualI, actualJ
		iExpected = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		jExpected = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(nvert, G2.nvert())
		self.assertEqual(G.nedge(), G2.nedge())
		self.assertEqual(G.nedge(), len(actualI))
		self.assertEqual(True, hasattr(G2,'_vFilter'))
		for ind in range(nvert):
			self.assertEqual(iExpected[ind], i[ind])
			self.assertEqual(jExpected[ind], j[ind])

	def test_addVFilter_copy_1filter(self):
		def eq1or2(x):
				return x==1 or x==2
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		G.addVFilter(eq1or2)
		G.vType = (Vec.range(nvert) % 3) + 1

		G2 = G.copy(copyFilter=True, doFilter=True)
		[actualI, actualJ, ign] = G2.e.toVec()
		iExpected = [3, 3, 4, 5, 1, 5, 2, 3]
		jExpected = [1, 2, 2, 2, 3, 3, 4, 5]
		self.assertEqual(6, G2.nvert())
		self.assertEqual(8, G2.nedge())
		self.assertEqual(G2.nedge(), len(actualI))
		for ind in range(len(iExpected)):
			self.assertEqual(iExpected[ind], actualI[ind])
			self.assertEqual(jExpected[ind], actualJ[ind])

	def test_addVFilter_copy_2filter(self):
		def eq1or2(x):
				return x==1 or x==2
		def gt1(x):
				return x>1
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		G.addVFilter(eq1or2)
		G.addVFilter(gt1)
		G.vType = (Vec.range(nvert) % 3) + 1

		G2 = G.copy(copyFilter=True, doFilter=True)
		[actualI, actualJ, ign] = G2.e.toVec()
		iExpected = [1, 0, 2, 1]
		jExpected = [0, 1, 1, 2]
		self.assertEqual(3, G2.nvert())
		self.assertEqual(4, G2.nedge())
		self.assertEqual(G2.nedge(), len(actualI))
		for ind in range(len(iExpected)):
			self.assertEqual(iExpected[ind], actualI[ind])
			self.assertEqual(jExpected[ind], actualJ[ind])

	def test_addVFilter_delVFilter(self):
		def eq1or2(x):
				return x==1 or x==2
		def gt1(x):
				return x>1
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		G.addVFilter(eq1or2)
		G.addVFilter(gt1)
		G.delVFilter(eq1or2)
		self.assertEqual(1, len(G._vFilter))
		G.delVFilter(gt1)
		self.assertEqual(0, len(G._vFilter))

	def test_addVFilter_delVFilter_all(self):
		def eq1or2(x):
				return x==1 or x==2
		def gt1(x):
				return x>1
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		G.addVFilter(eq1or2)
		G.addVFilter(gt1)
		G.delVFilter()
		self.assertEqual(False, hasattr(G,'_vFilter'))

	def test_delVFilter_negative(self):
		def eq1or2(x):
				return x==1 or x==2
		nvert = 8
		nedge = 13
		i = [4, 1, 4, 6, 7, 1, 7, 2, 7, 3, 5, 2, 4]
		j = [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
		self.assertEqual(len(i), nedge)
		self.assertEqual(len(j), nedge)
		G = self.initializeGraph(nvert, nedge, i, j)

		self.assertRaises(KeyError, DiGraph.delVFilter, G, eq1or2)

#	def test_semG_efilter_bfsTree(self):
#		def ge0lt5000(x):
#				return x%1000>=0 and x%1000<5 and int(x/1000)>=0 and int(x/1000)<5
#		nvert = 8
#		nedge = 13
#		i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
#		j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
#		v = i[:]				# easy way to get same-size vector
#		for ind in range(nedge):
#				v[ind] = i[ind]*1000 + j[ind]
#		G = self.initializeGraph(nvert, nedge, i, j, v)
#		parents = G.bfsTree(1, filter=(None, ge0lt5000))
#		parentsExpected = [-1, 1, 1, 4, 1, -1, -1, -1]
#		for ind in range(nvert):
#				self.assertEqual(parents[ind], parentsExpected[ind])


def runTests(verbosity = 1):
	testSuite = suite()
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)
	
	print "running again using filtered data (on-the-fly):"
	DiGraphTests.testFilter = True
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)
	
	print "running again using filtered data (materializing):"
	DiGraphTests.testMaterializingFilter = True
	unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
	suite = unittest.TestSuite()
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PageRankTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NormalizeEdgeWeightsTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(DegreeTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BFSTreeTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IsBFSTreeTests)) # bad, isBFSTree() not updated to transposed mat
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NeighborsTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PathsHopTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LoadTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ContractTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(EdgeStatTests))
	suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConnCompTests))
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInMethodTests)) # One fail, on duplicates in constructor
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CentralityTests)) # OK, just slow
	#suite.addTests(unittest.TestLoader().loadTestsFromTestCase(SemanticGraphTests)) # tests are out of date, may be obsoleted by OTF/Mat repeat runs
	return suite

if __name__ == '__main__':
	runTests()
