import unittest
from kdt import *
from kdt import pyCombBLAS as pcb

class DiGraphTests(unittest.TestCase):
    def initializeGraph(self, nvert, nedge, i, j, v=1):
        """
        Initialize a graph with edge weights equal to one or the input value.
        """
        iInd = ParVec(nedge)
        jInd = ParVec(nedge)
	if type(v) == int or type(v) == float:
            vInd = ParVec(nedge, v)
	else:
	    vInd = ParVec(nedge)
        for ind in range(nedge):
            iInd[ind] = i[ind]
            jInd[ind] = j[ind]
	    if type(v) != int and type(v) != float:
		vInd[ind] = v[ind]

        return DiGraph(iInd, jInd, vInd, nvert)

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
        [iInd, jInd, eW] = G.toParVec()
        w = [0.5, 1., 0.5, 0.5, 1., 0.5]

        for ind in range(nedge):
            self.assertEqual(i[ind], iInd[ind])
            self.assertEqual(j[ind], jInd[ind])
            self.assertEqual(eW[ind], w[ind])

    def test_small_out(self):
        [nvert, nedge, i, j, G] = self.small_test_graph()
        G.normalizeEdgeWeights(DiGraph.Out)
        [iInd, jInd, eW] = G.toParVec()
        w = [0.5, 1., 0.5, 0.5, 1., 0.5]

        for ind in range(nedge):
            self.assertEqual(i[ind], iInd[ind])
            self.assertEqual(j[ind], jInd[ind])
            self.assertEqual(eW[ind], w[ind])

    def test_small_in(self):
        [nvert, nedge, i, j, G] = self.small_test_graph()
        G.normalizeEdgeWeights(DiGraph.In)
        [iInd, jInd, eW] = G.toParVec()
        w = [0.5, 0.5, 1., 1., 0.5, 0.5]

        for ind in range(nedge):
            self.assertEqual(i[ind], iInd[ind])
            self.assertEqual(j[ind], jInd[ind])
            self.assertEqual(eW[ind], w[ind])

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
    def test_exactBC_twoDTorus(self):
	n = 16
	G = DiGraph.twoDTorus(n)
	nv = G.nvert()
	bc = G.centrality('exactBC',normalize=True)
	bcExpected = 0.0276826
	for ind in range(nv):
		self.assertAlmostEqual(bc[ind],bcExpected, 6)	

    def test_approxBC_twoDTorus(self):
	n = 16
	G = DiGraph.twoDTorus(n)
	nv = G.nvert()
	bc = G.centrality('approxBC',sample=1.0, normalize=True)
	bcExpected = 0.0276826
	for ind in range(nv):
		self.assertAlmostEqual(bc[ind],bcExpected, 6)	

    def test_approxBC_twoDTorus_sample(self):
	n = 16
	G = DiGraph.twoDTorus(n)
	nv = G.nvert()
	bc = G.centrality('approxBC',sample=0.05, normalize=True)
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
	parentsExpected = [-1, 1, 1, 4, 1, 2, 5, 4]
        
        G = self.initializeGraph(nvert, nedge, i, j)
	parents = G.bfsTree(1)
	for ind in range(nvert):
		self.assertEqual(parents[ind], parentsExpected[ind])

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

class LoadTests(DiGraphTests):
    def test_load(self):
	G = DiGraph.load('testfiles/small_nonsym_fp.mtx')
	self.assertEqual(G.nvert(),9)
	self.assertEqual(G.nedge(),18)
	[i, j, v] = G.toParVec()
	self.assertEqual(len(i),18)
	self.assertEqual(len(j),18)
	self.assertEqual(len(v),18)
	expectedI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 3, 1, 1, 8, 1, 8, 1, 6, 7]
	expectedJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	expectedV = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
			0.02, 0.01, 0.01, 1.6e+10, 0.01, 0.01, 0.01, 0.01, 0.01]

	for ind in range(len(i)):
		self.assertEqual(i[ind], expectedI[ind])
		self.assertEqual(j[ind], expectedJ[ind])
		self.assertEqual(v[ind], expectedV[ind])

    def test_load_bad_file(self):
        self.assertRaises(IOError, DiGraph.load, 'not_a_real_file.mtx')

class MaxTests(DiGraphTests):
    def test_max_out(self):
	nvert = 9
	nedge = 19
	i = [0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 6, 8, 7, 8, 1, 1, 1, 1, 1]
	j = [1, 0, 2, 1, 3, 1, 3, 2, 4, 3, 8, 6, 8, 7, 4, 5, 6, 7, 8]
	v = [01, 10, 12, 21, 13, 31, 23, 32, 34, 43, 68, 1.6e10, 78, 87, 14,
		15, 16, 17, 18]
        G = self.initializeGraph(nvert, nedge, i, j, v)
	self.assertEqual(G.nvert(), nvert)
	self.assertEqual(G.nedge(), nedge)
	outmax = G.max(dir=DiGraph.Out)
	inmax = G.max(dir=DiGraph.In)
	outmaxExpected = [1, 18, 23, 34, 43, 0, 68, 78, 1.6e10]
	inmaxExpected = [10, 31, 32, 43, 34, 15, 1.6e+10, 87, 78]
	self.assertEqual(len(outmax), len(outmaxExpected))
	self.assertEqual(len(inmax), len(inmaxExpected))

	for ind in range(len(outmax)):
		self.assertEqual(outmax[ind], outmaxExpected[ind])
		self.assertEqual(inmax[ind], inmaxExpected[ind])
	
class MinTests(DiGraphTests):
    def test_min_out(self):
	nvert = 9
	nedge = 19
	i = [0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 6, 8, 7, 8, 1, 1, 1, 1, 1]
	j = [1, 0, 2, 1, 3, 1, 3, 2, 4, 3, 8, 6, 8, 7, 4, 5, 6, 7, 8]
	v = [-01, -10, -12, -21, -13, -31, -23, -32, -34, -43, -68, -1.6e10, 
		-78, -87, -14, -15, -16, -17, -18]
        G = self.initializeGraph(nvert, nedge, i, j, v)
	self.assertEqual(G.nvert(), nvert)
	self.assertEqual(G.nedge(), nedge)
	outmin = G.min(dir=DiGraph.Out)
	inmin = G.min(dir=DiGraph.In)
	outminExpected = [-1, -18, -23, -34, -43, 0, -68, -78, -1.6e10]
	inminExpected = [-10, -31, -32, -43, -34, -15, -1.6e+10, -87, -78]
	self.assertEqual(len(outmin), len(outminExpected))
	self.assertEqual(len(inmin), len(inminExpected))

	for ind in range(len(outmin)):
		self.assertEqual(outmin[ind], outminExpected[ind])
		self.assertEqual(inmin[ind], inminExpected[ind])
	
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
	[actualI, actualJ, actualV] = G.toParVec()
        self.assertEqual(len(origI), len(actualI))
        self.assertEqual(len(origJ), len(actualJ))
        self.assertEqual(len(origV), len(actualV))
        for ind in range(len(origI)):
		self.assertEqual(origI[ind], actualI[ind])
		self.assertEqual(origJ[ind], actualJ[ind])
		self.assertEqual(origV[ind], actualV[ind])
        
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
	[actualI, actualJ, actualV] = G.toParVec()
        self.assertEqual(len(expI), len(actualI))
        self.assertEqual(len(expJ), len(actualJ))
        self.assertEqual(len(expV), len(actualV))
        for ind in range(len(origI)):
		self.assertEqual(expI[ind], actualI[ind])
		self.assertEqual(expJ[ind], actualJ[ind])
		self.assertEqual(expV[ind], actualV[ind])
        
    def test_add_simple(self):
	# ensure that a DiGraph constructor creates the number, source/
        # destination, and value pairs expected when all edges are 
        # in both DiGraphs.
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
	G1 = self.initializeGraph(nvert, nedge, origI, origJ, origV1)
	G2 = self.initializeGraph(nvert, nedge, origI, origJ, origV2)
        G3 = G1+G2
	[actualI, actualJ, actualV] = G3.toParVec()
        self.assertEqual(len(origI), len(actualI))
        self.assertEqual(len(origJ), len(actualJ))
        self.assertEqual(len(expV), len(actualV))
        for ind in range(len(origI)):
		self.assertEqual(origI[ind], actualI[ind])
		self.assertEqual(origJ[ind], actualJ[ind])
		self.assertEqual(expV[ind], actualV[ind])
        
    def test_add_union(self):
	# ensure that a DiGraph constructor creates the number, source/
        # destination, and value pairs expected when some edges are not
        # in both DiGraphs.
	nvert1 = 9
	nedge1 = 19
	origI1 = [1, 0, 2, 4, 1, 3, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
	origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	origV1 = [10, 1, 21, 41, 12, 32, 13, 23, 33, 14, 34, 15, 16, 1.6e+10,
		17, 87, 18, 68, 78]
	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
	nvert2 = 9
	nedge2 = 19
	origI2 = [7, 3, 6, 8, 5, 7, 4, 5, 6, 5, 7, 7, 2, 7, 2, 7, 0, 2, 5]
	origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	origV2 = [70, 31, 61, 81, 52, 72, 43, 53, 63, 54, 74, 75, 26, 1.6e+10,
		27, 77, 8, 28, 58]
	G2 = self.initializeGraph(nvert2, nedge2, origI2, origJ2, origV2)
        G3 = G1 + G2
	[actualI, actualJ, actualV] = G3.toParVec()
	expNvert = 9
	expNedge = 38
        expI = [1, 7, 0, 2, 3, 4, 6, 8, 1, 3, 5, 7, 1, 2, 3, 4, 5, 6, 1, 3, 5,
		 7, 1, 7, 1, 2, 7, 8, 1, 2, 7, 8, 0, 1, 2, 5, 6, 7]
        expJ = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
		 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
	expV = [10,70, 1,21,31,41,61,81,12,32,52,72,13,23,33,43,53,63,14,34,54,
		74,15,75,16,26,1.6e+10,1.6e+10,17,27,77,87,8,18,28,58,68,78]
	[actualI, actualJ, actualV] = G3.toParVec()
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

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PageRankTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NormalizeEdgeWeightsTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(DegreeTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CentralityTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BFSTreeTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IsBFSTreeTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LoadTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MaxTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MinTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInMethodTests))
    return suite

if __name__ == '__main__':
    runTests()
