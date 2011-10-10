import unittest
from kdt import *
from kdt import pyCombBLAS as pcb

class HyGraphTests(unittest.TestCase):
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

        return HyGraph(iInd, jInd, vInd, nvert)

class ConstructorTests(HyGraphTests):
    def test_toDiGraph(self):
	nvert = 7
	nSEdge = 9	# #SimpleEdge
	origI = [0, 0, 1, 1, 2, 1, 2, 2, 3]
	origJ = [1, 2, 2, 3, 3, 4, 4, 5, 6]
	origV = 1
	G = self.initializeGraph(nvert, nSEdge, origI, origJ, origV)
        diG = G.toDiGraph()
	[di, dj, dv] = diG.toParVec()
        diExpected = [1, 2, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 6]
	djExpected = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
	dvExpected = [1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1]
	self.assertEqual(7, diG.nvert())
	self.assertEqual(18, diG.nedge())
	self.assertEqual(18, len(diExpected))

        for ind in range(nvert):
            self.assertEqual(diExpected[ind], di[ind])
            self.assertEqual(djExpected[ind], dj[ind])
            self.assertEqual(dvExpected[ind], dv[ind])


class PageRankTests(HyGraphTests):
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

class NormalizeEdgeWeightsTests(HyGraphTests):
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

class DegreeTests(HyGraphTests):
    def test_degree_no_edges(self):
        nvert = 4
        nedge = 0
        i = []
        j = []
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)

        G = self.initializeGraph(nvert, nedge, i, j)
        deg = G.degree()
        for ind in range(nvert):
            self.assertEqual(deg[ind], 0)
            
    def test_degree_simple(self):
        nvert = 11
        nedge = 17
        i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
        j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)
        
        G = self.initializeGraph(nvert, nedge, i, j)
        deg = G.degree()
        degExpected = [0, 1, 1, 2, 3, 2, 2, 2, 2, 1, 1]

        for ind in range(nvert):
            self.assertEqual(deg[ind], degExpected[ind])

    def test_npin_no_edges(self):
        nvert = 4
        nedge = 0
        i = []
        j = []
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)

        G = self.initializeGraph(nvert, nedge, i, j)
        npin = G.npin()
        for ind in range(nedge):
            self.assertEqual(npin[ind], 0)
            
    def test_npin_simple(self):
        nSvert = 11
        nSedge = 17
        i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
        j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
        self.assertEqual(len(i), nSedge)
        self.assertEqual(len(j), nSedge)
        
        G = self.initializeGraph(nSvert, nSedge, i, j)
	nvert = G.nvert()
        self.assertEqual(11, nvert)
	nedge = G.nedge()
        self.assertEqual(6, nedge)
        npin = G.npin()
        npinExpected = [1, 7, 1, 1, 6, 1]

        for ind in range(nedge):
            self.assertEqual(npin[ind], npinExpected[ind])

    def test_rank_no_edges(self):
        nvert = 4
        nedge = 0
        i = []
        j = []
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)

        G = self.initializeGraph(nvert, nedge, i, j)
        rank = G.rank()
        self.assertEqual(rank, 0)
            
    def test_rank_simple(self):
        nSvert = 11
        nSedge = 17
        i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
        j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
        self.assertEqual(len(i), nSedge)
        self.assertEqual(len(j), nSedge)
        
        G = self.initializeGraph(nSvert, nSedge, i, j)
	nvert = G.nvert()
        self.assertEqual(11, nvert)
	nedge = G.nedge()
        self.assertEqual(6, nedge)
        rank = G.rank()
        self.assertEqual(rank, 7)

    def test_antirank_no_edges(self):
        nvert = 4
        nedge = 0
        i = []
        j = []
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)

        G = self.initializeGraph(nvert, nedge, i, j)
        antirank = G.antirank()
        self.assertEqual(antirank, 0)
            
    def test_antirank_simple(self):
        nSvert = 11
        nSedge = 17
        i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
        j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
        self.assertEqual(len(i), nSedge)
        self.assertEqual(len(j), nSedge)
        
        G = self.initializeGraph(nSvert, nSedge, i, j)
	nvert = G.nvert()
        self.assertEqual(11, nvert)
	nedge = G.nedge()
        self.assertEqual(6, nedge)
        antirank = G.antirank()
        self.assertEqual(antirank, 1)

class CentralityTests(HyGraphTests):
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

class BFSTreeTests(HyGraphTests):
    def test_bfsTree(self):
	nvert = 7
	nSEdge = 9	# #SimpleEdge
	origI = [0, 0, 1, 1, 2, 1, 2, 2, 3]
	origJ = [1, 2, 2, 3, 3, 4, 4, 5, 6]
	origV = 1
	G = self.initializeGraph(nvert, nSEdge, origI, origJ, origV)
	root = 1
	parentsExpected = [-1, 1, 1, 2, 2, 4, -1]
	parents = G.bfsTree(root)
        self.assertEqual(len(parentsExpected), len(parents))
        for ind in range(len(parents)):
		self.assertEqual(parentsExpected[ind], parents[ind])


class IsBFSTreeTests(HyGraphTests):
    def test_isBfsTree(self):
	nvert = 7
	nSEdge = 9	# #SimpleEdge
	origI = [0, 0, 1, 1, 2, 1, 2, 2, 3]
	origJ = [1, 2, 2, 3, 3, 4, 4, 5, 6]
	origV = 1
	G = self.initializeGraph(nvert, nSEdge, origI, origJ, origV)
	root = 1
	parents = G.bfsTree(root)
	ret = G.isBfsTree(root, parents)
	self.assertTrue(type(ret)==tuple)
	[ret2, levels] = ret
	self.assertTrue(ret2)
	levelsExpected = [-1, 0, 1, 2, 2, 3, -1]
	self.assertEqual(len(levelsExpected),len(levels))
	for i in range(len(levels)):
		self.assertEqual(levelsExpected[i],levels[i])

class NeighborsTests(HyGraphTests):
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
	for ind in range(nvert):
		self.assertEqual(neighbors[ind], neighborsExpected[ind])

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
	for ind in range(nvert):
		self.assertEqual(neighbors[ind], neighborsExpected[ind])

class PathsHopTests(HyGraphTests):
    def test_pathsHop(self):
        nvert = 8
        nedge = 13
        i = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
        j = [2, 4, 5, 7, 6, 1, 3, 7, 6, 3, 3, 4, 5]
        self.assertEqual(len(i), nedge)
        self.assertEqual(len(j), nedge)
	neighborsExpected = [-1, 4, -1, 4, -1, 2, -1, 4]
        
        G = self.initializeGraph(nvert, nedge, i, j)
	tmp = ParVec.range(8)
        starts = (tmp == 2) | (tmp == 4)
	neighbors = G.pathsHop(starts)
	for ind in range(nvert):
		self.assertEqual(neighbors[ind], neighborsExpected[ind])

class LoadTests(HyGraphTests):
    def test_load(self):
	G = HyGraph.load('testfiles/UFlorida_Pajek_Sandi_sandi.mtx')
	G._T();			# swap so edges are papers, not authors
	self.assertEqual(G.nvert(),314)
	self.assertEqual(G.nedge(),360)
	[i, j, v] = G.toParVec()
	self.assertEqual(len(i),613)
	self.assertEqual(len(j),613)
	self.assertEqual(len(v),613)
	expectedNdx = [100, 200, 300, 400, 500, 600]
	expectedI =   [ 23,  13, 222, 244, 310, 352]
	expectedJ =   [ 63, 121, 155, 204, 252, 309]
	expectedV =   1

	for ind in range(len(expectedNdx)):
		self.assertEqual(i[expectedNdx[ind]], expectedI[ind])
		self.assertEqual(j[expectedNdx[ind]], expectedJ[ind])
		self.assertEqual(v[expectedNdx[ind]], expectedV)

    def test_load_bad_file(self):
        self.assertRaises(IOError, DiGraph.load, 'not_a_real_file.mtx')

#    def test_UFget_simple_unsym(self):
#	G = UFget('Pajek/CSphd')
#        self.assertEqual(G.nvert(), 1882)
#	self.assertEqual(G.nedge(), 1740)
#
#    def test_UFget_simple_sym(self):
#	G = UFget('Pajek/dictionary28')
#        self.assertEqual(G.nvert(), 52652)
#	self.assertEqual(G.nedge(), 178076)

class MaxTests(HyGraphTests):
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
	
class MinTests(HyGraphTests):
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
	
class BuiltInMethodTests(HyGraphTests):
    def test_HyGraph_simple(self):
	# ensure that a simple HyGraph constructor creates the number, source/
        # destination, and value pairs expected.
	nvert = 7
	nSEdge = 9	# #SimpleEdge
	origI = [0, 0, 1, 1, 2, 1, 2, 2, 3]
	origJ = [1, 2, 2, 3, 3, 4, 4, 5, 6]
	origV = [1, 2, 12, 13, 23, 14, 24, 25, 46]
	G = self.initializeGraph(nvert, nSEdge, origI, origJ, origV)
	self.assertEqual(7, G.nvert())
	self.assertEqual(4, G.nedge())
	[actualI, actualJ, actualV] = G.toParVec()
        self.assertEqual(len(origI), len(actualI))
        self.assertEqual(len(origJ), len(actualJ))
        self.assertEqual(len(origV), len(actualV))
        for ind in range(len(origI)):
		self.assertEqual(origI[ind], actualI[ind])
		self.assertEqual(origJ[ind], actualJ[ind])
		self.assertEqual(origV[ind], actualV[ind])
        
    def test_HyGraph_no_verts(self):
        nvert = 0
        nedge = 6
        i = [0, 3, 2, 2, 1, 3]
        j = [0, 0, 1, 2, 3, 3]
        self.assertRaises(KeyError, self.initializeGraph, nvert, nedge, i, j)

#    def test_indexing_simple_scalar_scalar(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = 2
#	G2 = G[ndx,ndx]
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [0]
#	expJ = [0]
#	expV = [21]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_indexing_simple_scalar_null(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = 2
#	G2 = G[ndx,ndx]
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = []
#	expJ = []
#	expV = []
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_indexing_simple_ParVeclen1_scalar(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(1)
#	ndx[0] = 2
#	G2 = G[ndx,ndx]
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [0]
#	expJ = [0]
#	expV = [21]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_indexing_simple_ParVeclen1_null(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(1)
#	ndx[0] = 2
#	G2 = G[ndx,ndx]
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = []
#	expJ = []
#	expV = []
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_indexing_simple_ParVeclenk(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(3)
#	ndx[0] = 2
#	ndx[1] = 3
#	ndx[2] = 4
#	G2 = G[ndx,ndx]
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [1, 0, 2, 1]
#	expJ = [0, 1, 1, 2]
#	expV = [32, 23, 43, 34]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_subgraph_simple_scalar_scalar(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = 2
#	G2 = G.subgraph(ndx,ndx)
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [0]
#	expJ = [0]
#	expV = [21]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_subgraph_simple_scalar_null(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = 2
#	G2 = G.subgraph(ndx,ndx)
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = []
#	expJ = []
#	expV = []
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_subgraph_simple_ParVeclen1_scalar(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(1)
#	ndx[0] = 2
#	G2 = G.subgraph(ndx,ndx)
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [0]
#	expJ = [0]
#	expV = [21]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_subgraph_simple_ParVeclen1_null(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(1)
#	ndx[0] = 2
#	G2 = G.subgraph(ndx,ndx)
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = []
#	expJ = []
#	expV = []
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_subgraph_simple_ParVeclenk(self):
#	# ensure that a simple DiGraph constructor creates the number, source/
#        # destination, and value pairs expected.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	ndx = ParVec(3)
#	ndx[0] = 2
#	ndx[1] = 3
#	ndx[2] = 4
#	G2 = G.subgraph(ndx,ndx)
#	[actualI, actualJ, actualV] = G2.toParVec()
#	expI = [1, 0, 2, 1]
#	expJ = [0, 1, 1, 2]
#	expV = [32, 23, 43, 34]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_DiGraph_duplicates(self):
#	# ensure that a DiGraph constructor creates the number, source/
#        # destination, and value pairs expected when 3 input edges have
#        # the same source and destination.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 3, 3, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	expI = [1, 0, 2, 3, 1, 3, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	expJ = [0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	expV = [10, 1, 21, 31, 12, 32, 79, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G = self.initializeGraph(nvert, nedge, origI, origJ, origV)
#	[actualI, actualJ, actualV] = G.toParVec()
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(origI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_add_simple(self):
#	# ensure that DiGraph addition creates the number, source/
#        # destination, and value pairs expected when all edges are 
#        # in both DiGraphs.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, (1.6e+10)+1,
#		18, 88, 19, 69, 79]
#	expV = [21, 3, 43, 63, 25, 65, 27, 47, 87, 29, 69, 31, 33, (3.2e+10)+1,
#		35, 175, 37, 137, 157]
#	G1 = self.initializeGraph(nvert, nedge, origI, origJ, origV1)
#	G2 = self.initializeGraph(nvert, nedge, origI, origJ, origV2)
#        G3 = G1+G2
#	[actualI, actualJ, actualV] = G3.toParVec()
#        self.assertEqual(len(origI), len(actualI))
#        self.assertEqual(len(origJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(origI)):
#		self.assertEqual(origI[ind], actualI[ind])
#		self.assertEqual(origJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_add_union(self):
#	# ensure that DiGraph addition creates the number, source/
#        # destination, and value pairs expected when some edges are not
#        # in both DiGraphs.
#	nvert1 = 9
#	nedge1 = 19
#	origI1 = [1, 0, 2, 4, 1, 3, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV1 = [10, 1, 21, 41, 12, 32, 13, 23, 33, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
#	nvert2 = 9
#	nedge2 = 19
#	origI2 = [7, 3, 6, 8, 5, 7, 4, 5, 6, 5, 7, 7, 2, 7, 2, 7, 0, 2, 5]
#	origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV2 = [70, 31, 61, 81, 52, 72, 43, 53, 63, 54, 74, 75, 26, 1.6e+10,
#		27, 77, 8, 28, 58]
#	G2 = self.initializeGraph(nvert2, nedge2, origI2, origJ2, origV2)
#        G3 = G1 + G2
#	[actualI, actualJ, actualV] = G3.toParVec()
#	expNvert = 9
#	expNedge = 38
#        expI = [1, 7, 0, 2, 3, 4, 6, 8, 1, 3, 5, 7, 1, 2, 3, 4, 5, 6, 1, 3, 5,
#		 7, 1, 7, 1, 2, 7, 8, 1, 2, 7, 8, 0, 1, 2, 5, 6, 7]
#        expJ = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
#		 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
#	expV = [10,70, 1,21,31,41,61,81,12,32,52,72,13,23,33,43,53,63,14,34,54,
#		74,15,75,16,26,1.6e+10,1.6e+10,17,27,77,87,8,18,28,58,68,78]
#	[actualI, actualJ, actualV] = G3.toParVec()
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertEqual(expV[ind], actualV[ind])
#        
#    def test_multiply_simple(self):
#	# ensure that DiGraph addition creates the number, source/
#        # destination, and value pairs expected when all edges are 
#        # in both DiGraphs.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 7.3,
#		17, 87, 18, 68, 78]
#	origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, 8.3,
#		18, 88, 19, 69, 79]
#	expV = [110, 2, 462, 992, 156, 1056, 182, 552, 1892, 210, 1190, 240,
#		272, 60.59, 306, 7656, 342, 4692, 6162]
#	G1 = self.initializeGraph(nvert, nedge, origI, origJ, origV1)
#	G2 = self.initializeGraph(nvert, nedge, origI, origJ, origV2)
#        G3 = G1*G2
#	[actualI, actualJ, actualV] = G3.toParVec()
#        self.assertEqual(len(origI), len(actualI))
#        self.assertEqual(len(origJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(origI)):
#		self.assertEqual(origI[ind], actualI[ind])
#		self.assertEqual(origJ[ind], actualJ[ind])
#		self.assertAlmostEqual(expV[ind], actualV[ind])
#        
#    def test_multiply_intersection(self):
#	# ensure that DiGraph addition creates the number, source/
#        # destination, and value pairs expected when some edges are not
#        # in both DiGraphs.
#	nvert1 = 9
#	nedge1 = 19
#	origI1 = [1, 0, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
#	origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 16, 7.7,
#		17, 87, 8, 68, 78]
#	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
#	nvert2 = 9
#	nedge2 = 19
#	origI2 = [7, 3, 4, 8, 5, 7, 3, 5, 6, 3, 7, 7, 2, 8, 2, 7, 0, 2, 5]
#	origJ2 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV2 = [70, 31, 41, 81, 52, 72, 33, 53, 63, 34, 74, 75, 26, 7.7,
#		27, 77, 8, 28, 58]
#	G2 = self.initializeGraph(nvert2, nedge2, origI2, origJ2, origV2)
#        G3 = G1*G2
#	[actualI, actualJ, actualV] = G3.toParVec()
#	expNvert = 9
#	expNedge = 6
#        expI = [4, 5, 3, 3, 8, 0]
#        expJ = [1, 2, 3, 4, 6, 8]
#	expV = [1681, 2704, 1089, 1156, 59.29, 64]
#        self.assertEqual(len(expI), len(actualI))
#        self.assertEqual(len(expJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(expI)):
#		self.assertEqual(expI[ind], actualI[ind])
#		self.assertEqual(expJ[ind], actualJ[ind])
#		self.assertAlmostEqual(expV[ind], actualV[ind])
#
#    def test_div_simple(self):
#	# ensure that DiGraph addition creates the number, source/
#        # destination, and value pairs expected when all edges are 
#        # in both DiGraphs.
#	nvert = 9
#	nedge = 19
#	origI = [1, 0, 2, 3, 1, 3, 1, 2, 4, 1, 3, 1, 1, 8, 1, 8, 1, 6, 7]
#	origJ = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
#	origV1 = [10, 1, 21, 31, 12, 32, 13, 23, 43, 14, 34, 15, 16, 1.6e+10,
#		17, 87, 18, 68, 78]
#	origV2 = [11, 2, 22, 32, 13, 33, 14, 24, 44, 15, 35, 16, 17, (1.6e+10)+1,
#		18, 88, 19, 69, 79]
#	expV = [0.9090909091, 0.5, 0.9545454545, 0.96875, 0.92307692, 0.96969696, 
#		0.92857142, 0.95833333, 0.97727272, 0.93333333, 0.97142857, 0.93750000, 
#		0.94117647, 1, 0.94444444, 0.98863636, 0.94736842, 0.98550724, 0.98734177]
#	G1 = self.initializeGraph(nvert, nedge, origI, origJ, origV1)
#	G2 = self.initializeGraph(nvert, nedge, origI, origJ, origV2)
#        G3 = G1/G2
#	[actualI, actualJ, actualV] = G3.toParVec()
#        self.assertEqual(len(origI), len(actualI))
#        self.assertEqual(len(origJ), len(actualJ))
#        self.assertEqual(len(expV), len(actualV))
#        for ind in range(len(origI)):
#		self.assertEqual(origI[ind], actualI[ind])
#		self.assertEqual(origJ[ind], actualJ[ind])
#		self.assertAlmostEqual(expV[ind], actualV[ind])
        
class GeneralPurposeTests(HyGraphTests):
    def test_multNot(self):
	nvert1 = 9
	nedge1 = 19
	origI1 = [1, 0, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
	origJ1 = [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
		17, 87, 8, 68, 78]
	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
	nvert2 = 9
	nedge2 = 10
	origI2 = [7, 0, 4, 8, 5, 2, 7, 8, 1, 7]
	origJ2 = [0, 1, 1, 1, 2, 3, 5, 6, 7, 8]
	origV2 = [70, 1, 41, 81, 52, 23, 75, 8.6, 17, 78]
	G2 = self.initializeGraph(nvert2, nedge2, origI2, origJ2, origV2)
        G3 = G1.mulNot(G2)
	[actualI, actualJ, actualV] = G3.toParVec()
	expNvert = 9
	expNedge = 13
	expI = [1, 6, 1, 1, 3, 1, 3, 1, 1, 8, 0, 6]
	expJ = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8]
	expV = [10, 61, 12, 13, 33, 14, 34, 15, 1.6, 87, 8, 68]
        self.assertEqual(len(expI), len(actualI))
        self.assertEqual(len(expJ), len(actualJ))
        self.assertEqual(len(expV), len(actualV))
        for ind in range(len(expI)):
		self.assertEqual(expI[ind], actualI[ind])
		self.assertEqual(expJ[ind], actualJ[ind])
		self.assertAlmostEqual(expV[ind], actualV[ind])

    def test_scale_out(self):
	nvert1 = 9
	nedge1 = 19
	origI1 = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
	origJ1 = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
		17, 87, 8, 68, 78]
	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
	vec1 = SpParVec(nvert1)
	# vec[0] null, scaling a null column in G1
	vec1[1] = 1
	vec1[2] = 2
	vec1[3] = 3
	vec1[4] = 4
	vec1[5] = 5
	# vec[6] null, scaling a non-null column in G1
	vec1[7] = 7
	vec1[8] = 8
        G1.scale(vec1, dir=DiGraph.Out)
	[actualI, actualJ, actualV] = G1.toParVec()
	expI = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
	expJ = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	expV = [10, 1, 164, 61, 12, 260, 13, 46, 99, 14, 102, 15, 1.6, 68.8,
		17, 696, 8, 68, 546]
        self.assertEqual(len(expI), len(actualI))
        self.assertEqual(len(expJ), len(actualJ))
        self.assertEqual(len(expV), len(actualV))
        for ind in range(len(expI)):
		self.assertEqual(expI[ind], actualI[ind])
		self.assertEqual(expJ[ind], actualJ[ind])
		self.assertAlmostEqual(expV[ind], actualV[ind])

    def test_scale_in(self):
	nvert1 = 9
	nedge1 = 19
	origI1 = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
	origJ1 = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	origV1 = [10, 1, 41, 61, 12, 52, 13, 23, 33, 14, 34, 15, 1.6, 8.6,
		17, 87, 8, 68, 78]
	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
	vec1 = SpParVec(nvert1)
	# vec[0] null, scaling a null column in G1
	vec1[1] = 1
	vec1[2] = 2
	vec1[3] = 3
	vec1[4] = 4
	vec1[5] = 5
	# vec[6] null, scaling a non-null column in G1
	vec1[7] = 7
	vec1[8] = 8
        G1.scale(vec1, dir=DiGraph.In)
	[actualI, actualJ, actualV] = G1.toParVec()
	expI = [0, 1, 4, 6, 1, 5, 1, 2, 3, 1, 3, 1, 1, 8, 1, 8, 0, 6, 7]
	expJ = [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8]
	expV = [10, 1, 41, 61, 24, 104, 39, 69, 99, 56, 136, 75, 1.6, 8.6,
		119, 609, 64, 544, 624]
        self.assertEqual(len(expI), len(actualI))
        self.assertEqual(len(expJ), len(actualJ))
        self.assertEqual(len(expV), len(actualV))
        for ind in range(len(expI)):
		self.assertEqual(expI[ind], actualI[ind])
		self.assertEqual(expJ[ind], actualJ[ind])
		self.assertAlmostEqual(expV[ind], actualV[ind])


        
class LinearAlgebraTests(HyGraphTests):
    def test_matMul_1row1col(self):
	nvert1 = 16
	nedge1 = 4
	origI1 = [0, 0, 0, 0]
	origJ1 = [1, 3, 4, 12]
	origV1 = [1, 1, 1, 1]
	G1 = self.initializeGraph(nvert1, nedge1, origI1, origJ1, origV1)
	nvert2 = 16
	nedge2 = 4
	origI2 = [1, 3, 4, 12]
	origJ2 = [0, 0, 0, 0]
	origV2 = [1, 1, 1, 1]
	G2 = self.initializeGraph(nvert2, nedge2, origI2, origJ2, origV2)
	G3 = G1._SpMM(G2)
	self.assertEqual(G1.nvert(), G3.nvert())
	[i3, j3, v3] = G3.toParVec()
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

    def test_matMul_simple(self):
	G = DiGraph.load('testfiles/small_nonsym_fp.mtx')
	GT = G.copy()
	GT._T()
	G2 = G._SpMM(GT)
	self.assertEqual(G.nvert(),9)
	[i2, j2, v2] = G2.toParVec()
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


def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BuiltInMethodTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ConstructorTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PageRankTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NormalizeEdgeWeightsTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(DegreeTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CentralityTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BFSTreeTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IsBFSTreeTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(NeighborsTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(PathsHopTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LoadTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MaxTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(MinTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(GeneralPurposeTests))
    #suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LinearAlgebraTests))
    return suite

if __name__ == '__main__':
    runTests()
