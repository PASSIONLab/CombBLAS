import unittest

# This block allows the tests to be run from the command line from
# either this directory or the directory above. This should be
# temporary until more about the packaging is understood.
import sys
sys.path.append('.')
sys.path.append('..')

from kdt.DiGraph import DiGraph, ParVec
from kdt import pyCombBLAS as pcb

class PageRankTests(unittest.TestCase):

    def test_connected(self):
        G = DiGraph.fullyConnected(10)
        pr = G.pageRank()

        for prv in pr:
            self.assertAlmostEqual(0.1, prv, 7)

    def test_simple(self):
        # This test is drawn from the PageRank example at
        # http://en.wikipedia.org/wiki/File:PageRanks-Example.svg.
        nvert = 11
        nedges = 17
        i = [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4,  4, 5] 
        j = [3, 2, 3, 4, 5, 6, 7, 8, 1, 4, 5, 6, 7, 8, 9, 10, 4]
        self.assertEqual(len(i), nedges)
        self.assertEqual(len(j), nedges)
        
        iInd = ParVec(nedges)
        jInd = ParVec(nedges)
        vInd = ParVec(nedges, 1)
        for ind in range(nedges):
            iInd[ind] = i[ind]
            jInd[ind] = j[ind]

        spm = pcb.pySpParMat(nvert, nvert, iInd.dpv, jInd.dpv, vInd.dpv)
        G = DiGraph()
        G.spm = spm
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
        nedges = 21
        i = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4,  4, 5, 7, 10] 
        j = [3, 1, 2, 3, 4, 5, 6, 7, 8, 1, 4, 4, 5, 6, 7, 8, 9, 10, 4, 7, 10]
        self.assertEqual(len(i), nedges)
        self.assertEqual(len(j), nedges)
        
        iInd = ParVec(nedges)
        jInd = ParVec(nedges)
        vInd = ParVec(nedges, 1)
        for ind in range(nedges):
            iInd[ind] = i[ind]
            jInd[ind] = j[ind]

        spm = pcb.pySpParMat(nvert, nvert, iInd.dpv, jInd.dpv, vInd.dpv)
        G = DiGraph()
        G.spm = spm
        pr = G.pageRank(0.0001)

        expected = [0.032814, 0.38440, 0.34291, 0.03909, 0.08089, 0.03909, \
                    0.01617, 0.01617, 0.01617, 0.01617, 0.01617]
        for ind in range(nvert):
            self.assertAlmostEqual(pr[ind], expected[ind], 4)

def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(PageRankTests)
    return suite

if __name__ == '__main__':
    runTests()
