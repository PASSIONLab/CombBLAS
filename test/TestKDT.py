import unittest
import TestDiGraph
#import TestHyGraph
import TestMat
import TestVec
import TestSpVec

def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(TestDiGraph.suite())
#    suite.addTests(TestHyGraph.suite())
    suite.addTests(TestMat.suite())
    suite.addTests(TestVec.suite())
    suite.addTests(TestSpVec.suite())
    return suite

if __name__ == '__main__':
    runTests()
