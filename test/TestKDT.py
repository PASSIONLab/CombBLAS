import unittest
import TestDiGraph
import TestHyGraph
import TestParVec
import TestSpParVec

def runTests(verbosity = 1):
    testSuite = suite()
    unittest.TextTestRunner(verbosity=verbosity).run(testSuite)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(TestDiGraph.suite())
    suite.addTests(TestHyGraph.suite())
    suite.addTests(TestParVec.suite())
    suite.addTests(TestSpParVec.suite())
    return suite

if __name__ == '__main__':
    runTests()
