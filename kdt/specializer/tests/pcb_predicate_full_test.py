from pcb_predicate import *

import unittest

class FullTwitterTest(unittest.TestCase):
    def test_without_instancevar(self):
        class TwitterFilter(PcbUnaryPredicate):
            def __call__(self, e):
                if (e.count > 0  and e.latest < 1249084800):
                    return True
                else:
                    return False

        # if you don't import kdt, SWIG's type information hasn't loaded and you'll get a segfault
        import kdt

        # this should not throw an exception
        pred = TwitterFilter().get_predicate(types=["bool", "Obj2"])

    def test_with_instancevar(self):
        class TwitterFilter(PcbUnaryPredicate):
            def __init__(self, latest):
                self.latest = latest
                super(TwitterFilter, self).__init__()
            def __call__(self, e):
                if (e.count > 0 and e.latest < self.latest):
                    return True
                else:
                    return False

        # if you don't import kdt, SWIG's type information hasn't loaded and you'll get a segfault
        import kdt

        # this should not throw an exception
        pred = TwitterFilter(10000).get_predicate()

class FullBinaryPredicateTest(unittest.TestCase):
    def test_basic(self):
        class MyFilter(PcbBinaryPredicate):
            def __call__(self, x, y):
                return True

        import kdt

        # this should not throw an exception
        pred = MyFilter().get_predicate()



if __name__ == '__main__':
    unittest.main()
