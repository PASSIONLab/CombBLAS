from pcb_predicate_frontend import *
from pcb_predicate import *

import unittest

def get_ast(func):
    import ast, inspect
    return ast.parse(inspect.getsource(func).lstrip())

class TinyFilterTest(unittest.TestCase):
    def test_tiny(self):
        class TinyFilter(PcbUnaryPredicate):
            def __call__(self, v):
                return True
        model = PcbUnaryPredicateFrontEnd().parse(get_ast(TinyFilter.__call__))
        model_canonical = UnaryPredicate(Identifier("v"),
                                         BoolReturn(BoolConstant(True)))

        assert_contains_node_type(model, UnaryPredicate)
        assert_contains_node_type(model, Identifier)
        assert_contains_node_type(model, BoolConstant)
        self.assertEqual(model.input.name, "v")

    def test_tiny_with_if(self):
        class TinyFilterWithIf(PcbUnaryPredicate):
            def __call__(self, v):
                if True:
                    return True
                else:
                    return False
        
        model = PcbUnaryPredicateFrontEnd().parse(get_ast(TinyFilterWithIf.__call__))
        model_canonical = UnaryPredicate(Identifier("v"),
                                         IfExp(BoolConstant(True),
                                               BoolReturn(BoolConstant(True)),
                                               BoolReturn(BoolConstant(False))))
                                                          
        assert_contains_node_type(model, UnaryPredicate)
        assert_contains_node_type(model, Identifier)
        assert_contains_node_type(model, BoolConstant)
        assert_contains_node_type(model, IfExp)


class TwitterExampleTests(unittest.TestCase):
    def test_without_instancevar(self):
        class TwitterFilter(PcbUnaryPredicate):
            def __call__(self, e):
                if (e.isRetweet and e.latest < 100):
                    return True
                else:
                    return False

        model = PcbUnaryPredicateFrontEnd().parse(get_ast(TwitterFilter.__call__))

    def test_with_instancevar(self):
        class TwitterFilter(PcbUnaryPredicate):
            def __init__(self, latest):
                self.latest = latest
                super(TwitterFilter, self).__init__()
            def __call__(self, e):
                if (e.isRetweet and e.latest < self.latest):
                    return True
                else:
                    return False
        
        i = TwitterFilter(1000)
        print vars(i)
        model = PcbUnaryPredicateFrontEnd().parse(get_ast(TwitterFilter.__call__), env=vars(i))
        assert_contains_node_type(model, Constant)

def assert_contains_node_type(root_node, node_type):
    assert ContainsNodeTypeVisitor().contains(root_node, node_type), "Expected tree to contain node of type %s" % node_type

class ContainsNodeTypeVisitor(ast.NodeVisitor):
    def contains(self, root_node, node_type):
        self.node_type = node_type
        self.result = False
        self.visit(root_node)
        return self.result

    def visit(self, node):
        if isinstance(node, self.node_type):
            self.result = True
        self.generic_visit(node)

if __name__ == '__main__':
    unittest.main()