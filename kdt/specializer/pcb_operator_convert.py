import asp.codegen.ast_tools as ast_tools
import asp.codegen.cpp_ast as cpp_ast

# this should get pulled into Asp
class CppAttribute(cpp_ast.Generable):
    def __init__(self, value, attr):
        self._fields = ["value", "attr"]
        self.value = value
        self.attr = attr
    
    def generate(self, with_semicolon=False):
        yield "%s.%s" % (self.value, self.attr)

class CppClass(cpp_ast.Generable):
    def __init__(self, classname, body, parentclass=None):
        self.fields = ["classname", "parentclass"]
        self.classname = classname
        self.parentclass = parentclass
        self.body = body

    def generate(self, with_semicolon=True):
        decl = "class %s " % self.classname
        if self.parentclass:
            decl += ": public %s" % self.parentclass
        yield decl + "{\n" 
        # everything is public, for now
        yield "public:\n"
        for line in self.body.generate():
            yield line
        yield "};\n"

class PcbOperatorConvert(ast_tools.NodeTransformer):
    """
    This class is used to convert from a semantic model (expressed in terms of nodes in pcb_operator_sm)
    to a C++ AST that can then be further transformed or directly generated.
    """

    def convert(self, sm, name="myfunc", types=["bool", "double", "double"]):
        """
        Entry point.  Call this on the semantic model.  Returns the C++ AST.
        """
        # the types array tells us what the return and input types are
        self.types = types
        self.name = name
        return self.visit(sm)


    def visit_UnaryPredicate(self, node):
        #FIXME: this should actually have const in the signature for the input, but doesn't look like CodePy
        # supports this properly
        #
        # the proper signature should be bool foo (const T& x) const

        return cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value(self.types[0], self.name),
                                                                [cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[1], self.visit(node.input)))]),
                                             cpp_ast.Block(contents=[self.visit(node.body)]))
#                                             cpp_ast.Block(contents=[cpp_ast.Print('<<"IN_CALL"', True),self.visit(node.body)]))
              

    def visit_BinaryPredicate(self, node):
        #FIXME: same problem as UnaryPredicate
        return cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value(self.types[0], self.name),
                                                                [cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[1], self.visit(node.inputs[0]))),
                                                                 cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[2], self.visit(node.inputs[1])))]),
                                             cpp_ast.Block(contents=[self.visit(node.body)]))

    def visit_UnaryFunction(self, node):
        # FIXME: have to worry about type specialization on return & passed-in types
        return cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value(self.types[0], self.name),
                                                                [cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[1], self.visit(node.input)))]),
                                    cpp_ast.Block(contents=[self.visit(node.body)]))


    def visit_BinaryFunction(self, node):
        #FIXME: this should actually have const in the signature for the input, but doesn't look like CodePy
        # supports this properly
        #

        return cpp_ast.FunctionBody(cpp_ast.FunctionDeclaration(cpp_ast.Value(self.types[0], self.name),
                                                                [cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[1], self.visit(node.inputs[0]))),
                                                                 cpp_ast.Reference(cpp_ast.Value("const %s" % self.types[2], self.visit(node.inputs[1])))]),
                                             cpp_ast.Block(contents=[self.visit(node.body)]))
              

                                                                                                     


    def visit_BoolReturn(self, node):
        return cpp_ast.ReturnStatement(self.visit(node.value))

    def visit_FunctionReturn(self, node):
        #FIXME: is this correct?
        #return cpp_ast.ReturnStatement(cpp_ast.TypeCast(cpp_ast.Value("T", ""), cpp_ast.Line("*(new "+str(self.visit(node.ret_type))+")")))
        return cpp_ast.ReturnStatement(self.visit(node.value))

    def visit_BoolConstant(self, node):
        if node.value:
            return "true"
        else:
            return "false"

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return cpp_ast.CNumber(node.value)
        else:
            return cpp_ast.CName(node.value)
            

    def visit_IfExp(self, node):
        return cpp_ast.IfConv(self.visit(node.test), self.visit(node.body), else_=self.visit(node.orelse))

    def visit_Compare(self, node):
        import ast
        comparator_map = {ast.Eq:"==", ast.NotEq:"!=", ast.Lt:"<", ast.LtE:"<=", ast.Gt:">", ast.GtE:">="}
        return cpp_ast.Compare(self.visit(node.left), comparator_map[node.op.__class__], self.visit(node.right))
        
    def visit_BinaryOp(self, node):
        import ast
        binop_map = {ast.Add:"+", ast.Sub:"-", ast.And:"&&"}
        return cpp_ast.BinOp(self.visit(node.left), binop_map[node.op.__class__], self.visit(node.right))

    def visit_BoolOp(self, node):
        import ast
        boolop_map = {ast.And:"&&", ast.Or: "||"}
        #FIXME: add support for the NOT operation
        return cpp_ast.BinOp(self.visit(node.operands[0]), boolop_map[node.op.__class__], self.visit(node.operands[1]))

    def visit_Attribute(self, node):
        return CppAttribute(self.visit(node.value), self.visit(node.attr))

    def visit_Identifier(self, node):
        return cpp_ast.CName(node.name)
