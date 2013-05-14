import asp.tree_grammar
import ast
import types


#FIXME: add boolop node

# grammar for functions that return either doubleint or one of the
# user-defined Obj types
asp.tree_grammar.parse('''

UnaryFunction(input=Identifier, body=Expr)

BinaryFunction(inputs=Identifier*, body=Expr)
    check assert len(self.inputs)==2

Expr =  Constant
          | Identifier 
          | BinaryOp
          | BoolConstant
          | IfExp
          | Attribute
          | FunctionReturn
          | Compare
          | BoolOp


Identifier(name=types.StringType)


Compare(left=Expr, op=(ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE), right=Expr)

BoolOp(op=(ast.And | ast.Or | ast.Not), operands=Expr*)

Constant(value = types.IntType | types.FloatType)

BinaryOp(left=Expr, op=(ast.Add | ast.Sub | ast.Mul | ast.Div), right=Expr)

BoolConstant(value = types.BooleanType)


IfExp(test=(Compare|Attribute|Identifier|BoolConstant|BoolOp), body=Expr, orelse=Expr)



# this if for a.b
Attribute(value=Identifier, attr=Identifier)

FunctionReturn(value = Expr)
''', globals())