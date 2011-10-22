import pyCombBLAS as pcb

class info:
	@staticmethod
	def eps():
		"""
		Return IEEE floating point machine epsilon.
		The problem with this operation is that Python only provides a portable way to get this
		value in v2.6 and NumPy isn't always available. This function attempts to use whatever
		knows this value or returns a reasonable default otherwise.
		"""
		# try Python v2.6+ float_info
		try:
			from sys import float_info as fi
			return fi.epsilon
		except ImportError:
			pass
			
		# try Numpy
		try:
			import numpy as np
			return float(np.finfo(np.float).eps)
		except ImportError:
			pass
		except AttributeError:
			pass
			
		# return a reasonable value
		return 2.220446049250313e-16;

	def minInt():
		return -(2**62)

def sr(addFn, mulFn):
	return pcb.SemiringObj(addFn, mulFn)

####
#
# KDT-level wrapping of fast functions which only work on floating point values.
# If a unary or binary operation is composed of these functions then it will
# run at C++ speed when available and default back to a Python implementation
# when not available. When SEJITS integration is complete these will all be
# deprecated in favor of direct Python code.
#
####

# built-in semirings
sr_select2nd = pcb.SecondMaxSemiring()
		
# built-in operations that only work on floating point scalars
op_add = pcb.plus()
op_sub = pcb.minus()
op_mul = pcb.multiplies()
op_div = pcb.divides()
op_mod = pcb.modulus()
op_fmod = pcb.fmod()
op_pow = pcb.pow()
op_max  = pcb.max()
op_min = pcb.min()
op_bitAnd = pcb.bitwise_and()
op_bitOr = pcb.bitwise_or()
op_bitXor = pcb.bitwise_xor()
op_and = pcb.logical_and()
op_or = pcb.logical_or()
op_xor = pcb.logical_xor()
op_eq = pcb.equal_to()
op_ne = pcb.not_equal_to()
op_gt = pcb.greater()
op_lt = pcb.less()
op_ge = pcb.greater_equal()
op_le = pcb.less_equal()

op_id = pcb.identity()
# safemultinv()
op_abs = pcb.abs()
op_negate = pcb.negate()
op_bitNot = pcb.bitwise_not()
op_not = pcb.logical_not()
#totality()

class _complex_op:
	def __init__(self, op, name):
		self.pcb_op = op
		self.name = name
	

def op_set(val):
	return pcb.set(val)
	#ret = _complex_op(pcb.set(val), "set")
	#ret.val = val
	#return ret

def op_IfThenElse(predicate, runTrue, runFalse):
	return pcb.ifthenelse(predicate, runTrue, runFalse)
	#ret = _complex_op(pcb.ifthenelse(predicate, runTrue, runFalse), "ifthenelse")
	#ret.predicate = predicate
	#ret.runTrue = runTrue
	#ret.runFalse = runFalse
	#return ret

def op_bind1st(op, val):
	return pcb.bind1st(op, val)

def op_bind2nd(op, val):
	return pcb.bind2nd(op, val)

def op_compose1(f, g):
	return pcb.compose1(f, g)

def op_compose2(f, g1, g2):
	return pcb.compose2(f, g1, g2)

def op_not1(op):
	return pcb.not1(op)

def op_not2(op):
	return pcb.not1(op)

def _op_builtin_pyfunc(op):
	# binary functions
	if op == op_add:
		return lambda x, y: (x + y)
	if op == op_sub:
		return lambda x, y: (x - y)
	if op == op_mul:
		return lambda x, y: (x * y)
	if op == op_div:
		return lambda x, y: (x / y)
	if op == op_mod:
		return lambda x, y: (x % y)
	if op == op_fmod:
		raise NotImplementedError, 'fmod Python expression not implemented' 
		#return lambda x, y: (x % y)
	if op == op_pow:
		raise NotImplementedError, 'pow Python expression not implemented' 
		#return lambda x, y: (x y)
	if op == op_max:
		return lambda x, y: max(x, y)
	if op == op_min:
		return lambda x, y: min(x, y)
	if op == op_bitAnd:
		raise NotImplementedError, 'bitwise AND Python expression not implemented' 
		#return lambda x, y: (x y)
	if op == op_bitOr:
		raise NotImplementedError, 'bitwise OR Python expression not implemented' 
		#return lambda x, y: (x y)
	if op == op_bitXor:
		raise NotImplementedError, 'bitwise XOR Python expression not implemented' 
		#return lambda x, y: (x y)
	if op == op_and:
		return lambda x, y: (x and y)
	if op == op_or:
		return lambda x, y: (x or y)
	if op == op_xor:
		raise NotImplementedError, 'logical XOR Python expression not implemented' 
		#return lambda x, y: (x xor y)
	if op == op_eq:
		return lambda x, y: (x == y)
	if op == op_ne:
		return lambda x, y: (x != y)
	if op == op_gt:
		return lambda x, y: (x > y)
	if op == op_lt:
		return lambda x, y: (x < y)
	if op == op_ge:
		return lambda x, y: (x >= y)
	if op == op_le:
		return lambda x, y: (x <= y)
	
	# unary functions
	if op == op_id:
		return lambda x: (x)
	if op == op_abs:
		return lambda x: abs(x)
	if op == op_negate:
		return lambda x: (-x)
	if op == op_bitNot:
		return lambda x: (~x)
	if op == op_not:
		return lambda x: (not x)
	
	raise NotImplementedError, 'Unable to convert functor to Python expression.'
	
def _op_make_unary(op):
	if op is None:
		return None
	if isinstance(op, (pcb.UnaryFunction, pcb.UnaryFunctionObj)):
		return op
	return pcb.unaryObj(op)

def _op_make_binary(op):
	if op is None:
		return None
	if isinstance(op, (pcb.BinaryFunction, pcb.BinaryFunctionObj)):
		return op
	return pcb.binaryObj(op)

def _op_make_unary_pred(op):
	if op is None:
		return None
	if isinstance(op, (pcb.UnaryFunction, pcb.UnaryPredicateObj)):
		return op
	return pcb.unaryObjPred(op)

def _op_make_binary_pred(op):
	if op is None:
		return None
	if isinstance(op, (pcb.BinaryFunction, pcb.BinaryPredicateObj)):
		return op
	return pcb.binaryObjPred(op)


def _makePythonOp(op):
	if isinstance(op, (pcb.UnaryFunction, pcb.BinaryFunction)):
		return _op_builtin_pyfunc(op)
	elif isinstance(op, (pcb.UnaryFunctionObj, pcb.BinaryFunctionObj)):
		raise NotImplementedError, 'Unable to convert OBJ functor back to Python expression.'
	else:
		return op