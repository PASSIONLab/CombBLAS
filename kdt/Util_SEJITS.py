
## SEJITS helpers.
# SEJITS itself is in the specializer subdirectory.


# placeholder parent class for when SEJITS is disabled
class _SEJITS_diabled_callback_parent:
	pass

# the SEJITS callback parent classes
KDTUnaryPredicate = _SEJITS_diabled_callback_parent
KDTBinaryPredicate = _SEJITS_diabled_callback_parent
KDTUnaryFunction = _SEJITS_diabled_callback_parent
KDTBinaryFunction = _SEJITS_diabled_callback_parent

# load the real SEJITS callbacks
try:
	from specializer.pcb_predicate import PcbUnaryPredicate
	from specializer.pcb_predicate import PcbBinaryPredicate
	from specializer.pcb_function import PcbUnaryFunction
	from specializer.pcb_function import PcbBinaryFunction
except ImportError:
	pass

def SEJITS_enable(en):
"""
Enables/disables SEJITS by manipulating the callback parent classes.
"""
	global KDTUnaryPredicate
	global KDTBinaryPredicate
	global KDTUnaryFunction
	global KDTBinaryFunction

	if en:
		try:
			KDTUnaryPredicate = PcbUnaryPredicate
			KDTBinaryPredicate = PcbBinaryPredicate
			KDTUnaryFunction = PcbUnaryFunction
			KDTBinaryFunction = PcbBinaryFunction
		except NameError:
			raise RuntimeError, "SEJITS not available."
	else:
		KDTUnaryPredicate = _SEJITS_diabled_callback_parent
		KDTBinaryPredicate = _SEJITS_diabled_callback_parent
		KDTUnaryFunction = _SEJITS_diabled_callback_parent
		KDTBinaryFunction = _SEJITS_diabled_callback_parent

def SEJITS_enabled():
"""
Tests whether or not the SEJITS callback parent classes are set or not.
"""
	return KDTUnaryPredicate is not _SEJITS_diabled_callback_parent