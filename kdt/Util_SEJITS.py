
## SEJITS helpers.
# SEJITS itself is in the specializer subdirectory.


# placeholders for when SEJITS is disabled
class _SEJITS_diabled_callback_parent:
	pass

KDTUnaryPredicate = _SEJITS_diabled_callback_parent
KDTBinaryPredicate = _SEJITS_diabled_callback_parent
KDTUnaryFunction = _SEJITS_diabled_callback_parent
KDTBinaryFunction = _SEJITS_diabled_callback_parent

try:
	from specializer.pcb_predicate import PcbUnaryPredicate
	from specializer.pcb_predicate import PcbBinaryPredicate
	from specializer.pcb_function import PcbUnaryFunction
	from specializer.pcb_function import PcbBinaryFunction
	print "here"
except ImportError:
	pass
	
def SEJITS_enable(en):
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
	return KDTUnaryPredicate is not _SEJITS_diabled_callback_parent