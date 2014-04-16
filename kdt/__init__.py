# package marker

from Util import *
from Util import master, version, revision, _nproc, _rank
from Util import PDO_enable

from DiGraph import DiGraph
from HyGraph import HyGraph
from Vec import Vec
from Mat import Mat
#from SpVec import SpVec, info
#from DeVec import DeVec
from feedback import sendFeedback
from UFget import UFget, UFdownload

# import pyCombBLAS
try:
	import kdt.pyCombBLAS as pcb
except ImportError:
	raise ImportError,"Failed to import pyCombBLAS. If you just installed KDT, please use a different working directory. Python is loading the kdt module from the current directory (which is unbuilt), NOT from the installation."
Obj1 = pcb.Obj1
Obj2 = pcb.Obj2
import kdt.ObjMethods

import Algorithms

# SEJITS
#from Util_SEJITS import *

#try:
#	from specializer.pcb_predicate import PcbUnaryPredicate
#	from specializer.pcb_predicate import PcbBinaryPredicate
#	from specializer.pcb_function import PcbUnaryFunction
#	from specializer.pcb_function import PcbBinaryFunction
#	SEJITS_enable(True)
#except ImportError:
#	SEJITS_enable(False)




# The imports below are temporary. When their code is finalized
# they'll get merged into Algorithms.py and Mat.py
import eig
import SpectralClustering

#=======================================================
# SEJITS ===============================================

## SEJITS helpers.
# SEJITS itself is in the specializer subdirectory.


# placeholder parent class for when SEJITS is disabled
class _SEJITS_diabled_callback_parent(object):
	pass

# the SEJITS callback parent classes
KDTUnaryPredicate = _SEJITS_diabled_callback_parent  # for KDT v0.3 compatibility
KDTBinaryPredicate = _SEJITS_diabled_callback_parent  # for KDT v0.3 compatibility
KDTUnaryFunction = _SEJITS_diabled_callback_parent  # for KDT v0.3 compatibility
KDTBinaryFunction = _SEJITS_diabled_callback_parent  # for KDT v0.3 compatibility
Callback = _SEJITS_diabled_callback_parent

# load the real SEJITS callbacks
try:
	#from specializer.pcb_predicate import PcbUnaryPredicate
	#from specializer.pcb_predicate import PcbBinaryPredicate
	#from specializer.pcb_function import PcbUnaryFunction
	#from specializer.pcb_function import PcbBinaryFunction
	from specializer.pcb_callback import PcbCallback
except ImportError:
	pass

#from specializer.pcb_callback import PcbCallback

def SEJITS_enable(en):
	"""
	Enables/disables SEJITS by manipulating the callback parent classes.
	"""
	global KDTUnaryPredicate
	global KDTBinaryPredicate
	global KDTUnaryFunction
	global KDTBinaryFunction
	global Callback
	
	if en:
		#try:
		#global PcbUnaryPredicate
		#global PcbBinaryPredicate
		#global PcbUnaryFunction
		#global PcbBinaryFunction
		global PcbCallback

		#KDTUnaryPredicate = PcbUnaryPredicate
		#KDTBinaryPredicate = PcbBinaryPredicate
		#KDTUnaryFunction = PcbUnaryFunction
		#KDTBinaryFunction = PcbBinaryFunction
		KDTUnaryPredicate = PcbCallback
		KDTBinaryPredicate = PcbCallback
		KDTUnaryFunction = PcbCallback
		KDTBinaryFunction = PcbCallback
		Callback = PcbCallback

		#except NameError:
		#	raise RuntimeError, "SEJITS not available."
	else:
		KDTUnaryPredicate = _SEJITS_diabled_callback_parent
		KDTBinaryPredicate = _SEJITS_diabled_callback_parent
		KDTUnaryFunction = _SEJITS_diabled_callback_parent
		KDTBinaryFunction = _SEJITS_diabled_callback_parent
		Callback = _SEJITS_diabled_callback_parent

try:
	SEJITS_enable(True)
except NameError:
	pass

def SEJITS_enabled():
	"""
	Tests whether or not the SEJITS callback parent classes are set or not.
	"""
	return KDTUnaryPredicate is not _SEJITS_diabled_callback_parent
	

