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
try:
	import kdt.pyCombBLAS as pcb
except ImportError:
	raise ImportError,"Failed to import pyCombBLAS. If you just installed KDT, please use a different working directory. Python is loading the kdt module from the current directory (which is unbuilt), NOT from the installation."
Obj1 = pcb.Obj1
Obj2 = pcb.Obj2
import kdt.ObjMethods

import Algorithms

# SEJITS
from Util_SEJITS import *

try:
	from specializer.pcb_predicate import PcbUnaryPredicate
	from specializer.pcb_predicate import PcbBinaryPredicate
	from specializer.pcb_function import PcbUnaryFunction
	from specializer.pcb_function import PcbBinaryFunction
	SEJITS_enable(True)
except ImportError:
	SEJITS_enable(False)




# The imports below are temporary. When their code is finalized
# they'll get merged into Algorithms.py and Mat.py
import eig
import SpectralClustering
