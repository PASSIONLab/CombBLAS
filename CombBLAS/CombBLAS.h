#ifndef COMBBLAS_H
#define COMBBLAS_H

#if defined(COMBBLAS_BOOST)
	#include <boost/tr1/memory.hpp>
	#include <boost/tr1/unordered_map.hpp>
	#include <boost/tr1/tuple.hpp>
#elif defined(COMBBLAS_TR1)
	#include <tr1/memory>
	#include <tr1/unordered_map>
	#include <tr1/tuple>
#else // C++11
	#include <memory>
	#include <unordered_map>
	#include <tuple>
#endif

#include "SpTuples.h"
#include "SpDCCols.h"
#include "SpParMat.h"
#include "FullyDistVec.h"
#include "FullyDistSpVec.h"
#include "VecIterator.h"
#include "ParFriends.h"
#include "DistEdgeList.h"
#include "Semirings.h"
#include "Operations.h"

#endif
