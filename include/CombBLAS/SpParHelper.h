/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


/**
 * Functions that are used by multiple parallel matrix classes, but don't need the "this" pointer
 **/

#ifndef _SP_PAR_HELPER_H_
#define _SP_PAR_HELPER_H_

#include <vector>
#include <array>
#include <mpi.h>
#include "LocArr.h"
#include "CommGrid.h"
#include "MPIType.h"
#include "SpDefs.h"
#include "psort/psort.h"

namespace combblas {

class SpParHelper
{
public:
	template <typename IT>	
	static void ReDistributeToVector(int * & map_scnt, std::vector< std::vector< IT > > & locs_send, 
			std::vector< std::vector< std::string > > & data_send, std::vector<std::array<char, MAXVERTNAME>> & distmapper_array, const MPI_Comm & comm);

	template<typename KEY, typename VAL, typename IT>
	static void GlobalSelect(IT gl_rank, std::pair<KEY,VAL> * & low,  std::pair<KEY,VAL> * & upp, std::pair<KEY,VAL> * array, IT length, const MPI_Comm & comm);

	template<typename KEY, typename VAL, typename IT>
	static void BipartiteSwap(std::pair<KEY,VAL> * low, std::pair<KEY,VAL> * array, IT length, int nfirsthalf, int color, const MPI_Comm & comm);

	// Necessary because psort creates three 2D vectors of size p-by-p
	// One of those vector with 8 byte data uses 8*(4096)^2 = 128 MB space 
	// Per processor extra storage becomes:
	//	24 MB with 1K processors
	//	96 MB with 2K processors
	//	384 MB with 4K processors
	// 	1.5 GB with 8K processors
	template<typename KEY, typename VAL, typename IT>
	static void MemoryEfficientPSort(std::pair<KEY,VAL> * array, IT length, IT * dist, const MPI_Comm & comm);

    	template<typename KEY, typename VAL, typename IT>
    	static std::vector<std::pair<KEY,VAL>> KeyValuePSort(std::pair<KEY,VAL> * array, IT length, IT * dist, const MPI_Comm & comm);
	
	template<typename KEY, typename VAL, typename IT>
	static void DebugPrintKeys(std::pair<KEY,VAL> * array, IT length, IT * dist, MPI_Comm & World);

	template<typename IT, typename NT, typename DER>
	static void FetchMatrix(SpMat<IT,NT,DER> & MRecv, const std::vector<IT> & essentials, std::vector<MPI_Win> & arrwin, int ownind);

	template<typename IT, typename NT, typename DER>	
	static void BCastMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, const std::vector<IT> & essentials, int root);

	template<typename IT, typename NT, typename DER>	
	static void IBCastMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, const std::vector<IT> & essentials, int root, std::vector<MPI_Request> & indarrayReq , std::vector<MPI_Request> & numarrayReq);
    
    	template<typename IT, typename NT, typename DER>
    	static void GatherMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, int root);

	template<typename IT, typename NT, typename DER>
	static void SetWindows(MPI_Comm & comm1d, const SpMat< IT,NT,DER > & Matrix, std::vector<MPI_Win> & arrwin);

	template <typename IT, typename NT, typename DER>
	static void GetSetSizes(const SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI_Comm & comm1d);

	template <typename IT, typename DER>
	static void AccessNFetch(DER * & Matrix, int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group, IT ** sizes);

	template <typename IT, typename DER>
	static void LockNFetch(DER * & Matrix, int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group, IT ** sizes);

	static void StartAccessEpoch(int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group);
	static void PostExposureEpoch(int self, std::vector<MPI_Win> & arrwin, MPI_Group & group);
	static void LockWindows(int ownind, std::vector<MPI_Win> & arrwin);
	static void UnlockWindows(int ownind, std::vector<MPI_Win> & arrwin);

	static void Print(const std::string & s);
    	static void Print(const std::string & s, MPI_Comm & world);
	static void PrintFile(const std::string & s, const std::string & filename);
    	static void PrintFile(const std::string & s, const std::string & filename, MPI_Comm & world);
    	static void check_newline(int *bytes_read, int bytes_requested, char *buf);
   	static bool FetchBatch(MPI_File & infile, MPI_Offset & curpos, MPI_Offset end_fpos, bool firstcall, std::vector<std::string> & lines, int myrank);
    
	static void WaitNFree(std::vector<MPI_Win> & arrwin);
	static void FreeWindows(std::vector<MPI_Win> & arrwin);
};

}

#include "SpParHelper.cpp"

#endif
