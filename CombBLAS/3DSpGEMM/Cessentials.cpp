#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include "../CombBLAS.h"
#include "../SpParHelper.h"
#include "../LocArr.h"
#include "Glue.h"

#ifdef BUPC
extern "C" void Split_GetEssensials(void * full, void ** part1, void ** part2, SpDCCol_Essentials * sess1, SpDCCol_Essentials * sess2);
#endif


void Split_GetEssensials(void * full, void ** part1, void ** part2, SpDCCol_Essentials * sess1, SpDCCol_Essentials * sess2)
{
	SpParHelper::Print("About to split\n");
	SpDCCols<int32_t, double> * fullmat = (SpDCCols<int32_t, double> *) full;
	SpDCCols<int32_t, double> * part1mat = new SpDCCols<int32_t, double>();
	SpDCCols<int32_t, double> * part2mat = new SpDCCols<int32_t, double>();
	fullmat->Split( *part1mat, *part2mat);
	
	vector<int32_t> essA = part1mat->GetEssentials();
	sess1->nnz = essA[0];
	sess1->m = essA[1];
	sess1->n = essA[2];
	sess1->nzc = essA[3];
	
	vector<int32_t> essB = part2mat->GetEssentials();
	sess2->nnz = essB[0];
	sess2->m = essB[1];
	sess2->n = essB[2];
	sess2->nzc = essB[3];

	*part1 = (void*) part1mat;
	*part2 = (void*) part2mat;
	SpParHelper::Print("Essentials filled, returning...\n");
}

void MakeObject_FromArrays(void * part, SpDCCol_Arrays * sarrs, SpDCCol_Essentials * sess)
{
	SpDCCols<int32_t, double> * mat = new SpDCCols<int32_t, double>();    // create the object first

	// SpDCCols<int32_t, double>::CreateImpl(IT * _cp, IT * _jc, IT * _ir, NT * _numx, IT _nz, IT _nzc, IT _m, IT _n)
	mat->CreateImpl( (int32_t*) sarrs->cp, (int32_t*) sarrs->jc, (int32_t*) sarrs->ir, (double*) sarrs->num, sess->nnz, sess->nzc, sess->m, sess->n);
}

	/*
	MRecv.Create(essentials);		// allocate memory for arrays
 
	Arr<IT,NT> arrinfo = MRecv.GetArrays();
	assert( (arrwin.size() == arrinfo.totalsize()));

	// C-binding for MPI::Get
	//	int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        //    		int target_count, MPI_Datatype target_datatype, MPI_Win win)

	IT essk = 0;
	for(int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		arrwin[essk++].Get(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), ownind, 0, arrinfo.indarrs[i].count, MPIType<IT>());
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		arrwin[essk++].Get(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), ownind, 0, arrinfo.numarrs[i].count, MPIType<NT>());
	}
	*/


