#include "pyCombBLAS.h"

////////////////// OPERATORS

pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude)
{
	pySpParVec* ret = new pySpParVec();
	//ret->v = ::EWiseMult(a.v, b.v, exclude);
	cout << "EWiseMult(sparse, sparse) not implemented!" << endl;
	return ret;
}

pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero)
{
	pySpParVec* ret = new pySpParVec();
	FullyDistSpVec<int64_t, int64_t> result = EWiseMult(a.v, b.v, exclude, (int64_t)zero);
	ret->v.stealFrom(result);
	return ret;
}

void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero)
{
	a.v = EWiseMult(a.v, b.v, exclude, (int64_t)zero);
}



////////////////////////// INITALIZATION/FINALIZE

void init_pyCombBLAS_MPI()
{
	//cout << "calling MPI::Init" << endl;
	MPI::Init();
	//cblas_alltoalltime = 0;
	//cblas_allgathertime = 0;	

	/*
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	MPI::COMM_WORLD.Barrier();
	
	int sum = 0;
	int one = 1;
	MPI_Reduce(&one, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 

	cout << "I am proc " << myrank << " out of " << nprocs << ". Hear me roar!" << endl;
	if (myrank == 0) {
		cout << "We all reduced our ones to get " << sum;
		if (sum == nprocs)
			cout << ". Success! MPI works." << endl;
		else
			cout << ". SHOULD GET #PROCS! MPI is broken!" << endl;
	}
	*/
}

void finalize()
{
	//cout << "calling MPI::Finalize" << endl;
	MPI::Finalize();
}

bool root()
{
	return MPI::COMM_WORLD.Get_rank() == 0;
}
