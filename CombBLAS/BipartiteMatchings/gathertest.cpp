#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>




int main(int argc, char* argv[])
{
	
    // ------------ initialize MPI ---------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
	{
        int totalen = (int64_t) atoi(argv[1]);
        int locallen = totalen/nprocs;
        totalen = locallen * nprocs;
        
        vector<int64_t> ind1 (locallen, -1);
        vector<int64_t> ind2 (locallen, -1);
        vector<double> val (locallen, 0.0);
        vector<int> disp(nprocs);
        vector<int> recvcnt(nprocs);
        for(int i=0; i<nprocs; i++)
        {
            disp[i] = locallen*i;
            recvcnt[i] = locallen;
        }
        
        //double t1 = MPI_Wtime();
        
        vector<int64_t> recv1 ;
        vector<int64_t> recv2 ;
        vector<double> recv3 ;

        if(myrank == 0)
        {
            recv1.resize(totalen);
            recv2.resize(totalen);
            recv3.resize(totalen);
        }

 double t1 = MPI_Wtime();
        MPI_Gatherv(ind1.data(), locallen, MPIType<int64_t>(), recv1.data(), recvcnt.data(), disp.data(), MPIType<int64_t>(), 0, MPI_COMM_WORLD);
        MPI_Gatherv(ind2.data(), locallen, MPIType<int64_t>(), recv2.data(), recvcnt.data(), disp.data(), MPIType<int64_t>(), 0, MPI_COMM_WORLD);
        MPI_Gatherv(val.data(), locallen, MPIType<double>(), recv3.data(), recvcnt.data(), disp.data(), MPIType<double>(), 0, MPI_COMM_WORLD);
        
        double t2 = MPI_Wtime() - t1;
        if(myrank == 0)
        {
            cout << "time : " << t2*2 << endl;
        }
    }
	MPI_Finalize();
	return 0;
}


