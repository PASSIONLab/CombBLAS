#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"
#include "../CC.h"
#include "../WriteMCLClusters.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

int main(int argc, char* argv[])
{
    int nprocs, myrank, nthreads = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(true){ // To enforce that no MPI calls happen after MPI_Finalize()
        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        FullyDistVec<IT, IT> p( fullWorld);
        FullyDistVec<IT, IT> q( fullWorld);
        IT np = 20, nq = 10;
        p.iota(np, 0);
        q.iota(nq, np);

        //for (IT i = 0; i < np; i++){
            //IT x = p[i];
            //if(myrank == 0) printf("p[%ld]: %ld\n", i, x);
        //}

        //if(myrank == 0) printf("---\n");

        //for (IT i = 0; i < nq; i++){
            //IT x = q[i];
            //if(myrank == 0) printf("q[%ld]: %ld\n", i, x);
        //}

        //MPI_Barrier(MPI_COMM_WORLD);

        IT pLocLen = p.LocArrSize();
        IT qLocLen = q.LocArrSize();
        IT minLocLen = std::min(pLocLen, qLocLen);
        IT maxLocLen = std::max(pLocLen, qLocLen);
        std::vector<bool> pLocFlag(pLocLen, true);
        std::vector<bool> qLocFlag(qLocLen, true);

        std::mt19937 rng;
        rng.seed(myrank);
        std::uniform_int_distribution<IT> uidist(0, 999999999);
        std::uniform_real_distribution<NT> urdist(0, 1.0);
        for (IT i = 0; i < minLocLen; i++){
            if(urdist(rng) < double(minLocLen)/maxLocLen){
                IT pidx = uidist(rng) % pLocLen;
                while(pLocFlag[pidx] == false) pidx++;
                IT qidx = uidist(rng) % qLocLen;
                while(qLocFlag[qidx] == false) qidx++;
                //printf("myrank %d, swap %d: pidx %d <-> qidx %d\n", myrank, i, pidx, qidx);
                IT pv = p.GetLocalElement(pidx);
                IT qv = q.GetLocalElement(qidx);
                p.SetLocalElement(pidx, qv);
                q.SetLocalElement(qidx, pv);
                pLocFlag[pidx] = false;
                qLocFlag[qidx] = false;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        p.RandPerm(31416);
        q.RandPerm(31416);
        
        MPI_Barrier(MPI_COMM_WORLD);

        if(myrank == 0) printf("*** After Shuffle ***\n");

        for (IT i = 0; i < np; i++){
            IT x = p[i];
            if(myrank == 0) printf("p[%ld]: %ld\n", i, x);
        }

        if(myrank == 0) printf("---\n");

        for (IT i = 0; i < nq; i++){
            IT x = q[i];
            if(myrank == 0) printf("q[%ld]: %ld\n", i, x);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        FullyDistVec<IT,IT> temp(fullWorld, np+nq, 0); // Dummy vector to help calculate which process owns which index

        std::vector<int> sendcnt(nprocs, 0);
        std::vector<int> sdispls(nprocs+1);
        std::vector<int> recvcnt(nprocs, 0);
        std::vector<int> rdispls(nprocs+1);

        const std::vector<IT> pLocVec = p.GetLocVec();
        const std::vector<IT> qLocVec = q.GetLocVec();
        for (IT i = 0; i < pLocLen; i++){
            IT rLocIdx; // Index of the local array in the receiver side
            int owner = temp.Owner(pLocVec[i], rLocIdx);
            sendcnt[owner] = sendcnt[owner] + 1;
        }
        for (IT i = 0; i < qLocLen; i++){
            IT rLocIdx; // Index of the local array in the receiver side
            int owner = temp.Owner(qLocVec[i], rLocIdx);
            sendcnt[owner] = sendcnt[owner] + 1;
        }

        MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, p.getcommgrid()->GetWorld() );

        sdispls[0] = 0;
        rdispls[0] = 0;
        std::partial_sum(sendcnt.begin(), sendcnt.end(), sdispls.begin()+1);
        std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin()+1);

        int totsend = sdispls[sdispls.size()-1];
        int totrecv = rdispls[rdispls.size()-1];

        std::vector< std::tuple<IT, IT> > sendTuples(totsend);
        std::vector< std::tuple<IT, IT> > recvTuples(totrecv);

        //printf("myrank %d: totsend %d, totrecv %d\n", myrank, totsend, totrecv);

        std::vector<int> sidx(sdispls); // Copy sdispls array to use for preparing sendTuples

        for (IT i = 0; i < pLocLen; i++){
            IT rLocIdx; // Index of the local array in the receiver side
            int owner = temp.Owner(pLocVec[i], rLocIdx);
            //if(sidx[owner] >= totsend){
                //printf("myrank %d: sidx[%d] %d/%d\n", myrank, owner, sidx[owner], totsend-1);
            //}
            sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, pLocVec[i]);
            sidx[owner]++;
        }
        for (IT i = 0; i < qLocLen; i++){
            IT rLocIdx; // Index of the local array in the receiver side
            int owner = temp.Owner(qLocVec[i], rLocIdx);
            sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, qLocVec[i]);
            sidx[owner]++;
        }

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<IT,IT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);
        MPI_Alltoallv(sendTuples.data(), sendcnt.data(), sdispls.data(), MPI_tuple, recvTuples.data(), recvcnt.data(), rdispls.data(), MPI_tuple, p.getcommgrid()->GetWorld());

        std::vector<IT> rLocVec(totrecv);
        for(int i = 0; i < totrecv; i++){
            IT rLocIdx = std::get<0>(recvTuples[i]);
            IT rLocVal = std::get<1>(recvTuples[i]);
            rLocVec[rLocIdx] = rLocVal;
        }

        FullyDistVec<IT, IT> a(rLocVec, fullWorld);


        MPI_Barrier(MPI_COMM_WORLD);

        if(myrank == 0) printf("*** After Merge ***\n");

        for (IT i = 0; i < np + nq; i++){
            IT x = a[i];
            if(myrank == 0) printf("a[%ld]: %ld\n", i, x);
        }

    }

    MPI_Finalize();
    return 0;
}
