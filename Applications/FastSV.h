#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <string>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpHelper.h"

/**
 ** Connected components based on Shiloach-Vishkin algorithm
 **/

namespace combblas {

template <typename T1, typename T2>
struct Select2ndMinSR
{
    typedef typename promote_trait<T1,T2>::T_promote T_promote;
    static T_promote id(){ return std::numeric_limits<T_promote>::max(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };

    static T_promote add(const T_promote & arg1, const T_promote & arg2) {
        return std::min(arg1, arg2);
    }

    static T_promote multiply(const T1 & arg1, const T2 & arg2) {
        return static_cast<T_promote> (arg2);
    }

    static void axpy(const T1 a, const T2 & x, T_promote & y) {
        y = add(y, multiply(a, x));
    }
};

template<typename T>
class BinaryMin {
public:
    BinaryMin() = default;
    T operator()(const T &a, const T &b) {
        return std::min(a, b);
    }
};

template <typename IT>
IT LabelCC(FullyDistVec<IT, IT> & father, FullyDistVec<IT, IT> & cclabel)
{
    cclabel = father;
    cclabel.ApplyInd([](IT val, IT ind){return val==ind ? -1 : val;});
    FullyDistSpVec<IT, IT> roots (cclabel, [](IT val) { return val == -1; });
    roots.nziota(0);
    cclabel.Set(roots);
    cclabel = cclabel(father);
    return roots.getnnz();
}

template <class IT, class NT>
int ReduceAssign(FullyDistVec<IT,IT> &ind, FullyDistVec<IT,NT> &val,
        std::vector<std::vector<NT>> &reduceBuffer, NT MAX_FOR_REDUCE)
{
    auto commGrid = ind.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int myrank;
    MPI_Comm_rank(World,&myrank);

    std::vector<int> sendcnt (nprocs,0);
    std::vector<int> recvcnt (nprocs);
    std::vector<std::vector<IT>> indBuf(nprocs);
    std::vector<std::vector<NT>> valBuf(nprocs);

    int loclen = ind.LocArrSize();
    const IT *indices = ind.GetLocArr();
    const IT *values  = val.GetLocArr();
    for(IT i = 0; i < loclen; ++i) {
        IT locind;
        int owner = ind.Owner(indices[i], locind);
        if(reduceBuffer[owner].size() == 0) {
            indBuf[owner].push_back(locind);
            valBuf[owner].push_back(values[i]);
            sendcnt[owner]++;
        }
    }

    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    IT totrecv = std::accumulate(recvcnt.begin(),recvcnt.end(), static_cast<IT>(0));
    double reduceCost = ind.MyLocLength() * log2(nprocs); // bandwidth cost
    IT reducesize = 0;
    std::vector<IT> reducecnt(nprocs,0);
    
    int nreduce = 0;
    if(reduceCost < totrecv)
        reducesize = ind.MyLocLength();
    MPI_Allgather(&reducesize, 1, MPIType<IT>(), reducecnt.data(), 1, MPIType<IT>(), World);
    
    for(int i = 0; i < nprocs; ++i)
        if (reducecnt[i] > 0) nreduce++;
    
    if(nreduce > 0) {
        MPI_Request* requests = new MPI_Request[nreduce];
        MPI_Status* statuses = new MPI_Status[nreduce];

        int ireduce = 0;
        for (int i = 0; i < nprocs; ++i) {
            if(reducecnt[i] > 0) {
                reduceBuffer[i].resize(reducecnt[i], MAX_FOR_REDUCE); // this is specific to LACC
                for (int j = 0; j < sendcnt[i]; j++)
                    reduceBuffer[i][indBuf[i][j]] = std::min(reduceBuffer[i][indBuf[i][j]], valBuf[i][j]);
                if (myrank == i) // recv
                    MPI_Ireduce(MPI_IN_PLACE, reduceBuffer[i].data(), reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
                else // send
                    MPI_Ireduce(reduceBuffer[i].data(), NULL, reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
            }
        }
        MPI_Waitall(nreduce, requests, statuses);
        delete [] requests;
        delete [] statuses;
    }
    return nreduce;
}

template <class IT, class NT>
FullyDistSpVec<IT, NT> Assign(FullyDistVec<IT, IT> &ind, FullyDistVec<IT, NT> &val)
{
    IT globallen = ind.TotalLength();
    auto commGrid = ind.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int * rdispls = new int[nprocs+1];
    int * recvcnt = new int[nprocs];
    int * sendcnt = new int[nprocs](); // initialize to 0
    int * sdispls = new int[nprocs+1];
    
    std::vector<std::vector<NT> > reduceBuffer(nprocs);
    NT MAX_FOR_REDUCE = static_cast<NT>(globallen);
    int nreduce = ReduceAssign(ind, val, reduceBuffer, MAX_FOR_REDUCE);
    
    std::vector<std::vector<IT> > indBuf(nprocs);
    std::vector<std::vector<NT> > valBuf(nprocs);

    int loclen = ind.LocArrSize();
    const IT *indices = ind.GetLocArr();
    const IT *values  = val.GetLocArr();
    for(IT i = 0; i < loclen; ++i) {
        IT locind;
        int owner = ind.Owner(indices[i], locind);
        if(reduceBuffer[owner].size() == 0) {
            indBuf[owner].push_back(locind);
            valBuf[owner].push_back(values[i]);
            sendcnt[owner]++;
        }
    }

    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i = 0; i < nprocs; ++i) {
        sdispls[i + 1] = sdispls[i] + sendcnt[i];
        rdispls[i + 1] = rdispls[i] + recvcnt[i];
    }
    IT totsend = sdispls[nprocs];
    IT totrecv = rdispls[nprocs];
    
    std::vector<IT> sendInd(totsend);
    std::vector<NT> sendVal(totsend);
    for(int i=0; i < nprocs; ++i) {
        std::copy(indBuf[i].begin(), indBuf[i].end(), sendInd.begin()+sdispls[i]);
        std::vector<IT>().swap(indBuf[i]);
        std::copy(valBuf[i].begin(), valBuf[i].end(), sendVal.begin()+sdispls[i]);
        std::vector<NT>().swap(valBuf[i]);
    }
    std::vector<IT> recvInd(totrecv);
    std::vector<NT> recvVal(totrecv);

     MPI_Alltoallv(sendInd.data(), sendcnt, sdispls, MPIType<IT>(), recvInd.data(), recvcnt, rdispls, MPIType<IT>(), World);
    MPI_Alltoallv(sendVal.data(), sendcnt, sdispls, MPIType<IT>(), recvVal.data(), recvcnt, rdispls, MPIType<IT>(), World);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);

    int myrank;
    MPI_Comm_rank(World, &myrank);
    if(reduceBuffer[myrank].size() > 0)
        for(int i = 0; i<reduceBuffer[myrank].size(); i++)
            if(reduceBuffer[myrank][i] < MAX_FOR_REDUCE) {
                recvInd.push_back(i);
                recvVal.push_back(reduceBuffer[myrank][i]);
            }
    
    FullyDistSpVec<IT, NT> indexed(commGrid, globallen, recvInd, recvVal, false, false);
    return indexed;
}

template <class IT, class NT>
int replicate(const FullyDistVec<IT, NT> &dense, const FullyDistVec<IT, IT> &ri, std::vector<std::vector<NT> > &bcastBuffer)
{
    auto commGrid = dense.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    
    std::vector<int> sendcnt (nprocs, 0);
    std::vector<int> recvcnt (nprocs, 0);
    IT length = ri.LocArrSize();
    const IT *p = ri.GetLocArr();
    for(IT i = 0; i < length; ++i) {
        IT locind;
        int owner = dense.Owner(p[i], locind);
        sendcnt[owner]++;
    }
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    IT totrecv = std::accumulate(recvcnt.begin(), recvcnt.end(), static_cast<IT>(0));

    double broadcast_cost = dense.LocArrSize() * log2(nprocs); // bandwidth cost
    IT bcastsize = 0;
    std::vector<IT> bcastcnt(nprocs, 0);
    
    int nbcast = 0;
    if (broadcast_cost < totrecv)
        bcastsize = dense.LocArrSize();
    MPI_Allgather(&bcastsize, 1, MPIType<IT>(), bcastcnt.data(), 1, MPIType<IT>(), World);
    
    for (int i = 0; i < nprocs; i++)
        if (bcastcnt[i] > 0) nbcast++;

    if (nbcast > 0) {
        MPI_Request* requests = new MPI_Request[nbcast];
        MPI_Status* statuses = new MPI_Status[nbcast];
        int ibcast = 0;
        const NT * arr = dense.GetLocArr();
        for(int i = 0; i < nprocs; i++) {
            if (bcastcnt[i] > 0) {
                bcastBuffer[i].resize(bcastcnt[i]);
                std::copy(arr, arr + bcastcnt[i], bcastBuffer[i].begin());
                MPI_Ibcast(bcastBuffer[i].data(), bcastcnt[i], MPIType<NT>(), i, World, &requests[ibcast++]);
            }
        }
        MPI_Waitall(nbcast, requests, statuses);
        delete [] requests;
        delete [] statuses;
    }
    return nbcast;
}

template <class IT, class NT>
FullyDistVec<IT, NT> Extract(const FullyDistVec<IT, NT> &dense, const FullyDistVec<IT, IT> &ri)
{
    auto commGrid = ri.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    
    std::vector<std::vector<NT> > bcastBuffer(nprocs);
    int nbcast = replicate(dense, ri, bcastBuffer);

    std::vector<std::vector<IT> > data_req(nprocs);
    std::vector<std::vector<IT> > revr_map(nprocs);    // to put the incoming data to the correct location
    const NT * arr = dense.GetLocArr();

    IT length = ri.LocArrSize();
    const IT *p = ri.GetLocArr();

    std::vector<IT> q(length);
    for(IT i = 0; i < length; ++i) {
        IT locind;
        int owner = dense.Owner(p[i], locind);
        if(bcastBuffer[owner].size() == 0) {
            data_req[owner].push_back(locind);
            revr_map[owner].push_back(i);
        } else {
            q[i] = bcastBuffer[owner][locind];
        }
    }
    int *sendcnt = new int[nprocs];
    int *sdispls = new int[nprocs];
    for(int i = 0; i < nprocs; ++i)
        sendcnt[i] = (int) data_req[i].size();

    int *rdispls = new int[nprocs];
    int *recvcnt = new int[nprocs];

    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts

    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i = 0; i < nprocs - 1; ++i) {
        sdispls[i + 1] = sdispls[i] + sendcnt[i];
        rdispls[i + 1] = rdispls[i] + recvcnt[i];
    }
    IT totsend = std::accumulate(sendcnt, sendcnt + nprocs, static_cast<IT>(0));
    IT totrecv = std::accumulate(recvcnt, recvcnt + nprocs, static_cast<IT>(0));

    IT *sendbuf = new IT[totsend];
    for(int i = 0; i < nprocs; ++i) {
        std::copy(data_req[i].begin(), data_req[i].end(), sendbuf + sdispls[i]);
        std::vector<IT>().swap(data_req[i]);
    }
    IT *reversemap = new IT[totsend];
    for(int i = 0; i < nprocs; ++i) {
        std::copy(revr_map[i].begin(), revr_map[i].end(), reversemap + sdispls[i]);    // reversemap array is unique
        std::vector<IT>().swap(revr_map[i]);
    }
    IT *recvbuf = new IT[totrecv];
    MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete[] sendbuf;
    // access requested data
    NT *databack = new NT[totrecv];

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i = 0; i < totrecv; ++i)
        databack[i] = arr[recvbuf[i]];
    delete[] recvbuf;
    
    // communicate requested data
    NT *databuf = new NT[totsend];
    // the response counts are the same as the request counts
    MPI_Alltoallv(databack, recvcnt, rdispls, MPIType<IT>(), databuf, sendcnt, sdispls, MPIType<IT>(), World);
    // Create the output from databuf
    for(int i = 0; i < totsend; ++i)
        q[reversemap[i]] = databuf[i];

    DeleteAll(rdispls, recvcnt, databack);
    DeleteAll(sdispls, sendcnt, databuf, reversemap);
    return FullyDistVec<IT, IT>(q, commGrid);
}

template<typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> SV(SpParMat<IT,NT,DER> & A, IT & nCC)
{
    FullyDistVec<IT, IT> D(A.getcommgrid());
    D.iota(A.getnrow(), 0); // D[i] <- i
    FullyDistVec<IT, IT> gp(D);  // grandparent
    FullyDistVec<IT, IT> dup(D); // duplication of grandparent
    FullyDistVec<IT, IT> mngp(D); // minimum neighbor grandparent
    FullyDistVec<IT, IT> mod(D.getcommgrid(), A.getnrow(), 1);
    IT diff = D.TotalLength();
    for (int iter = 1; diff != 0; iter++) {
        if (diff * 50 > A.getnrow()) {
            mngp = SpMV<Select2ndMinSR<NT, IT> >(A, gp); // minimum of neighbors' grandparent
        } else {
            FullyDistSpVec<IT, IT> SpMod(mod, [](IT m){ return m; });
            FullyDistSpVec<IT, IT> SpG = EWiseApply<IT>(SpMod, gp,
                    [](IT m, IT p) { return p; },
                    [](IT m, IT p) { return true; },
                    false, static_cast<IT>(0));
            FullyDistSpVec<IT, IT> hooks(A.getcommgrid(), A.getnrow());
            SpMV<Select2ndMinSR<IT, IT> >(A, SpG, hooks, false);
            mngp.EWiseApply(hooks, BinaryMin<IT>(),
                    [](IT a, IT b){ return true; }, false, A.getnrow());
        }
        FullyDistSpVec<IT, IT> finalhooks = Assign(D, mngp);
        D.Set(finalhooks);
        D.EWiseApply(gp, BinaryMin<IT>());
        D.EWiseApply(mngp, BinaryMin<IT>());
        gp = Extract(D, D);
        dup.EWiseOut(gp, [](IT a, IT b) { return static_cast<IT>(a != b); }, mod);
        diff = static_cast<IT>(mod.Reduce(std::plus<IT>(), static_cast<IT>(0)));
        dup = gp;
        char out[100];
        sprintf(out, "Iteration %d: diff %ld\n", iter, diff);
        SpParHelper::Print(out);
    }
    FullyDistVec<IT, IT> cc(D.getcommgrid());
    nCC = LabelCC(gp, cc);
    return cc;
} /* SV() */

} /* namespace combblas */

