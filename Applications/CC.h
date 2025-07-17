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
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
//#define CC_TIMING 1

#define NONSTAR 0
#define STAR 1
#define CONVERGED 2
using namespace std;

/**
 ** Connected components based on Awerbuch-Shiloach algorithm
 **/

namespace combblas {
    
    template <typename T1, typename T2>
    struct Select2ndMinSR
    {
        typedef typename promote_trait<T1,T2>::T_promote T_promote;
        static T_promote id(){ return std::numeric_limits<T_promote>::max(); };
        static bool returnedSAID() { return false; }
        static MPI_Op mpi_op() { return MPI_MIN; };
        
        static T_promote add(const T_promote & arg1, const T_promote & arg2)
        {
            return std::min(arg1, arg2);
        }
        
        static T_promote multiply(const T1 & arg1, const T2 & arg2)
        {
            return static_cast<T_promote> (arg2);
        }
        
        static void axpy(const T1 a, const T2 & x, T_promote & y)
        {
            y = add(y, multiply(a, x));
        }
    };
    
    
    
    template <class T, class I>
    void omp_par_scan(T* A, T* B,I cnt)
    {
        int p=omp_get_max_threads();
        if(cnt<100*p){
            for(I i=1;i<cnt;i++)
                B[i]=B[i-1]+A[i-1];
            return;
        }
        I step_size=cnt/p;
        
#pragma omp parallel for
        for(int i=0; i<p; i++){
            int start=i*step_size;
            int end=start+step_size;
            if(i==p-1) end=cnt;
            if(i!=0)B[start]=0;
            for(I j=start+1; j<end; j++)
                B[j]=B[j-1]+A[j-1];
        }
        
        T* sum=new T[p];
        sum[0]=0;
        for(int i=1;i<p;i++)
            sum[i]=sum[i-1]+B[i*step_size-1]+A[i*step_size-1];
        
#pragma omp parallel for
        for(int i=1; i<p; i++){
            int start=i*step_size;
            int end=start+step_size;
            if(i==p-1) end=cnt;
            T sum_=sum[i];
            for(I j=start; j<end; j++)
                B[j]+=sum_;
        }
        delete[] sum;
    }
    
    
    
    
    // copied from usort so that we can select k
    // an increased value of k reduces the bandwidth cost, but increases the latency cost
    // this does not work when p is not power of two and a processor is not sending data,
    template <typename T>
    int Mpi_Alltoallv_kway(T* sbuff_, int* s_cnt_, int* sdisp_,
                           T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c, int kway=2)
    {
        int np, pid;
        MPI_Comm_size(c, &np);
        MPI_Comm_rank(c, &pid);
        
        if(np==1 || kway==1)
        {
            return MPI_Alltoallv(sbuff_, s_cnt_, sdisp_, MPIType<T>(), rbuff_, r_cnt_, rdisp_, MPIType<T>(), c);
        }
        
        int range[2]={0,np};
        
        std::vector<int> s_cnt(np);
#pragma omp parallel for
        for(int i=0;i<np;i++){
            s_cnt[i]=s_cnt_[i]*sizeof(T)+2*sizeof(int);
        }
        std::vector<int> sdisp(np); sdisp[0]=0;
        omp_par_scan(&s_cnt[0],&sdisp[0],np);
        
        char* sbuff=new char[sdisp[np-1]+s_cnt[np-1]];
#pragma omp parallel for
        for(int i=0;i<np;i++){
            ((int*)&sbuff[sdisp[i]])[0]=s_cnt[i];
            ((int*)&sbuff[sdisp[i]])[1]=pid;
            memcpy(&sbuff[sdisp[i]]+2*sizeof(int),&sbuff_[sdisp_[i]],s_cnt[i]-2*sizeof(int));
        }
        
        //int t_indx=0;
        int iter_cnt=0;
        while(range[1]-range[0]>1){
            iter_cnt++;
            if(kway>range[1]-range[0])
                kway=range[1]-range[0];
            
            std::vector<int> new_range(kway+1);
            for(int i=0;i<=kway;i++)
                new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
            int p_class=(std::upper_bound(&new_range[0],&new_range[kway],pid)-&new_range[0]-1);
            int new_np=new_range[p_class+1]-new_range[p_class];
            int new_pid=pid-new_range[p_class];
            
            //Communication.
            {
                std::vector<int> r_cnt    (new_np*kway, 0);
                std::vector<int> r_cnt_ext(new_np*kway, 0);
                //Exchange send sizes.
                for(int i=0;i<kway;i++){
                    MPI_Status status;
                    int cmp_np=new_range[i+1]-new_range[i];
                    int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;
                    assert(     (new_pid<cmp_np? true: new_range[i]+new_pid==new_range[i+1]  )); //Remove this.
                    MPI_Sendrecv(&s_cnt[new_range[i]-new_range[0]], cmp_np, MPI_INT, partner, 0,
                                 &r_cnt[new_np   *i ], new_np, MPI_INT, partner, 0, c, &status);
                    
                    //Handle extra communication.
                    if(new_pid==new_np-1 && cmp_np>new_np){
                        int partner=new_range[i+1]-1;
                        std::vector<int> s_cnt_ext(cmp_np, 0);
                        MPI_Sendrecv(&s_cnt_ext[       0], cmp_np, MPI_INT, partner, 0,
                                     &r_cnt_ext[new_np*i], new_np, MPI_INT, partner, 0, c, &status);
                    }
                }
                
                //Allocate receive buffer.
                std::vector<int> rdisp    (new_np*kway, 0);
                std::vector<int> rdisp_ext(new_np*kway, 0);
                int rbuff_size, rbuff_size_ext;
                char *rbuff, *rbuff_ext;
                {
                    omp_par_scan(&r_cnt    [0], &rdisp    [0],new_np*kway);
                    omp_par_scan(&r_cnt_ext[0], &rdisp_ext[0],new_np*kway);
                    rbuff_size     = rdisp    [new_np*kway-1] + r_cnt    [new_np*kway-1];
                    rbuff_size_ext = rdisp_ext[new_np*kway-1] + r_cnt_ext[new_np*kway-1];
                    rbuff     = new char[rbuff_size    ];
                    rbuff_ext = new char[rbuff_size_ext];
                }
                
                //Sendrecv data.
                //*
                int my_block=kway;
                while(pid<new_range[my_block]) my_block--;
                //      MPI_Barrier(c);
                for(int i_=0;i_<=kway/2;i_++){
                    int i1=(my_block+i_)%kway;
                    int i2=(my_block+kway-i_)%kway;
                    
                    for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
                        int i=(i_==0?i1:((j+my_block/i_)%2?i1:i2));
                        MPI_Status status;
                        int cmp_np=new_range[i+1]-new_range[i];
                        int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;
                        
                        int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
                        int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
                        int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;
                        
                        //          ttt=omp_get_wtime();
                        MPI_Sendrecv(&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0,
                                     &rbuff[rdisp[new_np  * i ]], r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ], MPI_BYTE, partner, 0, c, &status);
                        
                        //Handle extra communication.
                        if(pid==new_np-1 && cmp_np>new_np){
                            int partner=new_range[i+1]-1;
                            std::vector<int> s_cnt_ext(cmp_np, 0);
                            MPI_Sendrecv(                       NULL,                                                                       0, MPI_BYTE, partner, 0,
                                         &rbuff[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i], MPI_BYTE, partner, 0, c, &status);
                        }
                    }
                }
                
                //Rearrange received data.
                {
                    if(sbuff!=NULL) delete[] sbuff;
                    sbuff=new char[rbuff_size+rbuff_size_ext];
                    
                    std::vector<int>  cnt_new(2*new_np*kway, 0);
                    std::vector<int> disp_new(2*new_np*kway, 0);
                    for(int i=0;i<new_np;i++)
                        for(int j=0;j<kway;j++){
                            cnt_new[(i*2  )*kway+j]=r_cnt    [j*new_np+i];
                            cnt_new[(i*2+1)*kway+j]=r_cnt_ext[j*new_np+i];
                        }
                    omp_par_scan(&cnt_new[0], &disp_new[0],2*new_np*kway);
                    
#pragma omp parallel for
                    for(int i=0;i<new_np;i++)
                        for(int j=0;j<kway;j++){
                            memcpy(&sbuff[disp_new[(i*2  )*kway+j]], &rbuff    [rdisp    [j*new_np+i]], r_cnt    [j*new_np+i]);
                            memcpy(&sbuff[disp_new[(i*2+1)*kway+j]], &rbuff_ext[rdisp_ext[j*new_np+i]], r_cnt_ext[j*new_np+i]);
                        }
                    
                    //Free memory.
                    if(rbuff    !=NULL) delete[] rbuff    ;
                    if(rbuff_ext!=NULL) delete[] rbuff_ext;
                    
                    s_cnt.clear();
                    s_cnt.resize(new_np,0);
                    sdisp.resize(new_np);
                    for(int i=0;i<new_np;i++){
                        for(int j=0;j<2*kway;j++)
                            s_cnt[i]+=cnt_new[i*2*kway+j];
                        sdisp[i]=disp_new[i*2*kway];
                    }
                }
            }
            
            range[0]=new_range[p_class  ];
            range[1]=new_range[p_class+1];
        }
        
        //Copy data to rbuff_.
        std::vector<char*> buff_ptr(np);
        char* tmp_ptr=sbuff;
        for(int i=0;i<np;i++){
            int& blk_size=((int*)tmp_ptr)[0];
            buff_ptr[i]=tmp_ptr;
            tmp_ptr+=blk_size;
        }
#pragma omp parallel for
        for(int i=0;i<np;i++){
            int& blk_size=((int*)buff_ptr[i])[0];
            int& src_pid=((int*)buff_ptr[i])[1];
            assert(blk_size-2*sizeof(int)<=r_cnt_[src_pid]*sizeof(T));
            memcpy(&rbuff_[rdisp_[src_pid]],buff_ptr[i]+2*sizeof(int),blk_size-2*sizeof(int));
        }
        
        //Free memory.
        if(sbuff   !=NULL) delete[] sbuff;
        return 1;
        
    }
    
    
    
    template <typename T>
    int Mpi_Alltoallv(T* sbuff, int* s_cnt, int* sdisp,
                      T* rbuff, int* r_cnt, int* rdisp, MPI_Comm comm)
    {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);
        
        int commCnt = 0;
        for(int i = 0; i < nprocs; i++)
        {
            if(i==rank) continue;
            if(s_cnt[i] > 0) commCnt++;
            if(r_cnt[i] > 0) commCnt++;
        }
        int totalCommCnt = 0;
        MPI_Allreduce(&commCnt, &totalCommCnt, 1, MPI_INT, MPI_SUM, comm);
        
        if(totalCommCnt < 2*log2(nprocs))
        {
            return par::Mpi_Alltoallv_sparse(sbuff, s_cnt, sdisp, rbuff, r_cnt, rdisp, comm);
        }
        else if((nprocs & (nprocs - 1)) == 0) // processor count is power of 2
        {
            Mpi_Alltoallv_kway(sbuff, s_cnt, sdisp, rbuff, r_cnt, rdisp, comm);
        }
        else
        {
            return MPI_Alltoallv(sbuff, s_cnt, sdisp, MPIType<T>(), rbuff, r_cnt, rdisp, MPIType<T>(), comm);
        }
        
        return 1;
        
    }
    
    
    template <class IT, class NT>
    int replicate(const FullyDistVec<IT,NT> dense, FullyDistSpVec<IT,IT> ri, vector<vector<NT>> &bcastBuffer)
    {
        auto commGrid = dense.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        
        vector<int> sendcnt (nprocs,0);
        vector<int> recvcnt (nprocs,0);
        std::vector<IT> rinum = ri.GetLocalNum();
        IT riloclen = rinum.size();
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = dense.Owner(rinum[i], locind);
            sendcnt[owner]++;
        }
        
        MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
        IT totrecv = std::accumulate(recvcnt.begin(),recvcnt.end(), static_cast<IT>(0));
        
        double broadcast_cost = dense.LocArrSize() * log2(nprocs); // bandwidth cost
        IT bcastsize = 0;
        vector<IT> bcastcnt(nprocs,0);
        
        int nbcast = 0;
        if(broadcast_cost < totrecv)
        {
            bcastsize = dense.LocArrSize();
        }
        MPI_Allgather(&bcastsize, 1, MPIType<IT>(), bcastcnt.data(), 1, MPIType<IT>(), World);
        
        for(int i=0; i<nprocs; i++)
        {
            if(bcastcnt[i]>0) nbcast++;
        }
        
        if(nbcast > 0)
        {
            MPI_Request* requests = new MPI_Request[nbcast];
            assert(requests);
            
            MPI_Status* statuses = new MPI_Status[nbcast];
            assert(statuses);
            
            int ibcast = 0;
            const NT * arr = dense.GetLocArr();
            for(int i=0; i<nprocs; i++)
            {
                if(bcastcnt[i]>0)
                {
                    bcastBuffer[i].resize(bcastcnt[i]);
                    std::copy(arr, arr+bcastcnt[i], bcastBuffer[i].begin());
                    MPI_Ibcast(bcastBuffer[i].data(), bcastcnt[i], MPIType<NT>(), i, World, &requests[ibcast++]);
                }
            }
            
            MPI_Waitall(nbcast, requests, statuses);
            delete [] requests;
            delete [] statuses;
        }
        return nbcast;
    }
    
    // SubRef usign a sparse vector
    // given a dense vector dv and a sparse vector sv
    // sv_out[i]=dv[sv[i]] for all nonzero index i in sv
    // return sv_out
    // If sv has repeated entries, many processes are requesting same entries of dv from the same processes
    // (usually from the low rank processes in LACC)
    // In this case, it may be beneficial to broadcast some entries of dv so that dv[sv[i]] can be obtained locally.
    // This logic is implemented in this function: replicate(dense, ri, bcastBuffer)
    
    template <class IT, class NT>
    FullyDistSpVec<IT,NT> Extract (const FullyDistVec<IT,NT> dense, FullyDistSpVec<IT,IT> ri)
    {
        
#ifdef CC_TIMING
        double ts = MPI_Wtime();
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs<< " Extract timing: ";
#endif
        auto commGrid = ri.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        
        if(!(commGrid == dense.getcommgrid()))
        {
            std::cout << "Grids are not comparable for dense vector subsref" << std::endl;
            return FullyDistSpVec<IT,NT>();
        }
        
        
        
        vector<vector<NT>> bcastBuffer(nprocs);
#ifdef CC_TIMING
        double t1 = MPI_Wtime();
#endif
        int nbcast = replicate(dense, ri, bcastBuffer);
#ifdef CC_TIMING
        double bcast = MPI_Wtime() - t1;
        outs << "bcast ( " << nbcast << " ): " << bcast << " ";
#endif
        
        std::vector< std::vector< IT > > data_req(nprocs);
        std::vector< std::vector< IT > > revr_map(nprocs);    // to put the incoming data to the correct location
        const NT * arr = dense.GetLocArr();
        
        std::vector<IT> rinum = ri.GetLocalNum();
        IT riloclen = rinum.size();
        std::vector<NT> num(riloclen); // final output
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = dense.Owner(rinum[i], locind);
            if(bcastBuffer[owner].size() == 0)
            {
                data_req[owner].push_back(locind);
                revr_map[owner].push_back(i);
            }
            else
            {
                num[i] =bcastBuffer[owner][locind];
            }
        }
        
        int * sendcnt = new int[nprocs];
        int * sdispls = new int[nprocs];
        for(int i=0; i<nprocs; ++i)
            sendcnt[i] = (int) data_req[i].size();
        
        int * rdispls = new int[nprocs];
        int * recvcnt = new int[nprocs];
        
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
#ifdef CC_TIMING
        double all2ll1 = MPI_Wtime() - t1;
        outs << "all2ll1: " << all2ll1 << " ";
        
#endif
        sdispls[0] = 0;
        rdispls[0] = 0;
        for(int i=0; i<nprocs-1; ++i)
        {
            sdispls[i+1] = sdispls[i] + sendcnt[i];
            rdispls[i+1] = rdispls[i] + recvcnt[i];
        }
        IT totsend = std::accumulate(sendcnt,sendcnt+nprocs, static_cast<IT>(0));
        IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
        
        
        IT * sendbuf = new IT[totsend];
        for(int i=0; i<nprocs; ++i)
        {
            std::copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);
            std::vector<IT>().swap(data_req[i]);
        }
        
        
        IT * reversemap = new IT[totsend];
        for(int i=0; i<nprocs; ++i)
        {
            std::copy(revr_map[i].begin(), revr_map[i].end(), reversemap+sdispls[i]);    // reversemap array is unique
            std::vector<IT>().swap(revr_map[i]);
        }
        
        IT * recvbuf = new IT[totrecv];
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        
        Mpi_Alltoallv(sendbuf, sendcnt, sdispls, recvbuf, recvcnt, rdispls, World);
        
#ifdef CC_TIMING
        double all2ll2 = MPI_Wtime() - t1;
        outs << "all2ll2: " << all2ll2 << " ";
#endif
        delete [] sendbuf;
        
        // access requested data
        NT * databack = new NT[totrecv];
        
#ifdef THREADED
#pragma omp parallel for
#endif
        for(int i=0; i<totrecv; ++i)
            databack[i] = arr[recvbuf[i]];
        delete [] recvbuf;
        
        // communicate requested data
        NT * databuf = new NT[totsend];
        // the response counts are the same as the request counts
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        //Mpi_Alltoallv_sparse(databack, recvcnt, rdispls,databuf, sendcnt, sdispls, World);
        
        Mpi_Alltoallv(databack, recvcnt, rdispls,databuf, sendcnt, sdispls, World);
        
        
#ifdef CC_TIMING
        double all2ll3 = MPI_Wtime() - t1;
        outs << "all2ll3: " << all2ll3 << " ";
#endif
        
        // Create the output from databuf
        for(int i=0; i<totsend; ++i)
            num[reversemap[i]] = databuf[i];
        
        DeleteAll(rdispls, recvcnt, databack);
        DeleteAll(sdispls, sendcnt, databuf,reversemap);
        std::vector<IT> ind = ri.GetLocalInd ();
        IT globallen = ri.TotalLength();
        FullyDistSpVec<IT, NT> indexed(commGrid, globallen, ind, num, true, true);
        
#ifdef CC_TIMING
        double total = MPI_Wtime() - ts;
        outs << "others: " << total  - (bcast + all2ll1 + all2ll2 + all2ll3) << " ";
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
        
        return indexed;
        
    }
    
    
    
    template <class IT, class NT>
    int ReduceAssign(FullyDistSpVec<IT,IT> & ind, FullyDistSpVec<IT,NT> & val, vector<vector<NT>> &reduceBuffer, NT MAX_FOR_REDUCE)
    {
        auto commGrid = ind.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        int myrank;
        MPI_Comm_rank(World,&myrank);
        
        vector<int> sendcnt (nprocs,0);
        vector<int> recvcnt (nprocs);
        std::vector<std::vector<IT>> indBuf(nprocs);
        std::vector<std::vector<NT>> valBuf(nprocs);
        std::vector<IT> indices = ind.GetLocalNum();
        std::vector<NT> values = val.GetLocalNum();
        
        IT riloclen = indices.size();
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = ind.Owner(indices[i], locind);
            indBuf[owner].push_back(locind);
            valBuf[owner].push_back(values[i]);
            sendcnt[owner]++;
        }
        
        
        MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
        IT totrecv = std::accumulate(recvcnt.begin(),recvcnt.end(), static_cast<IT>(0));
        double reduceCost = ind.MyLocLength() * log2(nprocs); // bandwidth cost
        IT reducesize = 0;
        vector<IT> reducecnt(nprocs,0);
        
        int nreduce = 0;
        if(reduceCost < totrecv)
        {
            reducesize = ind.MyLocLength();
        }
        MPI_Allgather(&reducesize, 1, MPIType<IT>(), reducecnt.data(), 1, MPIType<IT>(), World);
        
        
        for(int i=0; i<nprocs; ++i)
        {
            if(reducecnt[i]>0) nreduce++;
        }
        
        if(nreduce > 0)
        {
            MPI_Request* requests = new MPI_Request[nreduce];
            assert(requests);
            
            MPI_Status* statuses = new MPI_Status[nreduce];
            assert(statuses);
            
            int ireduce = 0;
            for(int i=0; i<nprocs; ++i)
            {
                if(reducecnt[i]>0)
                {
                    reduceBuffer[i].resize(reducecnt[i], MAX_FOR_REDUCE); // this is specific to LACC
                    for(int j=0; j<sendcnt[i]; j++)
                        reduceBuffer[i][indBuf[i][j]] = std::min(reduceBuffer[i][indBuf[i][j]], valBuf[i][j]);
                    if(myrank==i)
                        MPI_Ireduce(MPI_IN_PLACE, reduceBuffer[i].data(), reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
                    else
                        MPI_Ireduce(reduceBuffer[i].data(), NULL, reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
                }
            }
            
            MPI_Waitall(nreduce, requests, statuses);
            //MPI_Barrier(World);
            delete [] requests;
            delete [] statuses;
        }
        
        
        
        
        return nreduce;
    }
    
    // for fixed value
    template <class IT, class NT>
    int ReduceAssign(FullyDistSpVec<IT,IT> & ind, NT val, vector<vector<NT>> &reduceBuffer, NT MAX_FOR_REDUCE)
    {
        auto commGrid = ind.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        int myrank;
        MPI_Comm_rank(World,&myrank);
        
        vector<int> sendcnt (nprocs,0);
        vector<int> recvcnt (nprocs);
        std::vector<std::vector<IT>> indBuf(nprocs);
        std::vector<IT> indices = ind.GetLocalNum();
        
        IT riloclen = indices.size();
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = ind.Owner(indices[i], locind);
            indBuf[owner].push_back(locind);
            sendcnt[owner]++;
        }
        
        
        MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
        IT totrecv = std::accumulate(recvcnt.begin(),recvcnt.end(), static_cast<IT>(0));
        double reduceCost = ind.MyLocLength() * log2(nprocs); // bandwidth cost
        IT reducesize = 0;
        vector<IT> reducecnt(nprocs,0);
        
        int nreduce = 0;
        if(reduceCost < totrecv)
        {
            reducesize = ind.MyLocLength();
        }
        MPI_Allgather(&reducesize, 1, MPIType<IT>(), reducecnt.data(), 1, MPIType<IT>(), World);
        
        
        for(int i=0; i<nprocs; ++i)
        {
            if(reducecnt[i]>0) nreduce++;
        }
        
        if(nreduce > 0)
        {
            MPI_Request* requests = new MPI_Request[nreduce];
            assert(requests);
            
            MPI_Status* statuses = new MPI_Status[nreduce];
            assert(statuses);
            
            int ireduce = 0;
            for(int i=0; i<nprocs; ++i)
            {
                if(reducecnt[i]>0)
                {
                    reduceBuffer[i].resize(reducecnt[i], MAX_FOR_REDUCE); // this is specific to LACC
                    for(int j=0; j<sendcnt[i]; j++)
                        reduceBuffer[i][indBuf[i][j]] = val;
                    if(myrank==i)
                        MPI_Ireduce(MPI_IN_PLACE, reduceBuffer[i].data(), reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
                    else
                        MPI_Ireduce(reduceBuffer[i].data(), NULL, reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
                }
            }
            
            MPI_Waitall(nreduce, requests, statuses);
            //MPI_Barrier(World);
            delete [] requests;
            delete [] statuses;
        }
        
        
        
        
        return nreduce;
    }
    
    
    // given two sparse vectors sv and val
    // sv_out[sv[i]] = val[i] for all nonzero index i in sv, whre sv_out is the output sparse vector
    // If sv has repeated entries, a process may receive the same values of sv from different processes
    // In this case, it may be beneficial to reduce some entries of sv so that sv_out[sv[i]] can be updated locally.
    // This logic is implemented in this function: ReduceAssign
    
    template <class IT, class NT>
    FullyDistSpVec<IT,NT> Assign (FullyDistSpVec<IT,IT> & ind, FullyDistSpVec<IT,NT> & val)
    {
        IT ploclen = ind.getlocnnz();
        if(ploclen != val.getlocnnz())
        {
            SpParHelper::Print("Assign error: Index and value vectors have different size !!!\n");
            return FullyDistSpVec<IT,NT>(ind.getcommgrid());
        }
        
        IT globallen = ind.TotalLength();
        IT maxInd = ind.Reduce(maximum<IT>(), (IT) 0 ) ;
        if(maxInd >= globallen)
        {
            std::cout << "At least one requested index is larger than the global length" << std::endl;
            return FullyDistSpVec<IT,NT>(ind.getcommgrid());
        }
        
#ifdef CC_TIMING
        double ts = MPI_Wtime();
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs<< " Assign timing: ";
#endif
        auto commGrid = ind.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        int * rdispls = new int[nprocs+1];
        int * recvcnt = new int[nprocs];
        int * sendcnt = new int[nprocs](); // initialize to 0
        int * sdispls = new int[nprocs+1];
        
        vector<vector<NT>> reduceBuffer(nprocs);
        
        
#ifdef CC_TIMING
        double t1 = MPI_Wtime();
#endif
        NT MAX_FOR_REDUCE = static_cast<NT>(globallen);
        int nreduce = ReduceAssign(ind, val, reduceBuffer, MAX_FOR_REDUCE);
        
#ifdef CC_TIMING
        double reduce = MPI_Wtime() - t1;
        outs << "reduce (" << nreduce << "): " << reduce << " ";
#endif
        
        
        
        std::vector<std::vector<IT>> indBuf(nprocs);
        std::vector<std::vector<NT>> valBuf(nprocs);
        std::vector<IT> indices = ind.GetLocalNum();
        std::vector<NT> values = val.GetLocalNum();
        IT riloclen = indices.size();
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = ind.Owner(indices[i], locind);
            if(reduceBuffer[owner].size() == 0)
            {
                indBuf[owner].push_back(locind);
                valBuf[owner].push_back(values[i]);
                sendcnt[owner]++;
            }
        }
        
        
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
#ifdef CC_TIMING
        double all2ll1 = MPI_Wtime() - t1;
        outs << "all2ll1: " << all2ll1 << " ";
#endif
        sdispls[0] = 0;
        rdispls[0] = 0;
        for(int i=0; i<nprocs; ++i)
        {
            sdispls[i+1] = sdispls[i] + sendcnt[i];
            rdispls[i+1] = rdispls[i] + recvcnt[i];
        }
        IT totsend = sdispls[nprocs];
        IT totrecv = rdispls[nprocs];
        
        
        vector<IT> sendInd(totsend);
        vector<NT> sendVal(totsend);
        for(int i=0; i<nprocs; ++i)
        {
            std::copy(indBuf[i].begin(), indBuf[i].end(), sendInd.begin()+sdispls[i]);
            std::vector<IT>().swap(indBuf[i]);
            std::copy(valBuf[i].begin(), valBuf[i].end(), sendVal.begin()+sdispls[i]);
            std::vector<NT>().swap(valBuf[i]);
        }
        
        vector<IT> recvInd(totrecv);
        vector<NT> recvVal(totrecv);
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        
        
        Mpi_Alltoallv(sendInd.data(), sendcnt, sdispls, recvInd.data(), recvcnt, rdispls, World);
        //MPI_Alltoallv(sendInd.data(), sendcnt, sdispls, MPIType<IT>(), recvInd.data(), recvcnt, rdispls, MPIType<IT>(), World);
#ifdef CC_TIMING
        double all2ll2 = MPI_Wtime() - t1;
        outs << "all2ll2: " << all2ll2 << " ";
#endif
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        
        Mpi_Alltoallv(sendVal.data(), sendcnt, sdispls, recvVal.data(), recvcnt, rdispls, World);
        
#ifdef CC_TIMING
        double all2ll3 = MPI_Wtime() - t1;
        outs << "all2ll3: " << all2ll3 << " ";
#endif
        DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
        
        
        int myrank;
        MPI_Comm_rank(World,&myrank);
        if(reduceBuffer[myrank].size()>0)
        {
            //cout << myrank << " : " << recvInd.size() << endl;
            for(int i=0; i<reduceBuffer[myrank].size(); i++)
            {
                
                if(reduceBuffer[myrank][i] < MAX_FOR_REDUCE)
                {
                    recvInd.push_back(i);
                    recvVal.push_back(reduceBuffer[myrank][i]);
                }
            }
        }
        
        FullyDistSpVec<IT, NT> indexed(commGrid, globallen, recvInd, recvVal, false, false);
        
        
        
#ifdef CC_TIMING
        double total = MPI_Wtime() - ts;
        outs << "others: " << total  - (reduce + all2ll1 + all2ll2 + all2ll3) << " ";
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
        return indexed;
        
    }
    
    
    // given a sparse vector sv
    // sv_out[sv[i]] = val for all nonzero index i in sv, whre sv_out is the output sparse vector
    // If sv has repeated entries, a process may receive the same values of sv from different processes
    // In this case, it may be beneficial to reduce some entries of sv so that sv_out[sv[i]] can be updated locally.
    // This logic is implemented in this function: ReduceAssign
    template <class IT, class NT>
    FullyDistSpVec<IT,NT> Assign (FullyDistSpVec<IT,IT> & ind, NT val)
    {
        IT globallen = ind.TotalLength();
        IT maxInd = ind.Reduce(maximum<IT>(), (IT) 0 ) ;
        if(maxInd >= globallen)
        {
            std::cout << "At least one requested index is larger than the global length" << std::endl;
            return FullyDistSpVec<IT,NT>(ind.getcommgrid());
        }
        
#ifdef CC_TIMING
        double ts = MPI_Wtime();
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs<< " Assign timing: ";
#endif
        auto commGrid = ind.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        int nprocs = commGrid->GetSize();
        int * rdispls = new int[nprocs+1];
        int * recvcnt = new int[nprocs];
        int * sendcnt = new int[nprocs](); // initialize to 0
        int * sdispls = new int[nprocs+1];
        
        vector<vector<NT>> reduceBuffer(nprocs);
        
        
#ifdef CC_TIMING
        double t1 = MPI_Wtime();
#endif
        NT MAX_FOR_REDUCE = static_cast<NT>(globallen);
        int nreduce = ReduceAssign(ind, val, reduceBuffer, MAX_FOR_REDUCE);
        
#ifdef CC_TIMING
        double reduce = MPI_Wtime() - t1;
        outs << "reduce ( " << nreduce << " ): " << reduce << " ";
#endif
        
        
        
        std::vector<std::vector<IT>> indBuf(nprocs);
        std::vector<IT> indices = ind.GetLocalNum();
        IT riloclen = indices.size();
        for(IT i=0; i < riloclen; ++i)
        {
            IT locind;
            int owner = ind.Owner(indices[i], locind);
            if(reduceBuffer[owner].size() == 0)
            {
                indBuf[owner].push_back(locind);
                sendcnt[owner]++;
            }
        }
        
        
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
#ifdef CC_TIMING
        double all2ll1 = MPI_Wtime() - t1;
        outs << "all2ll1: " << all2ll1 << " ";
#endif
        sdispls[0] = 0;
        rdispls[0] = 0;
        for(int i=0; i<nprocs; ++i)
        {
            sdispls[i+1] = sdispls[i] + sendcnt[i];
            rdispls[i+1] = rdispls[i] + recvcnt[i];
        }
        IT totsend = sdispls[nprocs];
        IT totrecv = rdispls[nprocs];
        
        
        vector<IT> sendInd(totsend);
        for(int i=0; i<nprocs; ++i)
        {
            std::copy(indBuf[i].begin(), indBuf[i].end(), sendInd.begin()+sdispls[i]);
            std::vector<IT>().swap(indBuf[i]);
        }
        
        vector<IT> recvInd(totrecv);
#ifdef CC_TIMING
        t1 = MPI_Wtime();
#endif
        
        
        Mpi_Alltoallv(sendInd.data(), sendcnt, sdispls, recvInd.data(), recvcnt, rdispls, World);
        //MPI_Alltoallv(sendInd.data(), sendcnt, sdispls, MPIType<IT>(), recvInd.data(), recvcnt, rdispls, MPIType<IT>(), World);
#ifdef CC_TIMING
        double all2ll2 = MPI_Wtime() - t1;
        outs << "all2ll2: " << all2ll2 << " ";
        outs << "all2ll3: " << 0 << " ";
#endif
        DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
        
        
        int myrank;
        MPI_Comm_rank(World,&myrank);
        vector<NT> recvVal(totrecv);
        if(reduceBuffer[myrank].size()>0)
        {
            //cout << myrank << " : " << recvInd.size() << endl;
            for(int i=0; i<reduceBuffer[myrank].size(); i++)
            {
                if(reduceBuffer[myrank][i] < MAX_FOR_REDUCE)
                {
                    recvInd.push_back(i);
                    recvVal.push_back(val);
                }
            }
        }
        FullyDistSpVec<IT, NT> indexed(commGrid, globallen, recvInd, recvVal, false, false);
        
#ifdef CC_TIMING
        double total = MPI_Wtime() - ts;
        outs << "others: " << total  - (reduce + all2ll1 + all2ll2) << " ";
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
        return indexed;
        
    }
    
    
    
    
    // special starcheck after conditional and unconditional hooking
    template <typename IT, typename NT, typename DER>
    void StarCheckAfterHooking(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & parent, FullyDistVec<IT,short>& star, FullyDistSpVec<IT,IT> condhooks, bool isStar2StarHookPossible)
    {
        // hooks are nonstars
        star.EWiseApply(condhooks, [](short isStar, IT x){return static_cast<short>(NONSTAR);},
                        false, static_cast<IT>(NONSTAR));
        
        if(isStar2StarHookPossible)
        {
            // this is not needed in the first iteration see the complicated proof in the paper
            // parents of hooks are nonstars
            // needed only after conditional hooking because in that case star can hook to a star
            FullyDistSpVec<IT, short> pNonStar= Assign(condhooks, NONSTAR);
            star.Set(pNonStar);
        }
        
        //star(parent)
        // If I am a star, I would like to know the star information of my parent
        // children of hooks and parents of hooks are nonstars
        // NOTE: they are not needed in the first iteration
        
        FullyDistSpVec<IT,short> spStars(star, [](short isStar){return isStar==STAR;});
        FullyDistSpVec<IT, IT> parentOfStars = EWiseApply<IT>(spStars, parent,
                                                              [](short isStar, IT p){return p;},
                                                              [](short isStar, IT p){return true;},
                                                              false, static_cast<short>(0));
        FullyDistSpVec<IT,short> isParentStar = Extract(star, parentOfStars);
        star.Set(isParentStar);
    }
    
    /*
     // In iteration 1: "stars" has both vertices belongihg to stars and nonstars (no converged)
     //                  we only process nonstars and identify starts from them
     // After iteration 1: "stars" has vertices belongihg to converged and nonstars (no stars)
     //                  we only process nonstars and identify starts from them
     template <typename IT>
     void StarCheck(FullyDistVec<IT, IT> & parents, FullyDistVec<IT,short>& stars)
     {
     
     // this is done here so that in the first iteration, we don't process STAR vertices
     FullyDistSpVec<IT,short> nonStars(stars, [](short isStar){return isStar==NONSTAR;});
     // initialize all nonstars to stars
     stars.Apply([](short isStar){return isStar==NONSTAR? STAR: isStar;});
     
     // identify vertices at level >= 2 (grandchildren of roots)
     FullyDistSpVec<IT, IT> pOfNonStars = EWiseApply<IT>(nonStars, parents,
     [](short isStar, IT p){return p;},
     [](short isStar, IT p){return true;},
     false, static_cast<short>(0));
     FullyDistSpVec<IT,IT> gpOfNonStars = Extract(parents, pOfNonStars);
     
     FullyDistSpVec<IT,short> keptNonStars = EWiseApply<short>(pOfNonStars, gpOfNonStars,
     [](IT p, IT gp){return static_cast<short>(NONSTAR);},
     [](IT p, IT gp){return p!=gp;},
     false, false, static_cast<IT>(0), static_cast<IT>(0));
     stars.Set(keptNonStars); // setting level > 2 vertices as nonstars
     
     // identify grand parents of kept nonstars
     FullyDistSpVec<IT,IT> gpOfKeptNonStars = EWiseApply<IT>(pOfNonStars, gpOfNonStars,
     [](IT p, IT gp){return gp;},
     [](IT p, IT gp){return p!=gp;},
     false, false, static_cast<IT>(0), static_cast<IT>(0));
     
     //FullyDistSpVec<IT, short> fixedNS = gpOfKeptNonStars;
     //fixedNS = NONSTAR;
     FullyDistSpVec<IT, short> gpNonStar= Assign(gpOfKeptNonStars, NONSTAR);
     stars.Set(gpNonStar);
     
     
     // remaining vertices: level-1 leaves of nonstars and any vertices in previous stars (iteration 1 only)
     FullyDistSpVec<IT,short> spStars(stars, [](short isStar){return isStar==STAR;});
     // further optimization can be done to remove previous stars
     
     FullyDistSpVec<IT, IT> pOfStars = EWiseApply<IT>(spStars, parents,
     [](short isStar, IT p){return p;},
     [](short isStar, IT p){return true;},
     false, static_cast<short>(0));
     
     FullyDistSpVec<IT,short> isParentStar = Extract(stars, pOfStars);
     stars.Set(isParentStar);
     }
     */
    
    // In iteration>1:
    //  We have only CONVERGED or NONSTAR vertices
    //  some of the NONSTAR vertices may become STAR in the last shortcut operation
    //  We would like to identify those new stars
    // In iteration 1:
    //  we have STAR and NONSTAR vertices
    //  every hooked vertex is marked as NONSTARs
    //  roots are marked as STARs (includign singletones)
    template <typename IT>
    void StarCheck(FullyDistVec<IT, IT> & parents, FullyDistVec<IT,short>& stars)
    {
        
        // this is done here so that in the first iteration, we don't process STAR vertices
        // all current nonstars
        FullyDistSpVec<IT,short> nonStars(stars, [](short isStar){return isStar==NONSTAR;});
        
        // initialize all nonstars to stars
        stars.Apply([](short isStar){return isStar==NONSTAR? STAR: isStar;});
        
        // parents of all current nonstars
        FullyDistSpVec<IT, IT> pOfNonStars = EWiseApply<IT>(nonStars, parents,
                                                            [](short isStar, IT p){return p;},
                                                            [](short isStar, IT p){return true;},
                                                            false, static_cast<short>(0));
        
        // parents of all current nonstars indexed by parent
        // any vertex with a child should be here
        // leaves are not present as indices, but roots are present
        FullyDistSpVec<IT,short> pOfNonStarsIdx = Assign(pOfNonStars, NONSTAR);
        // copy parent information (the values are grandparents)
        FullyDistSpVec<IT,IT> gpOfNonStars_pindexed = EWiseApply<IT>(pOfNonStarsIdx, parents,
                                                                     [](short isStar, IT p){return p;},
                                                                     [](short isStar, IT p){return true;},
                                                                     false, static_cast<short>(0));
        // identify if they are parents/grandparents of a vertex with level > 2
        FullyDistSpVec<IT,IT> temp = gpOfNonStars_pindexed;
        temp.setNumToInd();
        gpOfNonStars_pindexed = EWiseApply<IT>(temp, gpOfNonStars_pindexed,
                                               [](IT p, IT gp){return gp;},
                                               [](IT p, IT gp){return p!=gp;},
                                               false, false, static_cast<IT>(0), static_cast<IT>(0));
        
        // index has parents of vertices with level > 2
        // value has grand parents of vertices with level > 2
        // update parents
        // All vertices (except the root and leave ) in a non-star tree will be updated
        stars.EWiseApply(gpOfNonStars_pindexed, [](short isStar, IT idx){return static_cast<short>(NONSTAR);},
                         false, static_cast<IT>(NONSTAR));
        
        // now everything is updated except the root and leaves of nonstars
        // identify roots (indexed by level-1 vertices)
        FullyDistSpVec<IT,IT> rootsOfNonStars = EWiseApply<IT>(pOfNonStars, stars,
                                                               [](IT p, short isStar){return p;},
                                                               [](IT p, short isStar){return isStar==NONSTAR;},
                                                               false, static_cast<IT>(0));
        
        
        FullyDistSpVec<IT,short> rootsOfNonStarsIdx = Assign(rootsOfNonStars, NONSTAR);
        stars.Set( rootsOfNonStarsIdx);
        
        
        // remaining vertices
        // they must be stars (created after the shortcut) or level-1 leaves of a non-star
        FullyDistSpVec<IT,IT> pOflevel1V = EWiseApply<IT>(nonStars, stars,
                                                          [](short s, short isStar){return static_cast<IT> (s);},
                                                          [](short s, short isStar){return isStar==STAR;},
                                                          false, static_cast<short>(0));
        pOflevel1V = EWiseApply<IT>(pOflevel1V, parents,
                                    [](IT s, IT p){return p;},
                                    [](IT s, IT p){return true;},
                                    false, static_cast<IT>(0));
        
        FullyDistSpVec<IT,short> isParentStar = Extract(stars, pOflevel1V);
        stars.Set(isParentStar);
    }
    
    
    template <typename IT, typename NT, typename DER>
    FullyDistSpVec<IT, IT> ConditionalHook(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & parent, FullyDistVec<IT,short> stars, int iteration)
    {
        
#ifdef CC_TIMING
        double t1 = MPI_Wtime();
#endif
        FullyDistVec<IT, IT> minNeighborparent ( A.getcommgrid());
        minNeighborparent = SpMV<Select2ndMinSR<NT, IT>>(A, parent); // value is the minimum of all neighbors' parents
        
#ifdef CC_TIMING
        double tspmv =  MPI_Wtime() - t1;
#endif
        
        FullyDistSpVec<IT,IT> hooksMNP(stars, [](short isStar){return isStar==STAR;});
        hooksMNP = EWiseApply<IT>(hooksMNP, minNeighborparent, [](IT x, IT mnp){return mnp;},
                                  [](IT x, IT mnp){return true;}, false, static_cast<IT> (0));
        hooksMNP = EWiseApply<IT>(hooksMNP, parent, [](IT mnp, IT p){return mnp;},
                                  [](IT mnp, IT p){return p > mnp;}, false, static_cast<IT> (0));
        
        FullyDistSpVec<IT, IT> finalhooks (A.getcommgrid());
        if(iteration == 1)
        {
            finalhooks = hooksMNP;
        }
        else
        {
            FullyDistSpVec<IT,IT> hooksP = EWiseApply<IT>(hooksMNP, parent, [](IT mnp, IT p){return p;},
                                                          [](IT mnp, IT p){return true;}, false, static_cast<IT> (0));
            
            finalhooks = Assign(hooksP, hooksMNP);
            
        }
        parent.Set(finalhooks);
        
#ifdef CC_TIMING
        double tall =  MPI_Wtime() - t1;
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs << " Conditional Hooking Time: SpMV: " << tspmv << " Other: "<< tall-tspmv;
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
        return finalhooks;
    }
    
    
    template <typename IT, typename NT, typename DER>
    FullyDistSpVec<IT, IT> UnconditionalHook2(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & parents, FullyDistVec<IT,short> stars)
    {
        
#ifdef CC_TIMING
        double ts =  MPI_Wtime();
        double t1, tspmv;
#endif
        string spmv = "dense";
        IT nNonStars = stars.Reduce(std::plus<IT>(), static_cast<IT>(0), [](short isStar){return static_cast<IT>(isStar==NONSTAR);});
        IT nv = A.getnrow();
        
        FullyDistSpVec<IT, IT> hooks(A.getcommgrid(), nv);
        
        if(nNonStars * 50 < nv) // use SpMSpV
        {
            spmv = "sparse";
            FullyDistSpVec<IT,IT> nonStars(stars, [](short isStar){return isStar==NONSTAR;});
            FullyDistSpVec<IT, IT> pOfNonStars = EWiseApply<IT>(nonStars, parents,
                                                                [](short isStar, IT p){return p;},
                                                                [](short isStar, IT p){return true;},
                                                                false, static_cast<IT>(0));
            //hooks = SpMV<Select2ndMinSR<NT, IT>>(A, pOfNonStars);
#ifdef CC_TIMING
            t1 = MPI_Wtime();
#endif
            SpMV<Select2ndMinSR<NT, IT>>(A, pOfNonStars, hooks, false);
#ifdef CC_TIMING
            tspmv =  MPI_Wtime() - t1;
#endif
            hooks = EWiseApply<IT>(hooks, stars, [](IT mnp, short isStar){return mnp;},
                                   [](IT mnp, short isStar){return isStar==STAR;},
                                   false, static_cast<IT> (0));
        }
        else // use SpMV
        {
            FullyDistVec<IT, IT> parents1 = parents;
            parents1.EWiseApply(stars, [nv](IT p, short isStar){return isStar == STAR? nv: p;});
            
            
            FullyDistVec<IT, IT> minNeighborParent ( A.getcommgrid());
#ifdef CC_TIMING
            t1 = MPI_Wtime();
#endif
            minNeighborParent = SpMV<Select2ndMinSR<NT, IT>>(A, parents1); // value is the minimum of all neighbors' parents
#ifdef CC_TIMING
            tspmv =  MPI_Wtime() - t1;
#endif
            hooks = minNeighborParent.Find([nv](IT mnf){return mnf != nv;});
            hooks = EWiseApply<IT>(hooks, stars, [](IT mnp, short isStar){return mnp;},
                                   [](IT mnp, short isStar){return isStar==STAR;},
                                   false, static_cast<IT> (0));
        }
        
        
        
        
        FullyDistSpVec<IT,IT> hooksP = EWiseApply<IT>(hooks, parents, [](IT mnp, IT p){return p;},
                                                      [](IT mnp, IT p){return true;}, false, static_cast<IT> (0));
        
        FullyDistSpVec<IT, IT> finalHooks = Assign(hooksP, hooks);
        parents.Set(finalHooks);
        
#ifdef CC_TIMING
        double tall =  MPI_Wtime() - ts;
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs << " Unconditional Hooking Time " << spmv << " : " << tspmv << " Other: "<< tall-tspmv;
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
        
        return finalHooks;
        
    }
    
    
    
    template <typename IT>
    void Shortcut(FullyDistVec<IT, IT> & parent)
    {
        FullyDistVec<IT, IT> grandparent = parent(parent);
        parent = grandparent; // we can do it unconditionally because it is trivially true for stars
    }
    
    // before shortcut, we will make all remaining start as inactive
    // shortcut only on nonstar vertices
    // then find stars on nonstar vertices
    template <typename IT>
    void Shortcut(FullyDistVec<IT, IT> & parents, FullyDistVec<IT,short> stars)
    {
        FullyDistSpVec<IT,short> spNonStars(stars, [](short isStar){return isStar==NONSTAR;});
        FullyDistSpVec<IT, IT> parentsOfNonStars = EWiseApply<IT>(spNonStars, parents,
                                                                  [](short isStar, IT p){return p;},
                                                                  [](short isStar, IT p){return true;},
                                                                  false, static_cast<short>(0));
        FullyDistSpVec<IT,IT> grandParentsOfNonStars = Extract(parents, parentsOfNonStars);
        parents.Set(grandParentsOfNonStars);
    }
    
    
    
    
    template <typename IT, typename NT, typename DER>
    bool neigborsInSameCC(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & cclabel)
    {
        FullyDistVec<IT, IT> minNeighborCCLabel ( A.getcommgrid());
        minNeighborCCLabel = SpMV<Select2ndMinSR<NT, IT>>(A, cclabel);
        return minNeighborCCLabel==cclabel;
    }
    
    
    // works only on P=1
    template <typename IT, typename NT, typename DER>
    void Correctness(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & cclabel, IT nCC, FullyDistVec<IT,IT> parent)
    {
        DER* spSeq = A.seqptr(); // local submatrix
        
        for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
        {
            IT j = colit.colid(); // local numbering
            for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
            {
                IT i = nzit.rowid();
                if( cclabel[i] != cclabel[j])
                {
                    std::cout << i << " (" << parent[i] << ", "<< cclabel[i] << ") & "<< j << "("<< parent[j] << ", " << cclabel[j] << ")\n";
                }
            }
            
        }
    }
    
    // Input:
    // parent: parent of each vertex. parent is essentilly the root of the star which a vertex belongs to.
    //          parent of the root is itself
    // Output:
    // cclabel: connected components are incrementally labeled
    // returns the number of connected components
    // Example: input = [0, 0, 2, 3, 0, 2], output = (0, 0, 1, 2, 0, 1), return 3
    template <typename IT>
    IT LabelCC(FullyDistVec<IT, IT> & parent, FullyDistVec<IT, IT> & cclabel)
    {
        cclabel = parent;
        cclabel.ApplyInd([](IT val, IT ind){return val==ind ? -1 : val;});
        
        FullyDistSpVec<IT, IT> roots (cclabel, [](IT val){return val == -1;});
        // parents of leaves are still correct
        FullyDistSpVec<IT, IT> pOfLeaves (cclabel, [](IT val){return val != -1;});
        
        roots.nziota(0);
        cclabel.Set(roots);
        
        
        FullyDistSpVec<IT,IT> labelOfParents = Extract(cclabel, pOfLeaves);
        cclabel.Set(labelOfParents);
        //cclabel = cclabel(parent);
        return roots.getnnz();
    }
    
    
    template <typename IT, typename NT, typename DER>
    FullyDistVec<IT, IT> CC(SpParMat<IT,NT,DER> & A, IT & nCC)
    {
        IT nrows = A.getnrow();
        //A.AddLoops(1); // needed for isolated vertices: not needed anymore
        FullyDistVec<IT,IT> parent(A.getcommgrid());
        parent.iota(nrows, 0);    // parent(i)=i initially
        FullyDistVec<IT,short> stars(A.getcommgrid(), nrows, STAR);// initially every vertex belongs to a star
        int iteration = 1;
        std::ostringstream outs;
        
        // isolated vertices are marked as converged
        FullyDistVec<int64_t,double> degree = A.Reduce(Column, plus<double>(), 0.0, [](double val){return 1.0;});
        stars.EWiseApply(degree, [](short isStar, double degree){return degree == 0.0? CONVERGED: isStar;});
        
        int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
#endif
        SpParMat<IT,bool,SpDCCols < IT, bool >>  Abool = A;
        Abool.ActivateThreading(nthreads*4);
        
        
        while (true)
        {
#ifdef CC_TIMING
            double t1 = MPI_Wtime();
#endif
            FullyDistSpVec<IT, IT> condhooks = ConditionalHook(Abool, parent, stars, iteration);
#ifdef CC_TIMING
            double t_cond_hook =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            // Any iteration other than the first iteration,
            // a non-star is formed after a conditional hooking
            // In the first iteration, we can hook two vertices to create a star
            // After the first iteratio, only singletone CCs reamin isolated
            // Here, we are ignoring the first iteration (still correct, but may ignore few possible
            // unconditional hooking in the first iteration)
            // remove cond hooks from stars
            
            if(iteration > 1)
            {
                StarCheckAfterHooking(Abool, parent, stars, condhooks, true);
            }
            else
            {
                // explain
                stars.EWiseApply(condhooks, [](short isStar, IT x){return static_cast<short>(NONSTAR);},
                                 false, static_cast<IT>(NONSTAR));
                FullyDistSpVec<IT, short> pNonStar= Assign(condhooks, NONSTAR);
                stars.Set(pNonStar);
                // it does not create any cycle in the unconditional hooking, see the proof in the paper
            }
            
#ifdef CC_TIMING
            double t_starcheck1 =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            
            FullyDistSpVec<IT, IT> uncondHooks = UnconditionalHook2(Abool, parent, stars);
#ifdef CC_TIMING
            double t_uncond_hook =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            
            if(iteration > 1)
            {
                StarCheckAfterHooking(Abool, parent, stars, uncondHooks, false);
                stars.Apply([](short isStar){return isStar==STAR? CONVERGED: isStar;});
            }
            else
            {
                // explain
                stars.EWiseApply(uncondHooks, [](short isStar, IT x){return static_cast<short>(NONSTAR);},
                                 false, static_cast<IT>(NONSTAR));
            }
            
            IT nconverged = stars.Reduce(std::plus<IT>(), static_cast<IT>(0), [](short isStar){return static_cast<IT>(isStar==CONVERGED);});
            
            if(nconverged==nrows)
            {
                outs.clear();
                outs << "Iteration: " << iteration << " converged: " << nrows << " stars: 0" << " nonstars: 0" ;
                outs<< endl;
                SpParHelper::Print(outs.str());
                break;
            }
            
#ifdef CC_TIMING
            double t_starcheck2 =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            Shortcut(parent, stars);
#ifdef CC_TIMING
            double t_shortcut =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            
            
            StarCheck(parent, stars);
#ifdef CC_TIMING
            double t_starcheck =  MPI_Wtime() - t1;
            t1 = MPI_Wtime();
#endif
            IT nonstars = stars.Reduce(std::plus<IT>(), static_cast<IT>(0), [](short isStar){return static_cast<IT>(isStar==NONSTAR);});
            IT nstars = nrows - (nonstars + nconverged);
            
            
            
            
            
            
            
            double t2 = MPI_Wtime();
            outs.str("");
            outs.clear();
            outs << "Iteration: " << iteration << " converged: " << nconverged << " stars: " << nstars << " nonstars: " << nonstars;
#ifdef CC_TIMING
            //outs << " Time:  t_cond_hook: " << t_cond_hook << " t_starcheck1: " << t_starcheck1 << " t_uncond_hook: " << t_uncond_hook << " t_starcheck2: " << t_starcheck2 << " t_shortcut: " << t_shortcut << " t_starcheck: " << t_starcheck;
#endif
            outs<< endl;
            SpParHelper::Print(outs.str());
            
            iteration++;
            
            
        }
        
        FullyDistVec<IT, IT> cc(parent.getcommgrid());
        nCC = LabelCC(parent, cc);
        
        // TODO: Print to file
        //PrintCC(cc, nCC);
        //Correctness(A, cc, nCC, parent);
        
        return cc;
    }
    
    
    template <typename IT>
    void PrintCC(FullyDistVec<IT, IT> CC, IT nCC)
    {
        for(IT i=0; i< nCC; i++)
        {
            FullyDistVec<IT, IT> ith = CC.FindInds([i](IT val){return val == i;});
            ith.DebugPrint();
        }
    }
    
    // Print the size of the first 4 clusters
    template <typename IT>
    void First4Clust(FullyDistVec<IT, IT>& cc)
    {
        FullyDistSpVec<IT, IT> cc1 = cc.Find([](IT label){return label==0;});
        FullyDistSpVec<IT, IT> cc2 = cc.Find([](IT label){return label==1;});
        FullyDistSpVec<IT, IT> cc3 = cc.Find([](IT label){return label==2;});
        FullyDistSpVec<IT, IT> cc4 = cc.Find([](IT label){return label==3;});
        
        std::ostringstream outs;
        outs.str("");
        outs.clear();
        outs << "Size of the first component: " << cc1.getnnz() << std::endl;
        outs << "Size of the second component: " << cc2.getnnz() << std::endl;
        outs << "Size of the third component: " << cc3.getnnz() << std::endl;
        outs << "Size of the fourth component: " << cc4.getnnz() << std::endl;
    }
    
    
    template <typename IT>
    void HistCC(FullyDistVec<IT, IT> CC, IT nCC)
    {
        FullyDistVec<IT, IT> ccSizes(CC.getcommgrid(), nCC, 0);
        for(IT i=0; i< nCC; i++)
        {
            FullyDistSpVec<IT, IT> ith = CC.Find([i](IT val){return val == i;});
            ccSizes.SetElement(i, ith.getnnz());
        }
        
        IT largestCCSise = ccSizes.Reduce(maximum<IT>(), static_cast<IT>(0));
        
        
        const IT * locCCSizes = ccSizes.GetLocArr();
        int numBins = 200;
        std::vector<IT> localHist(numBins,0);
        for(IT i=0; i< ccSizes.LocArrSize(); i++)
        {
            IT bin = (locCCSizes[i]*(numBins-1))/largestCCSise;
            localHist[bin]++;
        }
        
        std::vector<IT> globalHist(numBins,0);
        MPI_Comm world = CC.getcommgrid()->GetWorld();
        MPI_Reduce(localHist.data(), globalHist.data(), numBins, MPIType<IT>(), MPI_SUM, 0, world);
        
        
        int myrank;
        MPI_Comm_rank(world,&myrank);
        if(myrank==0)
        {
            std::cout << "The largest component size: " << largestCCSise  << std::endl;
            std::ofstream output;
            output.open("hist.txt", std::ios_base::app );
            std::copy(globalHist.begin(), globalHist.end(), std::ostream_iterator<IT> (output, " "));
            output << std::endl;
            output.close();
        }
        
        
        //ccSizes.PrintToFile("histCC.txt");
    }
    
}

