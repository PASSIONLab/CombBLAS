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



#include "SpParMat3D.h"
#include "ParFriends.h"
#include "Operations.h"
#include "FileHeader.h"
extern "C" {
#include "mmio.h"
}
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <fstream>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace combblas
{
    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool csplit, bool special)
    {
        colsplit = csplit;
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        int nprocs = commGrid2D->GetSize();
        if(colsplit) commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, true, true));
        else commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, false, true));

        DER* spSeq = A2D.seqptr(); // local submatrix
        std::vector<DER> localChunks;
        int numChunks = (int)std::sqrt((float)nlayers);
        if(!colsplit) spSeq->Transpose();
        spSeq->ColSplit(numChunks, localChunks);
        if(!colsplit){
            for(int i = 0; i < numChunks; i++) localChunks[i].Transpose();
        }
        MPI_Barrier(commGrid3D->GetWorld());
        if(colsplit){
            if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
                //printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 0, localChunks[0].getnrow(), localChunks[0].getncol(), localChunks[0].getnnz());
                //printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 1, localChunks[1].getnrow(), localChunks[1].getncol(), localChunks[1].getnnz());
                //printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, localChunks[2].getnrow(), localChunks[2].getncol(), localChunks[2].getnnz());
            }
        }
        else{
            if(commGrid3D->myrank == 0 || commGrid3D->myrank == 1 || commGrid3D->myrank == 2){
                printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 0, localChunks[0].getnrow(), localChunks[0].getncol(), localChunks[0].getnnz());
                printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 1, localChunks[1].getnrow(), localChunks[1].getncol(), localChunks[1].getnnz());
                printf("myrank: %d, sendchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, localChunks[2].getnrow(), localChunks[2].getncol(), localChunks[2].getnnz());
            }
        }

        IT datasize;
        NT x = 0.0;
        std::vector<DER> recvChunks;
        SpecialExchangeData(localChunks, commGrid3D->specialWorld, datasize, x, commGrid3D->world3D, recvChunks);
        IT concat_row = 0, concat_col = 0;
        for(int i  = 0; i < numChunks; i++){
            if(colsplit) recvChunks[i].Transpose();
            concat_row = std::max(concat_row, recvChunks[i].getnrow());
            concat_col = concat_col + recvChunks[i].getncol();
        }
        DER * localMatrix = new DER(0, concat_row, concat_col, 0);
        localMatrix->ColConcatenate(recvChunks);
        if(colsplit) localMatrix->Transpose();
        if(colsplit){
            if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
                ////printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 0, recvChunks[0].getnrow(), recvChunks[0].getncol(), recvChunks[0].getnnz());
                ////printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 1, recvChunks[1].getnrow(), recvChunks[1].getncol(), recvChunks[1].getnnz());
                ////printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, recvChunks[2].getnrow(), recvChunks[2].getncol(), recvChunks[2].getnnz());
                //printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, localMatrix->getnrow(), localMatrix->getncol(), localMatrix->getnnz());
            }
        }
        else{
            if(commGrid3D->myrank == 0 || commGrid3D->myrank == 1 || commGrid3D->myrank == 2){
                //printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 0, recvChunks[0].getnrow(), recvChunks[0].getncol(), recvChunks[0].getnnz());
                //printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 1, recvChunks[1].getnrow(), recvChunks[1].getncol(), recvChunks[1].getnnz());
                //printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, recvChunks[2].getnrow(), recvChunks[2].getncol(), recvChunks[2].getnnz());
                printf("myrank: %d, recvchunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, localMatrix->getnrow(), localMatrix->getncol(), localMatrix->getnnz());
            }
        }
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }
   
    
    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnrow() const
    {
        IT totalrows_layer = layermat->getnrow();
        IT totalrows = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalrows_layer, &totalrows, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalrows = totalrows_layer;
        return totalrows;
    }
    
    
    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getncol() const
    {
        IT totalcols_layer = layermat->getncol();
        IT totalcols = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalcols_layer, &totalcols, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalcols = totalcols_layer;
        return totalcols;
    }


    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnnz() const
    {
        IT totalnz_layer = layermat->getnnz();
        IT totalnz = 0;
        if(!colsplit)
        {
            MPI_Allreduce( &totalnz_layer, &totalnz, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        }
        else
            totalnz = totalnz_layer;
        return totalnz;
    }


    template <class IT, class NT, class DER>
    vector<DER> SpecialExchangeData( std::vector<DER> & localChunks, MPI_Comm World, IT& datasize, NT dummy, MPI_Comm secondaryWorld, vector<DER> & recvChunks){
        //int myrank;
        //MPI_Comm_rank(secondaryWorld, &myrank);
        int numChunks = localChunks.size();

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);

        int * sendcnt = new int[numChunks];
        int * sendprfl = new int[numChunks*3];
        int * sdispls = new int[numChunks]();
        int * recvcnt = new int[numChunks];
        int * recvprfl = new int[numChunks*3];
        int * rdispls = new int[numChunks]();

        IT totsend = 0;
        for(IT i=0; i<numChunks; ++i){
            sendprfl[i*3] = localChunks[i].getnnz();
            sendprfl[i*3+1] = localChunks[i].getnrow();
            sendprfl[i*3+2] = localChunks[i].getncol();
            sendcnt[i] = sendprfl[i*3];
            totsend += sendcnt[i];
        }

        MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, World);

        for(IT i = 0; i < numChunks; i++){
            recvcnt[i] = recvprfl[i*3];
        }

        std::partial_sum(sendcnt, sendcnt+numChunks-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+numChunks-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+numChunks, static_cast<IT>(0));

        std::vector< std::tuple<IT,IT,NT> > sendTuples;
        for (int i = 0; i < numChunks; i++){
            for(typename DER::SpColIter colit = localChunks[i].begcol(); colit != localChunks[i].endcol(); ++colit){
                for(typename DER::SpColIter::NzIter nzit = localChunks[i].begnz(colit); nzit != localChunks[i].endnz(colit); ++nzit){
                    NT val = nzit.value();
                    sendTuples.push_back(std::make_tuple(nzit.rowid(), colit.colid(), nzit.value()));
                }
            }
        }
        //if(myrank == 12){
            //int i = 1;
            //printf("[SENDING] Processor: %d, Chunk: %d\n", myrank, i);
            //printf("nnz: %d, mdim: %d, ndim: %d\n",sendprfl[i*3], sendprfl[i*3+1], sendprfl[i*3+2]);
            //printf("first tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(sendTuples[sdispls[i]]), 
                    //get<1>(sendTuples[sdispls[i]]), 
                    //get<2>(sendTuples[sdispls[i]]));
            //printf("last tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    //get<1>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    //get<2>(sendTuples[sdispls[i]+sendcnt[i]-1]));
            ////for(int j = sdispls[i]; j < sdispls[i]+sendcnt[i]; j++){
                ////cout << get<0>(sendTuples[j]) << " " << get<1>(sendTuples[j]) << " " << get<2>(sendTuples[j]) << endl;
            ////}
        //}
        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);

        DeleteAll(sendcnt, sendprfl, sdispls);
        sendTuples.clear();
        sendTuples.shrink_to_fit();

        //std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
        //MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        tuple<IT, IT, NT> ** tempTuples = new tuple<IT, IT, NT>*[numChunks];
        //vector<DER> recvChunks;
        for (int i = 0; i < numChunks; i++){
            tempTuples[i] = new tuple<IT, IT, NT>[recvcnt[i]];
            memcpy(tempTuples[i], recvTuples+rdispls[i], recvcnt[i]*sizeof(tuple<IT, IT, NT>));
            recvChunks.push_back(DER(SpTuples<IT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], tempTuples[i]), false));
        }

        //if(myrank == 6){
            //int i = 2;
            //printf("[RECEIVING] Processor: %d, Chunk: %d\n", myrank, i);
            //printf("nnz: %d, mdim: %d, ndim: %d\n",recvprfl[i*3], recvprfl[i*3+1], recvprfl[i*3+2]);
            //printf("first tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(recvTuples[rdispls[i]]), 
                    //get<1>(recvTuples[rdispls[i]]), 
                    //get<2>(recvTuples[rdispls[i]]));
            //printf("last tuple: < %lld, %lld, %lf >\n", 
                    //get<0>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    //get<1>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    //get<2>(recvTuples[rdispls[i]+recvcnt[i]-1]));
            ////for(int j = rdispls[i]; j < rdispls[i]+recvcnt[i]; j++){
                ////cout << get<0>(recvTuples[j]) << " " << get<1>(recvTuples[j]) << " " << get<2>(recvTuples[j]) << endl;
            ////}
        //}
        DeleteAll(recvcnt, recvprfl, rdispls, recvTuples);
        //for(int i = 0; i < numChunks; i++){
            //delete[] tempTuples[i];
        //}
        //delete[] tempTuples;
        //DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
        //MPI_Type_free(&MPI_tuple);
        //datasize = totrecv;
        //return recvTuples;
        return recvChunks;
    }
}
