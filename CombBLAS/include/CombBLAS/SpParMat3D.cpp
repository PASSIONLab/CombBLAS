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
#include <string>

namespace combblas
{
    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (DER * localMatrix, std::shared_ptr<CommGrid3D> grid3d): commGrid3D(grid3d)
    {
        assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
        commGrid3D = grid3d;
        colsplit = true;
        MPI_Comm_size(commGrid3D->fiberWorld, &nlayers);
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }

    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool csplit, bool special): nlayers(nlayers), colsplit(csplit)
    {
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
            for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
        }

        // Some necessary processing before exchanging data
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        std::vector<DER> sendChunks(nlayers);
        for(int i = 0; i < sendChunks.size(); i++){
            sendChunks[i] = DER(0, 0, 0, 0);
        }
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber / sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        else{
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber % sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        MPI_Barrier(commGrid3D->GetWorld());

        IT datasize; NT x = 0.0;
        std::vector<DER> recvChunks;

        SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, commGrid3D->world3D, recvChunks);
        IT concat_row = 0, concat_col = 0;
        for(int i  = 0; i < recvChunks.size(); i++){
            if(colsplit) recvChunks[i].Transpose();
            concat_row = std::max(concat_row, recvChunks[i].getnrow());
            concat_col = concat_col + recvChunks[i].getncol();
        }
        DER * localMatrix = new DER(0, concat_row, concat_col, 0);
        localMatrix->ColConcatenate(recvChunks);
        if(colsplit) localMatrix->Transpose();
        //printf("%d : %d x %d\n", commGrid3D->myrank, localMatrix->getnrow(), localMatrix->getncol());
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }
    
    template <class IT, class NT, class DER>
    SpParMat<IT, NT, DER> SpParMat3D<IT, NT, DER>::Convert2D(){
        DER * spSeq = layermat->seqptr();
        std::vector<DER> localChunks;
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        if(colsplit) spSeq->Transpose();
        /**/
        vector<DER> tempChunks_1;
        vector<DER> tempChunks_2;
        spSeq->ColSplit(sqrtLayers * commGrid3D->gridCols, tempChunks_1);
        for(int i = 0; i < sqrtLayers; i++){
            std::vector<DER> tempChunks_3;
            IT concat_row = 0, concat_col = 0;
            for(int j = 0; j < commGrid3D->gridCols; j++){
                int k = i*commGrid3D->gridCols+j;
                concat_row = std::max(concat_row, tempChunks_1[k].getnrow());
                concat_col = concat_col + tempChunks_1[k].getncol();
                tempChunks_3.push_back(tempChunks_1[k]);
            }
            localChunks.push_back(DER(0, concat_row, concat_col, 0));
            localChunks[i].ColConcatenate(tempChunks_3);
        }
        /**/
        //spSeq->ColSplit(sqrtLayers, localChunks);
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
        }
        std::vector<DER> sendChunks(nlayers);
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        for(int i = 0; i < sendChunks.size(); i++){
            sendChunks[i] = DER(0, 0, 0, 0);
        }
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++){
                int rcvRankInFiber = ( ( commGrid3D->rankInFiber / sqrtLayer ) * sqrtLayer ) + i;
                sendChunks[rcvRankInFiber] = localChunks[i];
            }
        }
        IT datasize; NT x=1.0;
        std::vector<DER> recvChunks;
        SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, commGrid3D->world3D, recvChunks);

        IT concat_row = 0, concat_col = 0;
        for(int i  = 0; i < recvChunks.size(); i++){
            if(!colsplit) recvChunks[i].Transpose();
            concat_row = std::max(concat_row, recvChunks[i].getnrow());
            concat_col = concat_col + recvChunks[i].getncol();
        }
        DER * localMatrix = new DER(0, concat_row, concat_col, 0);
        localMatrix->ColConcatenate(recvChunks);
        if(!colsplit) localMatrix->Transpose();
        std::shared_ptr<CommGrid> grid2d;
        grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
        SpParMat<IT, NT, DER> mat2D(localMatrix, grid2d);
        return mat2D;
    }
    
    template <class IT, class NT, class DER>
    template <typename SR>
    SpParMat3D<IT, NT, DER> SpParMat3D< IT,NT,DER >::mult(SpParMat3D<IT, NT, DER> & B){
    //void SpParMat3D< IT,NT,DER >::mult(SpParMat3D<IT, NT, DER> & B){
        //printf("%d = %d\n", A.getncol(), B.getnrow());
        SpParMat<IT, NT, DER>* Blayermat = B.layermat;
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("%d: %d = %d\n", commGrid3D->myrank, layermat->getncol(), Blayermat->getnrow());
        //printf("myrank %d\tA.rankInFiber %d\tA.rankInLayer %d\tB.rankInFiber %d\tB.rankInLayer %d\t:\t[%d x %d] X [%d x %d]\n", 
                //commGrid3D->myrank, commGrid3D->rankInFiber, commGrid3D->rankInLayer, B.commGrid3D->rankInFiber, B.commGrid3D->rankInLayer,
                //layermat->getnrow(), layermat->getncol(), Blayermat->getnrow(), Blayermat->getncol());
        //printf("myrank %d\tA.rankInFiber %d\tA.rankInLayer %d\tB.rankInFiber %d\tB.rankInLayer %d\t:\t[%d x %d] X [%d x %d]\n", 
                //commGrid3D->myrank, commGrid3D->rankInFiber, commGrid3D->rankInLayer, B.commGrid3D->rankInFiber, B.commGrid3D->rankInLayer,
                //layermat->seqptr()->getnrow(), layermat->seqptr()->getncol(), Blayermat->seqptr()->getnrow(), Blayermat->seqptr()->getncol());
        typedef PlusTimesSRing<NT, NT> PTFF;
        SpParMat<IT, NT, DER> C3D_layer = Mult_AnXBn_DoubleBuff<PTFF, NT, DER>(*layermat, *Blayermat);
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        DER* C3D_localMat = C3D_layer.seqptr();
        //printf("%d C3D_layer: %d x %d\n", commGrid3D->myrank, C3D_layer.getnrow(), C3D_layer.getncol());
        //printf("%d C3D_localMat: %d x %d\n", commGrid3D->myrank, C3D_localMat->getnrow(), C3D_localMat->getncol());
        IT grid3dCols = commGrid3D->gridCols;
        IT grid2dCols = grid3dCols * sqrtLayers;
        IT x = C3D_layer.getncol();
        IT y = x / grid2dCols;
        vector<IT> divisions2d;
        for(IT i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
        divisions2d.push_back(C3D_layer.getncol()-(grid2dCols-1)*y);
        vector<IT> divisions2dChunk;
        IT start = (commGrid3D->rankInLayer % grid3dCols) * sqrtLayers;
        IT end = start + sqrtLayers;
        for(IT i = start; i < end; i++){
            divisions2dChunk.push_back(divisions2d[i]);
        }
        vector<IT> divisions3d;
        for(int i = 0; i < divisions2dChunk.size(); i++){
            IT y = divisions2dChunk[i]/sqrtLayers;
            for(int j = 0; j < sqrtLayers-1; j++) divisions3d.push_back(y);
            divisions3d.push_back(divisions2dChunk[i]-(sqrtLayers-1)*y);
        }
        vector<DER> sendChunks;
        C3D_localMat->ColSplit(divisions3d, sendChunks);
        
        //vector<DER> tempChunks_1;
        //vector<DER> tempChunks_2;
        //C3D_localMat->ColSplit(sqrtLayers * commGrid3D->gridCols, tempChunks_1);
        //for(int i = 0; i < sqrtLayers; i++){
            //std::vector<DER> tempChunks_3;
            //IT concat_row = 0, concat_col = 0;
            //for(int j = 0; j < commGrid3D->gridCols; j++){
                //int k = i*commGrid3D->gridCols+j;
                //concat_row = std::max(concat_row, tempChunks_1[k].getnrow());
                //concat_col = concat_col + tempChunks_1[k].getncol();
                //tempChunks_3.push_back(tempChunks_1[k]);
            //}
            //tempChunks_2.push_back(DER(0, concat_row, concat_col, 0));
            //tempChunks_2[i].ColConcatenate(tempChunks_3);
        //}
        //vector<DER> sendChunks;
        //for(int i = 0; i < tempChunks_2.size(); i++){
            //tempChunks_2[i].ColSplit(sqrtLayers, sendChunks);
        //}
        vector<DER> rcvChunks;
        IT datasize; NT dummy = 0.0;
        SpecialExchangeData( sendChunks, commGrid3D->fiberWorld, datasize, dummy, commGrid3D->fiberWorld, rcvChunks);
        DER * localMatrix = new DER(0, rcvChunks[0].getnrow(), rcvChunks[0].getncol(), 0);
        for(int i = 0; i < rcvChunks.size(); i++) *localMatrix += rcvChunks[i];
        std::shared_ptr<CommGrid3D> grid3d;
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), nlayers, 0, 0, true, true));
        SpParMat3D<IT, NT, DER> C3D(localMatrix, grid3d);
        //if(commGrid3D->myrank == 35){
            //for(int i = 0; i < rcvChunks.size(); i++){
                //printf("[%dx%d] ", rcvChunks[i].getnrow(), rcvChunks[i].getncol());
            //}
            //printf("\n");
        //}
        return C3D;
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
        MPI_Allreduce( &totalnz_layer, &totalnz, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        return totalnz;
    }


    template <class IT, class NT, class DER>
    vector<DER> SpecialExchangeData( std::vector<DER> & sendChunks, MPI_Comm World, IT& datasize, NT dummy, MPI_Comm secondaryWorld, vector<DER> & recvChunks){
        int numChunks = sendChunks.size();
        int myrank;
        MPI_Comm_rank(secondaryWorld, &myrank);

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
            sendprfl[i*3] = sendChunks[i].getnnz();
            sendprfl[i*3+1] = sendChunks[i].getnrow();
            sendprfl[i*3+2] = sendChunks[i].getncol();
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
        for(int i = 0; i < numChunks; i++){
            for(typename DER::SpColIter colit = sendChunks[i].begcol(); colit != sendChunks[i].endcol(); ++colit){
                for(typename DER::SpColIter::NzIter nzit = sendChunks[i].begnz(colit); nzit != sendChunks[i].endnz(colit); ++nzit){
                    NT val = nzit.value();
                    sendTuples.push_back(std::make_tuple(nzit.rowid(), colit.colid(), nzit.value()));
                }
            }
        }

        //if(dummy == 1.0) printf("totrecv: %d\n", totrecv);
        MPI_Barrier(MPI_COMM_WORLD);
        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
        //if(dummy == 1.0) printf("Here\n");
        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);

        DeleteAll(sendcnt, sendprfl, sdispls);
        sendTuples.clear();
        sendTuples.shrink_to_fit();

        tuple<IT, IT, NT> ** tempTuples = new tuple<IT, IT, NT>*[numChunks];
        for (int i = 0; i < numChunks; i++){
            tempTuples[i] = new tuple<IT, IT, NT>[recvcnt[i]];
            memcpy(tempTuples[i], recvTuples+rdispls[i], recvcnt[i]*sizeof(tuple<IT, IT, NT>));
            recvChunks.push_back(DER(SpTuples<IT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], tempTuples[i]), false));
        }
        
        // Free all memory except tempTuples; Because that memory is holding data of newly created local matrices after receiving.
        DeleteAll(recvcnt, recvprfl, rdispls, recvTuples); 
        MPI_Type_free(&MPI_tuple);
        return recvChunks;
    }
}
