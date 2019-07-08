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
        for(int i = 0; i < localChunks.size(); i++){
            int rcvRankInFiber = (colsplit) ? ( ( ( commGrid3D->rankInFiber / sqrtLayer ) * sqrtLayer ) + i ) : ( ( ( commGrid3D->rankInFiber % sqrtLayer ) * sqrtLayer ) + i );
            sendChunks[rcvRankInFiber] = localChunks[i];
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
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }
    
    template <class IT, class NT, class DER>
    SpParMat<IT, NT, DER> SpParMat3D<IT, NT, DER>::Convert2D(){
        DER * spSeq = layermat->seqptr();
        std::vector<DER> localChunks;
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        IT grid3dCols = commGrid3D->gridCols; IT grid3dRows = commGrid3D->gridRows;
        IT grid2dCols = grid3dCols * sqrtLayers; IT grid2dRows = grid3dRows * sqrtLayers;
        IT x = (colsplit) ? layermat->getnrow() : layermat->getncol();
        IT y = (colsplit) ? (x / grid2dRows) : (x / grid2dCols);
        vector<IT> divisions2d;
        if(colsplit){
            for(IT i = 0; i < grid2dRows-1; i++) divisions2d.push_back(y);
            divisions2d.push_back(layermat->getnrow()-(grid2dRows-1)*y);
        }
        else{
            for(IT i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
            divisions2d.push_back(layermat->getncol()-(grid2dCols-1)*y);
        }
        vector<IT> divisions2dChunk;
        IT start = (colsplit) ? ((commGrid3D->rankInLayer / grid3dRows) * sqrtLayers) : ((commGrid3D->rankInLayer % grid3dCols) * sqrtLayers);
        IT end = start + sqrtLayers;
        for(IT i = start; i < end; i++){
            divisions2dChunk.push_back(divisions2d[i]);
        }
        if(colsplit) spSeq->Transpose();
        spSeq->ColSplit(divisions2dChunk, localChunks);
        if(colsplit){
            for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
        }
        std::vector<DER> sendChunks(nlayers);
        for(int i = 0; i < sendChunks.size(); i++){
            sendChunks[i] = DER(0, 0, 0, 0);
        }
        for(int i = 0; i < localChunks.size(); i++){
            int rcvRankInFiber = (colsplit) ? ( ( ( commGrid3D->rankInFiber / sqrtLayers ) * sqrtLayers ) + i ) : ( ( ( commGrid3D->rankInFiber % sqrtLayers ) * sqrtLayers ) + i );
            sendChunks[rcvRankInFiber] = localChunks[i];
        }
        IT datasize; NT z=1.0;
        std::vector<DER> recvChunks;
        SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, z, commGrid3D->world3D, recvChunks);

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
        SpParMat<IT, NT, DER>* Blayermat = B.layermat;
        MPI_Barrier(MPI_COMM_WORLD);
        typedef PlusTimesSRing<NT, NT> PTFF;
        SpParMat<IT, NT, DER> C3D_layer = Mult_AnXBn_DoubleBuff<PTFF, NT, DER>(*layermat, *Blayermat);
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        DER* C3D_localMat = C3D_layer.seqptr();
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
        
        vector<DER> rcvChunks;
        IT datasize; NT dummy = 0.0;
        SpecialExchangeData( sendChunks, commGrid3D->fiberWorld, datasize, dummy, commGrid3D->fiberWorld, rcvChunks);
        DER * localMatrix = new DER(0, rcvChunks[0].getnrow(), rcvChunks[0].getncol(), 0);
        for(int i = 0; i < rcvChunks.size(); i++) *localMatrix += rcvChunks[i];
        std::shared_ptr<CommGrid3D> grid3d;
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), nlayers, 0, 0, true, true));
        SpParMat3D<IT, NT, DER> C3D(localMatrix, grid3d);
        return C3D;
    }

    template <class IT, class NT, class DER>
    template <typename SR>
    SpParMat3D<IT, NT, DER> SpParMat3D<IT, NT, DER>::MemEfficientSpGEMM3D(SpParMat3D<IT, NT, DER> & B, 
            int phases, NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct, int kselectVersion, double perProcessMemory){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        if(getncol() != B.getnrow()){
            std::ostringstream outs;
            outs << "Can not multiply, dimensions does not match"<< std::endl;
            outs << getncol() << " != " << B.getnrow() << std::endl;
            SpParHelper::Print(outs.str());
            MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        }
        if(phases < 1 || phases >= getncol()){
            SpParHelper::Print("MemEfficientSpGEMM: The value of phases is too small or large. Resetting to 1.\n");
            phases = 1;
        }
        else{
            int p = 16;
            int64_t perNNZMem_in = sizeof(IT)*2 + sizeof(NT);
            int64_t perNNZMem_out = sizeof(IT)*2 + sizeof(NT);

            int64_t lannz = layermat->getlocalnnz();
            int64_t gannz;
            MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, commGrid3D->GetWorld());
            int64_t inputMem = gannz * perNNZMem_in * 4;

            int64_t asquareNNZ = EstPerProcessNnzSUMMA(*layermat, *(B.layermat));
            int64_t gasquareNNZ;
            MPI_Allreduce(&asquareNNZ, &gasquareNNZ, 1, MPIType<int64_t>(), MPI_MAX, commGrid3D->GetFiberWorld());
            int64_t asquareMem = gasquareNNZ * perNNZMem_out * 2;

            int64_t d = ceil( (gasquareNNZ * sqrt(p))/ B.layermat->getlocalcols() );
            int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
            int64_t kselectmem = B.layermat->getlocalcols() * k * 8 * 3;

            // estimate output memory
            int64_t outputNNZ = (B.layermat->getlocalcols() * k)/sqrt(p);
            int64_t outputMem = outputNNZ * perNNZMem_in * 2;

            //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
            double remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
            if(remainingMem > 0){
                // Omitting phase calculation for now. Can be uncommented later again.
                //phases = 1 + ceil((asquareMem+kselectmem) / remainingMem);
            }
            else{
                if(myrank == 0){
                    cout << "Not enough memory available" << endl;
                }
            }
        }

        vector<DER> PiecesOfB;
        DER CopyB = *(B.layermat->seqptr());
        CopyB.ColSplit(phases, PiecesOfB);

        int stages = commGrid3D->gridRows;
        IT ** ARecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
        IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);

        SpParHelper::GetSetSizes( *(layermat->seqptr()), ARecvSizes, layermat->getcommgrid()->GetRowWorld());

        DER * ARecv;
        DER * BRecv;

        int Aself = layermat->getcommgrid()->GetRankInProcRow();
        int Bself = (B.layermat)->getcommgrid()->GetRankInProcCol();
        
        /*
         *  Calculate, accross fibers, which process should get how many columns 
         *  after redistribution
         * */
        int sqrtLayers = (int)std::sqrt((float)nlayers);
        IT grid3dCols = commGrid3D->gridCols;
        IT grid2dCols = grid3dCols * sqrtLayers;
        IT x = (B.layermat)->getncol();
        IT y = x / grid2dCols;
        vector<IT> divisions2d;
        for(IT i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
        divisions2d.push_back((B.layermat)->getncol()-(grid2dCols-1)*y);
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

        DER * localLayerResultant = new DER(0, layermat->seqptr()->getnrow(), divisions3d[commGrid3D->rankInFiber], 0);
        SpParMat<IT, NT, DER> layerResultant(localLayerResultant, commGrid3D->layerWorld);

        for(int p = 0; p < phases; p++){
            SpParHelper::GetSetSizes(PiecesOfB[p], BRecvSizes, (B.layermat)->getcommgrid()->GetColWorld());
            vector< SpTuples<IT, NT> *> tomerge;
            for(int i = 0; i < stages; i++){
                vector<IT> ess;
                if(i == Aself) ARecv = layermat->seqptr();
                else{
                    ess.resize(DER::esscount);
                    for(int j = 0; j < DER::esscount; j++)
                        ess[j] = ARecvSizes[j][i];
                    ARecv = new DER();
                }
                SpParHelper::BCastMatrix(layermat->getcommgrid()->GetRowWorld(), *ARecv, ess, i);

                ess.clear();
                if(i == Bself) BRecv = &(PiecesOfB[p]);
                else{
                    ess.resize(DER::esscount);
                    for(int j = 0; j < DER::esscount; j++)
                        ess[j] = BRecvSizes[j][i];
                    BRecv = new DER();
                }
                SpParHelper::BCastMatrix((B.layermat)->getcommgrid()->GetColWorld(), *BRecv, ess, i);
                
                MPI_Barrier(MPI_COMM_WORLD);
                SpTuples<IT, NT> * C_cont = LocalSpGEMM<SR, NT>(*ARecv, *BRecv, i!=Aself, i!=Bself);
                if(!C_cont->isZero())
                    tomerge.push_back(C_cont);
                else
                    delete C_cont;
            }

            /*
             *  Merge all the resultants for each stage of SUMMA operation.
             *  Basically same as summing all the elements up as C1 = A0B1+A1B3
             * */
            SpTuples<IT, NT> * OnePieceOfC_tuples = MultiwayMerge<SR>(tomerge, layermat->seqptr()->getnrow(), PiecesOfB[p].getncol(), true);

            /*
             *  Create a local matrix witht the tuples got for this phase of multiplication
             * */
            DER OnePieceOfC(*OnePieceOfC_tuples, false);

            /*
             *  Pad OnePieceOfC with empty matrices on left and right to match 
             *  the dimension that it was supposed to be in the case of multiplication without phases
             * */
            int ncol_left = 0, ncol_right = 0, ncol_total = 0;
            for(int j = 0; j < p; j++) ncol_left += PiecesOfB[j].getncol();
            ncol_total += ncol_left;
            ncol_total += OnePieceOfC.getncol();
            for(int j = p+1; j < phases; j++) ncol_right += PiecesOfB[j].getncol();
            ncol_total += ncol_right;
            vector<DER>chunksWithPadding;
            chunksWithPadding.push_back(DER(0, OnePieceOfC.getnrow(), ncol_left, 0));
            chunksWithPadding.push_back(OnePieceOfC);
            chunksWithPadding.push_back(DER(0, OnePieceOfC.getnrow(), ncol_right, 0));
            DER * paddedMatrix = new DER(0, OnePieceOfC.getnrow(), ncol_total, 0);
            paddedMatrix->ColConcatenate(chunksWithPadding);

            /*
             *  Now column split the padded matrix for 3D reduction and do the it
             * */
            vector<DER> sendChunks;
            paddedMatrix->ColSplit(divisions3d, sendChunks);
            vector<DER> rcvChunks;
            IT datasize; NT dummy = 0.0;
            SpecialExchangeData( sendChunks, commGrid3D->fiberWorld, datasize, dummy, commGrid3D->fiberWorld, rcvChunks);
            DER * phaseResultant = new DER(0, rcvChunks[0].getnrow(), rcvChunks[0].getncol(), 0);
            for(int i = 0; i < rcvChunks.size(); i++) *phaseResultant += rcvChunks[i];
            SpParMat<IT, NT, DER> phaseResultantLayer(phaseResultant, commGrid3D->layerWorld);
            MCLPruneRecoverySelect(phaseResultantLayer, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);
            layerResultant += phaseResultantLayer;
        }
        SpHelper::deallocate2D(ARecvSizes, DER::esscount);
        SpHelper::deallocate2D(BRecvSizes, DER::esscount);

        std::shared_ptr<CommGrid3D> grid3d;
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), nlayers, 0, 0, true, true));
        DER * localResultant = new DER(*localLayerResultant);
        SpParMat3D<IT, NT, DER> C3D(localResultant, grid3d);
        //printf("myrank: %d, row: %d, col: %d, nnz: %d\n", myrank, C3D.seqptr()->getnrow(), C3D.seqptr()->getncol(), C3D.seqptr()->getnnz());
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
        if(colsplit)
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

        MPI_Barrier(MPI_COMM_WORLD);
        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
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
