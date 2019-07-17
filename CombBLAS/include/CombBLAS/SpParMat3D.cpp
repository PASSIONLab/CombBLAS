/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
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
    SpParMat3D< IT,NT,DER >::SpParMat3D (DER * localMatrix, std::shared_ptr<CommGrid3D> grid3d, bool colsplit, bool special = false): commGrid3D(grid3d), colsplit(colsplit), special(special){
        assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
        MPI_Comm_size(commGrid3D->fiberWorld, &nlayers);
        layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
    }

    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool colsplit, bool special = false): nlayers(nlayers), colsplit(colsplit), special(special){
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        int nprocs = commGrid2D->GetSize();
        commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, special));
        if(special){
            DER* spSeq = A2D.seqptr(); // local submatrix
            std::vector<DER> localChunks;
            int numChunks = (int)std::sqrt((float)commGrid3D->GetGridLayers());
            if(!colsplit) spSeq->Transpose();
            spSeq->ColSplit(numChunks, localChunks);
            if(!colsplit){
                for(int i = 0; i < localChunks.size(); i++) localChunks[i].Transpose();
            }

            // Some necessary processing before exchanging data
            int sqrtLayer = (int)std::sqrt((float)commGrid3D->GetGridLayers());
            std::vector<DER> sendChunks(commGrid3D->GetGridLayers());
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
        else {
            IT nrows = A2D.getnrow();
            IT ncols = A2D.getncol();
            int pr2d = commGrid2D->GetGridRows();
            int pc2d = commGrid2D->GetGridCols();
            int rowrank2d = commGrid2D->GetRankInProcRow();
            int colrank2d = commGrid2D->GetRankInProcCol();
            IT m_perproc2d = nrows / pr2d;
            IT n_perproc2d = ncols / pc2d;
            DER* spSeq = A2D.seqptr(); // local submatrix
            IT localRowStart2d = colrank2d * m_perproc2d; // first row in this process
            IT localColStart2d = rowrank2d * n_perproc2d; // first col in this process


            LIT lrow3d, lcol3d;
            std::vector<IT> tsendcnt(nprocs,0);
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
            {
                IT gcol = colit.colid() + localColStart2d;
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                {
                    IT grow = nzit.rowid() + localRowStart2d;
                    int owner = Owner(nrows, ncols, grow, gcol, lrow3d, lcol3d); //3D owner
                    tsendcnt[owner]++;
                }
            }

            std::vector<std::vector<std::tuple<IT,IT, NT>>> sendTuples (nprocs);
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
            {
                IT gcol = colit.colid() + localColStart2d;
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                {
                    IT grow = nzit.rowid() + localRowStart2d;
                    NT val = nzit.value();
                    int owner = Owner(nrows, ncols, grow, gcol, lrow3d, lcol3d); //3D owner
                    sendTuples[owner].push_back(std::make_tuple(lrow3d, lcol3d, val));
                }
            }

            IT datasize;
            std::tuple<IT,IT,NT>* recvTuples = ExchangeData(sendTuples, commGrid2D->GetWorld(), datasize);

            IT mdim, ndim;
            LocalDim(nrows, ncols, mdim, ndim);
            //cout << mdim << " " << ndim << " "<< datasize << endl;
            SpTuples<IT, NT>spTuples3d(datasize, mdim, ndim, recvTuples);

            DER * localm3d = new DER(spTuples3d, false);
            std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer;

            layermat = new SpParMat<IT, NT, DER>(localm3d, commGridLayer);
        }
    }

    template <class IT, class NT,class DER>
    template <typename LIT>
    int SpParMat3D<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const {
        // first map to Layer 0
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer; // CommGrid for my layer
        int procrows = commGridLayer->GetGridRows();
        int proccols = commGridLayer->GetGridCols();
        int nlayers = commGrid3D->GetGridLayers();
        
        IT m_perproc_L0 = total_m / procrows;
        IT n_perproc_L0 = total_n / proccols;
        
        int procrow_L0; // within a layer
        if(m_perproc_L0 != 0){
            procrow_L0 = std::min(static_cast<int>(grow / m_perproc_L0), procrows-1);
        }
        else{
            // all owned by the last processor row
            procrow_L0 = procrows -1;
        }
        int proccol_L0;
        if(n_perproc_L0 != 0){
            proccol_L0 = std::min(static_cast<int>(gcol / n_perproc_L0), proccols-1);
        }
        else{
            proccol_L0 = proccols-1;
        }
        
        IT lrow_L0 = grow - (procrow_L0 * m_perproc_L0);
        IT lcol_L0 = gcol - (proccol_L0 * n_perproc_L0);
        int layer;
        // next, split and scatter
        if(colsplit){
            IT n_perproc;

            if(proccol_L0 < commGrid3D->gridCols-1)
                n_perproc = n_perproc_L0 / nlayers;
            else
                n_perproc = (total_n - (n_perproc_L0 * proccol_L0)) / nlayers;

            if(n_perproc != 0)
                layer = std::min(static_cast<int>(lcol_L0 / n_perproc), nlayers-1);
            else
                layer = nlayers-1;
            
            lrow = lrow_L0;
            lcol = lcol_L0 - (layer * n_perproc);
        }
        else{
            IT m_perproc;

            if(procrow_L0 < commGrid3D->gridRows-1)
                m_perproc = m_perproc_L0 / nlayers;
            else
                m_perproc = (total_m - (m_perproc_L0 * procrow_L0)) / nlayers;

            if(m_perproc != 0)
                layer = std::min(static_cast<int>(lrow_L0 / m_perproc), nlayers-1);
            else
                layer = nlayers-1;
            
            lcol = lcol_L0;
            lrow = lrow_L0 - (layer * m_perproc);
        }
        int proccol_layer = proccol_L0;
        int procrow_layer = procrow_L0;
        return commGrid3D->GetRank(layer, procrow_layer, proccol_layer);
    }

    template <class IT, class NT,class DER>
    void SpParMat3D<IT,NT,DER>::LocalDim(IT total_m, IT total_n, IT &localm, IT& localn) const
    {
        // first map to Layer 0 and then split
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer; // CommGrid for my layer
        int procrows = commGridLayer->GetGridRows();
        int proccols = commGridLayer->GetGridCols();
        int nlayers = commGrid3D->GetGridLayers();

        IT localm_L0 = total_m / procrows;
        IT localn_L0 = total_n / proccols;

        if(commGridLayer->GetRankInProcRow() == commGrid3D->gridCols-1)
        {
            localn_L0 = (total_n - localn_L0*(commGrid3D->gridCols-1));
        }
        if(commGridLayer->GetRankInProcCol() == commGrid3D->gridRows-1)
        {
            localm_L0 = (total_m - localm_L0 * (commGrid3D->gridRows-1));
        }
        if(colsplit)
        {
            localn = localn_L0/nlayers;
            if(commGrid3D->rankInFiber == (commGrid3D->gridLayers-1))
                localn = localn_L0 - localn * (commGrid3D->gridLayers-1);
            localm = localm_L0;
        }
        else
        {
            localm = localm_L0/nlayers;
            if(commGrid3D->rankInFiber == (commGrid3D->gridLayers-1))
                localm = localm_L0 - localm * (commGrid3D->gridLayers-1);
            localn = localn_L0;
        }
    }
    
    template <class IT, class NT, class DER>
    SpParMat<IT, NT, DER> SpParMat3D<IT, NT, DER>::Convert2D(){
        if(special){
            DER * spSeq = layermat->seqptr();
            std::vector<DER> localChunks;
            int sqrtLayers = (int)std::sqrt((float)commGrid3D->GetGridLayers());
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
            std::vector<DER> sendChunks(commGrid3D->GetGridLayers());
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
        else{
            int nprocs = commGrid3D->GetSize();
            int nlayers = commGrid3D->GetGridLayers();
            int proccols = commGrid3D->GetGridCols();
            int procrows = commGrid3D->GetGridRows();
            IT total_m = getnrow();
            IT total_n = getncol();
            IT m_perproc_L0 = total_m / procrows;
            IT n_perproc_L0 = total_n / proccols;
            IT m_perproc = colsplit ? m_perproc_L0 : (m_perproc_L0 / nlayers);
            IT n_perproc = colsplit ? (n_perproc_L0 / nlayers) : n_perproc_L0;

            std::shared_ptr<CommGrid> grid2d;
            grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
            SpParMat<IT, NT, DER> A2D (grid2d);

            std::vector< std::vector < std::tuple<IT,IT,NT> > > data(nprocs);
            DER* spSeq = layermat->seqptr(); // local submatrix
            IT locsize = 0;
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit){
                IT lcol = colit.colid();
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit){
                    IT lrow = nzit.rowid();
                    NT val = nzit.value();
                    IT lrow_L0 = colsplit ? lrow : ((commGrid3D->rankInLayer * m_perproc) + lrow); 
                    IT lcol_L0 = colsplit ? ((commGrid3D->rankInLayer * n_perproc) + lcol) : lcol;
                    IT grow = (commGrid3D->commGridLayer->GetRankInProcCol() * m_perproc_L0) + lrow_L0;
                    IT gcol = (commGrid3D->commGridLayer->GetRankInProcRow() * n_perproc_L0) + lcol_L0;
                    
                    IT lrow2d, lcol2d;
                    int owner = A2D.Owner(total_m, total_n, grow, gcol, lrow2d, lcol2d);
                    data[owner].push_back(std::make_tuple(lrow2d,lcol2d,val));
                    locsize++;
                }
            }
            //printf("myrank %d: %d\n", commGrid3D->myrank, locsize);
            A2D.SparseCommon(data, locsize, total_m, total_n, maximum<NT>());
            
            return A2D;
        }
    }
    
    template <class IT, class NT, class DER>
    template <typename SR>
    SpParMat3D<IT, NT, DER> SpParMat3D< IT,NT,DER >::mult(SpParMat3D<IT, NT, DER> & B){
        SpParMat<IT, NT, DER>* Blayermat = B.layermat;
        MPI_Barrier(MPI_COMM_WORLD);
        typedef PlusTimesSRing<NT, NT> PTFF;
        SpParMat<IT, NT, DER> C3D_layer = Mult_AnXBn_DoubleBuff<PTFF, NT, DER>(*layermat, *Blayermat);
        int sqrtLayers = (int)std::sqrt((float)commGrid3D->GetGridLayers());
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
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), commGrid3D->GetGridLayers(), 0, 0, true));
        SpParMat3D<IT, NT, DER> C3D(localMatrix, grid3d, isColSplit(), isSpecial());
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
            // 3D Specific
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
        
        /*
         *  Calculate, accross fibers, which process should get how many columns 
         *  after redistribution
         * */
        vector<IT> divisions3d;
        if(special){
            vector<IT> divisions2d;
            int sqrtLayers = (int)std::sqrt((float)commGrid3D->GetGridLayers());
            IT grid3dCols = commGrid3D->gridCols;
            IT grid2dCols = grid3dCols * sqrtLayers;
            IT x = (B.layermat)->getncol();
            IT y = x / grid2dCols;
            for(IT i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
            divisions2d.push_back((B.layermat)->getncol()-(grid2dCols-1)*y);
            vector<IT> divisions2dChunk;
            IT start = (commGrid3D->rankInLayer % grid3dCols) * sqrtLayers;
            IT end = start + sqrtLayers;
            for(int i = start; i < end; i++){
                divisions2dChunk.push_back(divisions2d[i]);
            }
            for(int i = 0; i < divisions2dChunk.size(); i++){
                IT y = divisions2dChunk[i]/sqrtLayers;
                for(int j = 0; j < sqrtLayers-1; j++) divisions3d.push_back(y);
                divisions3d.push_back(divisions2dChunk[i]-(sqrtLayers-1)*y);
            }
        }
        else{
            // Partitioning for 3D can be achieved by dividing local columns in #layers equal partitions
            IT x = ((B.layermat)->seqptr())->getncol();
            int nlayers = commGrid3D->GetGridLayers();
            IT y = x / nlayers;
            for(int i = 0; i < nlayers-1; i++) divisions3d.push_back(y);
            divisions3d.push_back(x-(nlayers-1)*y);
        }

        DER * localLayerResultant = new DER(0, layermat->seqptr()->getnrow(), divisions3d[commGrid3D->rankInFiber], 0);
        SpParMat<IT, NT, DER> layerResultant(localLayerResultant, commGrid3D->layerWorld);

        for(int p = 0; p < phases; p++){
            DER * OnePieceOfB = new DER(PiecesOfB[p]);
            SpParMat<IT, NT, DER> OnePieceOfBLayer(OnePieceOfB, commGrid3D->layerWorld);
            SpParMat<IT, NT, DER> OnePieceOfCLayer = Mult_AnXBn_Synch<SR, NT, DER>(*(layermat), OnePieceOfBLayer);
            DER * OnePieceOfC = OnePieceOfCLayer.seqptr();

            /*
             *  Pad OnePieceOfC with empty matrices on left and right to match 
             *  the dimension that it was supposed to be in the case of multiplication without phases
             * */
            int ncol_left = 0, ncol_right = 0, ncol_total = 0;
            for(int j = 0; j < p; j++) ncol_left += PiecesOfB[j].getncol();
            ncol_total += ncol_left;
            ncol_total += OnePieceOfC->getncol();
            for(int j = p+1; j < phases; j++) ncol_right += PiecesOfB[j].getncol();
            ncol_total += ncol_right;
            vector<DER>chunksWithPadding;
            chunksWithPadding.push_back(DER(0, OnePieceOfC->getnrow(), ncol_left, 0));
            chunksWithPadding.push_back(*OnePieceOfC);
            chunksWithPadding.push_back(DER(0, OnePieceOfC->getnrow(), ncol_right, 0));
            DER * paddedMatrix = new DER(0, OnePieceOfC->getnrow(), ncol_total, 0);
            paddedMatrix->ColConcatenate(chunksWithPadding);

            /*
             *  Now column split the padded matrix for 3D reduction and do it
             * */
            vector<DER> sendChunks;
            if(special) paddedMatrix->ColSplit(divisions3d, sendChunks);
            else paddedMatrix->ColSplit(commGrid3D->GetGridLayers(), sendChunks);
            vector<DER> rcvChunks;
            IT datasize; NT dummy = 0.0;
            SpecialExchangeData( sendChunks, commGrid3D->fiberWorld, datasize, dummy, commGrid3D->fiberWorld, rcvChunks);
            DER * phaseResultant = new DER(0, rcvChunks[0].getnrow(), rcvChunks[0].getncol(), 0);
            for(int i = 0; i < rcvChunks.size(); i++) *phaseResultant += rcvChunks[i];
            SpParMat<IT, NT, DER> phaseResultantLayer(phaseResultant, commGrid3D->layerWorld);

            MCLPruneRecoverySelect(phaseResultantLayer, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);

            layerResultant += phaseResultantLayer;
        }

        std::shared_ptr<CommGrid3D> grid3d;
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), commGrid3D->GetGridLayers(), 0, 0, true));
        DER * localResultant = new DER(*localLayerResultant);
        SpParMat3D<IT, NT, DER> C3D(localResultant, grid3d, isColSplit(), isSpecial());
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

    template <class IT, class NT>
    std::tuple<IT,IT,NT>*  ExchangeData(std::vector<std::vector<std::tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World, IT& datasize)
    {

        /* Create/allocate variables for vector assignment */
        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);

        int nprocs;
        MPI_Comm_size(World, &nprocs);

        int * sendcnt = new int[nprocs];
        int * recvcnt = new int[nprocs];
        int * sdispls = new int[nprocs]();
        int * rdispls = new int[nprocs]();

        // Set the newly found vector entries
        IT totsend = 0;
        for(IT i=0; i<nprocs; ++i)
        {
            sendcnt[i] = tempTuples[i].size();
            totsend += tempTuples[i].size();
        }

        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);

        std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));

        std::vector< std::tuple<IT,IT,NT> > sendTuples(totsend);
        for(int i=0; i<nprocs; ++i)
        {
            copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
            std::vector< std::tuple<IT,IT,NT> >().swap(tempTuples[i]);    // clear memory
        }

        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
        //std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);
        DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
        MPI_Type_free(&MPI_tuple);
        datasize = totrecv;
        return recvTuples;
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
