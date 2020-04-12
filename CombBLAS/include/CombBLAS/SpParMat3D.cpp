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
#include "CombBLAS/CombBLAS.h"
#include <unistd.h>

namespace combblas
{
    template <class IT, class NT, class DER>
    SpParMat3D<IT, NT, DER>::~SpParMat3D(){
        // No need to delete layermat because it is a smart pointer
        //delete layermat;
    }

    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (DER * localMatrix, std::shared_ptr<CommGrid3D> grid3d, bool colsplit, bool special = false): commGrid3D(grid3d), colsplit(colsplit), special(special){
        assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
        MPI_Comm_size(commGrid3D->fiberWorld, &nlayers);
        layermat.reset(new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld));
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

            SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, recvChunks);
            typename DER::LocalIT concat_row = 0, concat_col = 0;
            for(int i  = 0; i < recvChunks.size(); i++){
                if(colsplit) recvChunks[i].Transpose();
                concat_row = std::max(concat_row, recvChunks[i].getnrow());
                concat_col = concat_col + recvChunks[i].getncol();
            }
            DER * localMatrix = new DER(0, concat_row, concat_col, 0);
            localMatrix->ColConcatenate(recvChunks);
            if(colsplit) localMatrix->Transpose();
            //layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
            layermat.reset(new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld));
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

            std::vector< std::vector< std::tuple<LIT,LIT, NT> > > sendTuples (nprocs);
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

            LIT datasize;
            std::tuple<LIT,LIT,NT>* recvTuples = ExchangeData(sendTuples, commGrid2D->GetWorld(), datasize);

            IT mdim, ndim;
            LocalDim(nrows, ncols, mdim, ndim);
            SpTuples<LIT, NT>spTuples3d(datasize, mdim, ndim, recvTuples);
            DER * localm3d = new DER(spTuples3d, false);
            //layermat = new SpParMat<IT, NT, DER>(localm3d, commGrid3D->GetCommGridLayer());
            layermat.reset(new SpParMat<IT, NT, DER>(localm3d, commGrid3D->GetCommGridLayer()));
        }
    }

    // Create a new copy of a 3D matrix in row split or column split manner
    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat3D< IT,NT,DER > & A, bool colsplit): colsplit(colsplit){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        typedef typename DER::LocalIT LIT;
        auto AcommGrid3D = A.getcommgrid3D();
        int nprocs = AcommGrid3D->GetSize();
        commGrid3D.reset(new CommGrid3D(AcommGrid3D->GetWorld(), AcommGrid3D->GetGridLayers(), 0, 0, A.isSpecial()));

        // Intialize these two variables for new SpParMat3D
        special = A.isSpecial();
        nlayers = AcommGrid3D->GetGridLayers();

        DER * spSeq = A.seqptr(); // local submatrix
        DER * localMatrix = new DER(*spSeq);
        if((A.isColSplit() && !colsplit) || (!A.isColSplit() && colsplit)){
            // If given matrix is column split and desired matrix is row split
            // Or if given matrix is row split and desired matrix is column split
            std::vector<DER> sendChunks;
            int numChunks = commGrid3D->GetGridLayers();
            if(!colsplit) localMatrix->Transpose();
            localMatrix->ColSplit(numChunks, sendChunks);
            if(!colsplit){
                for(int i = 0; i < sendChunks.size(); i++) sendChunks[i].Transpose();
            }

            IT datasize; NT x = 71.0;
            std::vector<DER> recvChunks;

            SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, x, recvChunks);

            typename DER::LocalIT concat_row = 0, concat_col = 0;
            for(int i  = 0; i < recvChunks.size(); i++){
                if(colsplit) recvChunks[i].Transpose();
                concat_row = std::max(concat_row, recvChunks[i].getnrow());
                concat_col = concat_col + recvChunks[i].getncol();
            }
            localMatrix = new DER(0, concat_row, concat_col, 0);
            localMatrix->ColConcatenate(recvChunks);
            if(colsplit) localMatrix->Transpose();
        }
        else{
            // If given and desired matrix both are row split
            // Or if given and desired matrix both are column split
            // Do nothing
        }
        //layermat = new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld);
        layermat.reset(new SpParMat<IT, NT, DER>(localMatrix, commGrid3D->layerWorld));
    }
    
    /*
     *  Only calculates owner in terms of non-special distribution
     * */
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
        typedef typename DER::LocalIT LIT;
        if(special){
            DER * spSeq = layermat->seqptr();
            std::vector<DER> localChunks;
            int sqrtLayers = (int)std::sqrt((float)commGrid3D->GetGridLayers());
            LIT grid3dCols = commGrid3D->gridCols; LIT grid3dRows = commGrid3D->gridRows;
            LIT grid2dCols = grid3dCols * sqrtLayers; LIT grid2dRows = grid3dRows * sqrtLayers;
            IT x = (colsplit) ? layermat->getnrow() : layermat->getncol();
            LIT y = (colsplit) ? (x / grid2dRows) : (x / grid2dCols);
            vector<LIT> divisions2d;
            if(colsplit){
                for(LIT i = 0; i < grid2dRows-1; i++) divisions2d.push_back(y);
                divisions2d.push_back(layermat->getnrow()-(grid2dRows-1)*y);
            }
            else{
                for(LIT i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
                divisions2d.push_back(layermat->getncol()-(grid2dCols-1)*y);
            }
            vector<LIT> divisions2dChunk;
            LIT start = (colsplit) ? ((commGrid3D->rankInLayer / grid3dRows) * sqrtLayers) : ((commGrid3D->rankInLayer % grid3dCols) * sqrtLayers);
            LIT end = start + sqrtLayers;
            for(LIT i = start; i < end; i++){
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
            SpecialExchangeData(sendChunks, commGrid3D->fiberWorld, datasize, z, recvChunks);

            LIT concat_row = 0, concat_col = 0;
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
            int nProcs = commGrid3D->GetSize(); // Total number of processes in the process grid
            int nGridLayers = commGrid3D->GetGridLayers(); // Number of layers in the process grid
            int nGridCols = commGrid3D->GetGridCols(); // Number of process columns in a layer of the grid, which can be thought of L0
            int nGridRows = commGrid3D->GetGridRows(); // Number of process rows in a layer of the grid, which can be thought of L0
            int rankInProcCol_L0 = commGrid3D->commGridLayer->GetRankInProcCol();
            int rankInProcRow_L0 = commGrid3D->commGridLayer->GetRankInProcRow();
            IT m = getnrow(); // Total number of rows of the matrix
            IT n = getncol(); // Total number of columns of the matrix
            IT a = n / nGridCols;
            IT b = n - (a * (nGridCols - 1));
            IT c = m / nGridRows;
            IT d = m - (c * (nGridRows - 1));
            IT w = a / nGridLayers;
            IT x = a - (w * (nGridLayers - 1));
            IT y = b / nGridLayers;
            IT z = b - (y * (nGridLayers - 1));
            IT p = c / nGridLayers;
            IT q = c - (p * (nGridLayers - 1));
            IT r = d / nGridLayers;
            IT s = d - (r * (nGridLayers - 1));

            std::shared_ptr<CommGrid> grid2d;
            grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
            SpParMat<IT, NT, DER> A2D (grid2d);

            std::vector< std::vector < std::tuple<LIT,LIT,NT> > > data(nProcs);
            DER* spSeq = layermat->seqptr(); // local submatrix
            LIT locsize = 0;
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit){
                LIT lcol = colit.colid();
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit){
                    LIT lrow = nzit.rowid();
                    NT val = nzit.value();
                    LIT lrow_L0, lcol_L0;
                    if(colsplit){
                        // If 3D distribution is column split
                        lrow_L0 = lrow;
                        if(commGrid3D->commGridLayer->GetRankInProcRow() < (nGridCols-1)){
                            // If this process is not last in the process column
                            lcol_L0 = w * commGrid3D->rankInFiber + lcol;
                        }
                        else{
                            // If this process is last in the process column
                            lcol_L0 = y * commGrid3D->rankInFiber + lcol;
                        }
                    }
                    else{
                        // If 3D distribution is rowsplit
                        lcol_L0 = lcol; 
                        if(commGrid3D->commGridLayer->GetRankInProcCol() < (nGridRows-1)){
                            // If this process is not last in the process column
                            lrow_L0 = p * commGrid3D->rankInFiber + lrow;
                        }
                        else{
                            // If this process is last in the process column
                            lrow_L0 = r * commGrid3D->rankInFiber + lrow;
                        }
                    }
                    IT grow = commGrid3D->commGridLayer->GetRankInProcCol() * c + lrow_L0;
                    IT gcol = commGrid3D->commGridLayer->GetRankInProcRow() * a + lcol_L0;
                    
                    LIT lrow2d, lcol2d;
                    int owner = A2D.Owner(m, n, grow, gcol, lrow2d, lcol2d);
                    data[owner].push_back(std::make_tuple(lrow2d,lcol2d,val));
                    locsize++;
                }
            }
            A2D.SparseCommon(data, locsize, m, n, maximum<NT>());
            
            return A2D;
        }
    }

    template <class IT, class NT, class DER>
    template <typename SR>
    SpParMat3D<IT, NT, DER> SpParMat3D<IT, NT, DER>::MemEfficientSpGEMM3D(SpParMat3D<IT, NT, DER> & B, 
            int phases, NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct, int kselectVersion, double perProcessMemory){
        int myrank;
        double vm_usage, resident_set;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        typedef typename DER::LocalIT LIT;

        /* 
         * Check if A and B are multipliable 
         * */
        if(getncol() != B.getnrow()){
            std::ostringstream outs;
            outs << "Can not multiply, dimensions does not match"<< std::endl;
            outs << getncol() << " != " << B.getnrow() << std::endl;
            SpParHelper::Print(outs.str());
            MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        }

        /* 
         * If provided number of phase is too low or too high then reset value of phase as 1 
         * */
        if(phases < 1 || phases >= getncol()){
            SpParHelper::Print("MemEfficientSpGEMM3D: The value of phases is too small or large. Resetting to 1.\n");
            phases = 1;
        }
        double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9; // To time different parts of the function
#ifdef TIMING
        MPI_Barrier(B.getcommgrid()->GetWorld());
        t0 = MPI_Wtime();
#endif
        /* 
         * If per process memory is provided then calculate number of phases 
         * Otherwise, proceed to multiplication.
         * */
        if(perProcessMemory > 0) {
            int calculatedPhases = CalculateNumberOfPhases<SR>(B, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion, perProcessMemory);
            //if(calculatedPhases > phases) phases = calculatedPhases;
            phases = calculatedPhases;
        }
        else{
            // Do nothing
        }
#ifdef TIMING
        MPI_Barrier(B.getcommgrid()->GetWorld());
        t1 = MPI_Wtime();
        mcl3d_symbolictime+=(t1-t0);
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tSymbolic stage time: %lf\n", (t1-t0));
#endif
        
        
        /*
         * Calculate, accross fibers, which process should get how many columns after redistribution
         * */
        vector<LIT> divisions3d;
        B.CalculateColSplitDistributionOfLayer(divisions3d);

        /*
         * Split B according to calculated number of phases
         * For better load balancing split B into nlayers*phases chunks
         * */
        vector<DER*> PiecesOfB;
        vector<DER*> tempPiecesOfB;
        DER CopyB = *(B.layermat->seqptr());
        CopyB.ColSplit(divisions3d, tempPiecesOfB); // Split B into `nlayers` chunks at first
        for(int i = 0; i < tempPiecesOfB.size(); i++){
            vector<DER*> temp;
            tempPiecesOfB[i]->ColSplit(phases, temp); // Split each chunk of B into `phases` chunks
            for(int j = 0; j < temp.size(); j++){
                PiecesOfB.push_back(temp[j]);
            }
        }

        //DER * localLayerResultant = new DER(0, layermat->seqptr()->getnrow(), divisions3d[commGrid3D->rankInFiber], 0);
        //SpParMat<IT, NT, DER> layerResultant(localLayerResultant, commGrid3D->layerWorld);
        if(myrank == 0){
            fprintf(stderr, "[MemEfficientSpGEMM3D]\tRunning with phase: %d\n", phases);
        }

        //for(int p = 0; p < phases; p++){
        for(int p = 0; p < std::min(phases, mcl3d_max_phase); p++){
            /*
             * At the start of each phase take appropriate pieces from previously created pieces of local B matrix
             * Appropriate means correct pieces so that 3D-merge can be properly load balanced.
             * */
            vector<LIT> lbDivisions3d;
            LIT totalLocalColumnInvolved = 0;
            vector<DER*> targetPiecesOfB;
            for(int i = 0; i < PiecesOfB.size(); i++){
                if(i % phases == p){
                    targetPiecesOfB.push_back(new DER(*(PiecesOfB[i])));
                    lbDivisions3d.push_back(PiecesOfB[i]->getncol());
                    totalLocalColumnInvolved += PiecesOfB[i]->getncol();
                }
            }

            /*
             * Create new local matrix by concatenating appropriately picked pieces
             * */
            DER * OnePieceOfB = new DER(0, (B.layermat)->seqptr()->getnrow(), totalLocalColumnInvolved, 0);
            OnePieceOfB->ColConcatenate(targetPiecesOfB);
            vector<DER*>().swap(targetPiecesOfB);

            /*
             * Create a new layer-wise distributed matrix with the newly created local matrix for this phase
             * This matrix is used in SUMMA multiplication of respective layer
             * */
            SpParMat<IT, NT, DER> OnePieceOfBLayer(OnePieceOfB, commGrid3D->layerWorld);
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t0 = MPI_Wtime();
#endif
            /*
             *  SUMMA Starts
             * */

            int stages, dummy; 	// last two parameters of ProductGrid are ignored for this multiplication
            std::shared_ptr<CommGrid> GridC = ProductGrid((layermat->getcommgrid()).get(), 
                                                          (OnePieceOfBLayer.getcommgrid()).get(), 
                                                          stages, dummy, dummy);		
            IT C_m = layermat->seqptr()->getnrow();
            IT C_n = OnePieceOfBLayer.seqptr()->getncol();

            IT ** ARecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
            IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
            
            SpParHelper::GetSetSizes( *(layermat->seqptr()), ARecvSizes, (layermat->getcommgrid())->GetRowWorld() );
            SpParHelper::GetSetSizes( *(OnePieceOfBLayer.seqptr()), BRecvSizes, (OnePieceOfBLayer.getcommgrid())->GetColWorld() );

            // Remotely fetched matrices are stored as pointers
            DER * ARecv; 
            DER * BRecv;
            std::vector< SpTuples<IT,NT>  *> tomerge;

            int Aself = (layermat->getcommgrid())->GetRankInProcRow();
            int Bself = (OnePieceOfBLayer.getcommgrid())->GetRankInProcCol();	

            double Abcast_time = 0;
            double Bbcast_time = 0;
            double Local_multiplication_time = 0;
            
            for(int i = 0; i < stages; ++i) {
                std::vector<IT> ess;	

                if(i == Aself){
                    ARecv = layermat->seqptr();	// shallow-copy 
                }
                else{
                    ess.resize(DER::esscount);
                    for(int j=0; j<DER::esscount; ++j) {
                        ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
                    }
                    ARecv = new DER();				// first, create the object
                }
                MPI_Barrier(layermat->getcommgrid()->GetWorld());
#ifdef TIMING
                MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t2 = MPI_Wtime();
#endif
                if (Aself != i) {
                    ARecv->Create(ess);
                }

                Arr<IT,NT> Aarrinfo = ARecv->GetArrays();

                for(unsigned int idx = 0; idx < Aarrinfo.indarrs.size(); ++idx) {
                    MPI_Bcast(Aarrinfo.indarrs[idx].addr, Aarrinfo.indarrs[idx].count, MPIType<IT>(), i, GridC->GetRowWorld());
                }

                for(unsigned int idx = 0; idx < Aarrinfo.numarrs.size(); ++idx) {
                    MPI_Bcast(Aarrinfo.numarrs[idx].addr, Aarrinfo.numarrs[idx].count, MPIType<NT>(), i, GridC->GetRowWorld());
                }
#ifdef TIMING
                MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t3 = MPI_Wtime();
                mcl3d_Abcasttime += (t3-t2);
                Abcast_time += (t3-t2);
#endif
                ess.clear();	
                
                if(i == Bself){
                    BRecv = OnePieceOfBLayer.seqptr();	// shallow-copy
                }
                else{
                    ess.resize(DER::esscount);		
                    for(int j=0; j<DER::esscount; ++j)	{
                        ess[j] = BRecvSizes[j][i];	
                    }	
                    BRecv = new DER();
                }
#ifdef TIMING
                MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t2 = MPI_Wtime();
#endif
                if (Bself != i) {
                    BRecv->Create(ess);	
                }

                Arr<IT,NT> Barrinfo = BRecv->GetArrays();

                for(unsigned int idx = 0; idx < Barrinfo.indarrs.size(); ++idx) {
                    MPI_Bcast(Barrinfo.indarrs[idx].addr, Barrinfo.indarrs[idx].count, MPIType<IT>(), i, GridC->GetColWorld());
                }

                for(unsigned int idx = 0; idx < Barrinfo.numarrs.size(); ++idx) {
                    MPI_Bcast(Barrinfo.numarrs[idx].addr, Barrinfo.numarrs[idx].count, MPIType<NT>(), i, GridC->GetColWorld());
                }
                //if(Bself == i){
                    //fprintf(stderr, "myrank: %d broadcasting %d nonzero columns\n", myrank, Barrinfo.indarrs[1].count);
                    //fprintf(stderr, "myrank: %d broadcasting %d nonzeros\n", myrank, Barrinfo.numarrs[0].count);
                    //fprintf(stderr, "sssss\n");
                //}
#ifdef TIMING
                MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t3 = MPI_Wtime();
                mcl3d_Bbcasttime += (t3-t2);
                Bbcast_time += (t3-t2);
#endif
                MPI_Barrier(layermat->getcommgrid()->GetWorld());

#ifdef TIMING
                //MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t2 = MPI_Wtime();
#endif
                SpTuples<IT,NT> * C_cont = LocalHybridSpGEMM<SR, NT>
                                (*ARecv, *BRecv, // parameters themselves
                                i != Aself, 	// 'delete A' condition
                                i != Bself);	// 'delete B' condition
#ifdef TIMING
                //MPI_Barrier(layermat->getcommgrid()->GetWorld());
                t3 = MPI_Wtime();
                mcl3d_localspgemmtime += (t3-t2);
                Local_multiplication_time += (t3-t2);
#endif
                
                if(!C_cont->isZero()) tomerge.push_back(C_cont);
            }

            SpHelper::deallocate2D(ARecvSizes, DER::esscount);
            SpHelper::deallocate2D(BRecvSizes, DER::esscount);

#ifdef TIMING
            //MPI_Barrier(layermat->getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
            SpTuples<IT,NT> * C_tuples = MultiwayMergeHash<SR>(tomerge, C_m, C_n, true);
#ifdef TIMING
            //MPI_Barrier(layermat->getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            mcl3d_SUMMAmergetime += (t3-t2);
#endif

#ifdef TIMING   
            if(myrank == 0){
                fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAbcast_time: %lf\n", p, Abcast_time);
                fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tBbcast_time: %lf\n", p, Bbcast_time);
                fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tLocal_multiplication_time: %lf\n", p, Local_multiplication_time);
                fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA Merge time: %lf\n", p, (t3-t2));
            }
#endif
            /*
             *  SUMMA Ends
             * */
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t1 = MPI_Wtime();
            mcl3d_SUMMAtime += (t1-t0);
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA time: %lf\n", p, (t1-t0));
#endif
            /*
             * 3d-reduction starts
             * */
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t0 = MPI_Wtime();
#endif
            MPI_Datatype MPI_tuple;
            MPI_Type_contiguous(sizeof(std::tuple<LIT,LIT,NT>), MPI_CHAR, &MPI_tuple);
            MPI_Type_commit(&MPI_tuple);
            
            /*
             *  Create a profile with information regarding data to be sent and received between layers 
             *  These memory allocation needs to be `int` specifically because some of these arrays would be used in communication
             *  This is requirement is for MPI as MPI_Alltoallv takes pointer to integer exclusively as count and displacement
             * */
            int * sendcnt    = new int[getcommgrid()->GetGridLayers()];
            int * sendprfl   = new int[getcommgrid()->GetGridLayers()*3];
            int * sdispls    = new int[getcommgrid()->GetGridLayers()]();
            int * recvcnt    = new int[getcommgrid()->GetGridLayers()];
            int * recvprfl   = new int[getcommgrid()->GetGridLayers()*3];
            int * rdispls    = new int[getcommgrid()->GetGridLayers()]();

            vector<IT> lbDivisions3dPrefixSum(lbDivisions3d.size());
            lbDivisions3dPrefixSum[0] = 0;
            std::partial_sum(lbDivisions3d.begin(), lbDivisions3d.end()-1, lbDivisions3dPrefixSum.begin()+1);
            ColLexiCompare<IT,NT> comp;
            IT totsend = C_tuples->getnnz();
            
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
#pragma omp parallel for
            for(int i=0; i < getcommgrid()->GetGridLayers(); ++i){
                IT start_col = lbDivisions3dPrefixSum[i];
                IT end_col = lbDivisions3dPrefixSum[i] + lbDivisions3d[i];
                std::tuple<IT, IT, NT> search_tuple_start(0, start_col, NT());
                std::tuple<IT, IT, NT> search_tuple_end(0, end_col, NT());
                std::tuple<IT, IT, NT>* start_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_start, comp);
                std::tuple<IT, IT, NT>* end_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_end, comp);
                // This type casting is important from semantic point of view
                sendcnt[i] = (int)(end_it - start_it);
                sendprfl[i*3+0] = (int)(sendcnt[i]); // Number of nonzeros in ith chunk
                sendprfl[i*3+1] = (int)(layermat->seqptr()->getnrow()); // Number of rows in ith chunk
                sendprfl[i*3+2] = (int)(lbDivisions3d[i]); // Number of columns in ith chunk
            }
            std::partial_sum(sendcnt, sendcnt+getcommgrid()->GetGridLayers()-1, sdispls+1);
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoall data ready: %lf\n", p, (t3-t2));
#endif

            // Send profile ready. Now need to update the tuples to reflect correct column id after column split.
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
            for(int i=0; i < getcommgrid()->GetGridLayers(); ++i){
#pragma omp parallel for schedule(static)
                for(int j = 0; j < sendcnt[i]; j++){
                    std::get<1>(C_tuples->tuples[sdispls[i]+j]) = std::get<1>(C_tuples->tuples[sdispls[i]+j]) - lbDivisions3dPrefixSum[i];
                }
            }
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoallv data ready: %lf\n", p, (t3-t2));
#endif

#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
            MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, commGrid3D->fiberWorld);
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoall: %lf\n", p, (t3-t2));
#endif
#ifdef TIMING
            //MPI_Barrier(getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
            for(int i = 0; i < getcommgrid()->GetGridLayers(); i++) recvcnt[i] = recvprfl[i*3];
            std::partial_sum(recvcnt, recvcnt+getcommgrid()->GetGridLayers()-1, rdispls+1);
            IT totrecv = std::accumulate(recvcnt,recvcnt+getcommgrid()->GetGridLayers(), static_cast<IT>(0));
            //std::tuple<LIT,LIT,NT>* recvTuples = new std::tuple<LIT,LIT,NT>[totrecv];
            std::tuple<LIT,LIT,NT>* recvTuples = static_cast<std::tuple<LIT,LIT,NT>*> (::operator new (sizeof(std::tuple<LIT,LIT,NT>[totrecv])));
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAllocation of receive data: %lf\n", p, (t3-t2));
#endif

#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t2 = MPI_Wtime();
#endif
            MPI_Alltoallv(C_tuples->tuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, commGrid3D->fiberWorld);
            delete C_tuples;
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoallv: %lf\n", p, (t3-t2));
#endif
            vector<SpTuples<IT, NT>*> recvChunks(getcommgrid()->GetGridLayers());
#pragma omp parallel for
            for (int i = 0; i < getcommgrid()->GetGridLayers(); i++){
                recvChunks[i] = new SpTuples<LIT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], recvTuples + rdispls[i], true, false);
            }

            // Free all memory except tempTuples; Because that memory is holding data of newly created local matrices after receiving.
            DeleteAll(sendcnt, sendprfl, sdispls);
            DeleteAll(recvcnt, recvprfl, rdispls); 
            MPI_Type_free(&MPI_tuple);
            /*
             * 3d-reduction ends 
             * */
            
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t1 = MPI_Wtime();
            mcl3d_reductiontime += (t1-t0);
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tReduction time: %lf\n", p, (t1-t0));
#endif
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t0 = MPI_Wtime();
            t2 = MPI_Wtime();
#endif
            /*
             * 3d-merge starts 
             * */
            SpTuples<IT, NT> * merged_tuples = MultiwayMergeHash<SR, IT, NT>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false); // Do not delete
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t3 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMultiway Merge: %lf\n", p, (t3-t2));
            mcl3d_layer_nnzc += merged_tuples->getnnz();
#endif
            //DER * phaseResultant = new DER(*merged_tuples, false);
            delete merged_tuples;
            // Do not delete elements of recvChunks, because that would give segmentation fault due to double free
            //delete [] recvTuples;
            ::operator delete(recvTuples);
            for(int i = 0; i < recvChunks.size(); i++){
                recvChunks[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
                delete recvChunks[i];
            }
            vector<SpTuples<IT,NT>*>().swap(recvChunks);
            //SpParMat<IT, NT, DER> phaseResultantLayer(phaseResultant, commGrid3D->layerWorld);
            /*
             * 3d-merge ends
             * */
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t1 = MPI_Wtime();
            mcl3d_3dmergetime += (t1-t0);
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\t3D Merge time: %lf\n", p, (t1-t0));
#endif

#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t0 = MPI_Wtime();
#endif
            //MCLPruneRecoverySelect(phaseResultantLayer, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);
#ifdef TIMING
            //MPI_Barrier(B.getcommgrid()->GetWorld());
            t1 = MPI_Wtime();
            mcl3d_kselecttime += (t1-t0);
            if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMCLPruneRecoverySelect time: %lf\n",p, (t1-t0));
#endif
            
            //if(jj == 0) layerResultant += phaseResultantLayer;
            //phaseResultantLayer.FreeMemory();
#ifdef TIMING
            if(myrank == 0) fprintf(stderr, "***\n");
#endif
        }
        for(int i = 0; i < PiecesOfB.size(); i++) delete PiecesOfB[i];

        std::shared_ptr<CommGrid3D> grid3d;
        grid3d.reset(new CommGrid3D(commGrid3D->GetWorld(), commGrid3D->GetGridLayers(), commGrid3D->GetGridRows(), commGrid3D->GetGridCols(), isSpecial()));
        //DER * localResultant = new DER(*localLayerResultant);
        DER * localResultant = new DER();
        SpParMat3D<IT, NT, DER> C3D(localResultant, grid3d, isColSplit(), isSpecial());
        return C3D;
    }

    template <class IT, class NT, class DER>
    template <typename SR>
    int SpParMat3D<IT, NT, DER>::CalculateNumberOfPhases(SpParMat3D<IT, NT, DER> & B, 
            NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct, int kselectVersion, double perProcessMemory){
        int myrank;
        double vm_usage, resident_set;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        int p, phases;
        MPI_Comm_size(getcommgrid3D()->GetLayerWorld(),&p);
        int64_t perNNZMem_in = sizeof(IT)*2 + sizeof(NT);
        int64_t perNNZMem_out = sizeof(IT)*2 + sizeof(NT);

        int64_t lannz = layermat->getlocalnnz();
        int64_t gannz = 0;
        MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, getcommgrid3D()->GetWorld());
        int64_t inputMem = gannz * perNNZMem_in * 4; // Four pieces per process: one piece of own A and B, one piece of received A and B
        
        int64_t asquareNNZ = EstPerProcessNnzSUMMA(*layermat, *(B.layermat), true);
        int64_t gasquareNNZ;
        // 3D Specific
        MPI_Allreduce(&asquareNNZ, &gasquareNNZ, 1, MPIType<int64_t>(), MPI_MAX, commGrid3D->GetFiberWorld());
        int64_t asquareMem = gasquareNNZ * perNNZMem_out * 4; // because += operation needs three times memory!!
        if(myrank == 0) {
            fprintf(stderr, "Per process gasquareNNZ: %lld\n", gasquareNNZ);
        }

        int64_t d = ceil( (gasquareNNZ * sqrt(p))/ layermat->getlocalcols() );
        int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
        //int64_t k = int64_t(std::max(selectNum, recoverNum));

        // estimate output memory
        //int64_t outputNNZ =ceil((layermat->getlocalcols() * k)/sqrt(p)); // If kselect is run
        int64_t outputNNZ =ceil((layermat->getlocalcols() * d)/sqrt(p)); // If keselect is not run
        int64_t outputMem = outputNNZ * perNNZMem_out * 2;
        //double remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
        double remainingMem = perProcessMemory*1000000000 - inputMem; // If output of each phase is discarded
        int64_t kselectmem = layermat->getlocalcols() * k * sizeof(NT) * 3;

        ////inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        //if(remainingMem > 0){
            //phases = ceil((asquareMem+kselectmem) / remainingMem); // If kselect is run
            //return phases;
        //}
        //else return -1;

        phases = ceil(asquareMem / remainingMem);   // If kselect is not run
        return phases;
    }
    
    /*
     *  Calculate, which process accross fiber should get how many columns 
     *  if layer matrix of this 3D matrix is distributed in column split way
     * */
    template <class IT, class NT, class DER>
    void SpParMat3D<IT,NT,DER>::CalculateColSplitDistributionOfLayer(vector<typename DER::LocalIT> & divisions3d){
        if(special){
            vector<IT> divisions2d;
            int sqrtLayers = (int)std::sqrt((float)commGrid3D->GetGridLayers());
            int grid3dCols = commGrid3D->GetGridCols();
            int grid2dCols = grid3dCols * sqrtLayers;
            IT x = (layermat)->getncol();
            IT y = x / grid2dCols;
            for(int i = 0; i < grid2dCols-1; i++) divisions2d.push_back(y);
            divisions2d.push_back(x-(grid2dCols-1)*y);
            vector<IT> divisions2dChunk;
            IT start = (commGrid3D->rankInLayer % grid3dCols) * sqrtLayers;
            IT end = start + sqrtLayers;
            for(int i = start; i < end; i++){
                divisions2dChunk.push_back(divisions2d[i]);
            }
            for(int i = 0; i < divisions2dChunk.size(); i++){
                IT z = divisions2dChunk[i]/sqrtLayers;
                for(int j = 0; j < sqrtLayers-1; j++) divisions3d.push_back(z);
                divisions3d.push_back(divisions2dChunk[i]-(sqrtLayers-1)*z);
            }
        }
        else{
            // For non-special distribution, partitioning for 3D can be achieved by dividing local columns in #layers equal partitions
            IT x = layermat->seqptr()->getncol();
            int nlayers = commGrid3D->GetGridLayers();
            IT y = x / nlayers;
            for(int i = 0; i < nlayers-1; i++) divisions3d.push_back(y);
            divisions3d.push_back(x-(nlayers-1)*y);
        }
    }

    /*
     * Checks if the layer matrix is 2D SpParMat compatible
     * */
    template <class IT, class NT, class DER>
    bool SpParMat3D<IT,NT,DER>::CheckSpParMatCompatibility(){
        IT nLayerCols = layermat->getncol();
        IT nLayerRows = layermat->getnrow();
        IT localCols = layermat->getlocalcols();
        IT localRows = layermat->getlocalrows();
        int nGridCols = layermat->getcommgrid()->GetGridCols();
        int nGridRows = layermat->getcommgrid()->GetGridRows();
        int idxGridRow = layermat->getcommgrid()->GetRankInProcCol();
        int idxGridCol = layermat->getcommgrid()->GetRankInProcRow();
        IT x, y, a, b;
        x = nLayerRows / nGridRows;
        y = (nLayerRows % nGridRows == 0) ? x : (nLayerRows - x * (nGridRows - 1)); 
        a = nLayerCols / nGridCols;
        b = (nLayerCols % nGridCols == 0) ? a : (nLayerCols - a * (nGridCols - 1)); 
        bool flag = true;
        if(idxGridRow == nGridRows-1){
            if(localRows != y) flag = false;
        }
        else{
            if(localRows != x) flag = false;
        }
        if(idxGridCol == nGridCols-1){
            if(localCols != b) flag = false;
        }
        else{
            if(localCols != a) flag = false;
        }
        return flag;
    }

    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnrow() const {
        IT totalrows_layer = layermat->getnrow();
        IT totalrows = 0;
        if(!colsplit) MPI_Allreduce( &totalrows_layer, &totalrows, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        else totalrows = totalrows_layer;
        return totalrows;
    }
    
    
    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getncol() const {
        IT totalcols_layer = layermat->getncol();
        IT totalcols = 0;
        if(colsplit) MPI_Allreduce( &totalcols_layer, &totalcols, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        else totalcols = totalcols_layer;
        return totalcols;
    }


    template <class IT, class NT, class DER>
    IT SpParMat3D< IT,NT,DER >::getnnz() const {
        IT totalnz_layer = layermat->getnnz();
        IT totalnz = 0;
        MPI_Allreduce( &totalnz_layer, &totalnz, 1, MPIType<IT>(), MPI_SUM, commGrid3D->fiberWorld);
        return totalnz;
    }

    template <class IT, class NT>
    std::tuple<IT,IT,NT>* ExchangeData(std::vector<std::vector<std::tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World, IT& datasize)
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
    void SpecialExchangeData( std::vector<DER> & sendChunks, MPI_Comm World, IT& datasize, NT dummy, vector<DER> & recvChunks){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        double vm_usage, resident_set;
        typedef typename DER::LocalIT LIT;
        int numChunks = sendChunks.size();

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<LIT,LIT,NT>), MPI_CHAR, &MPI_tuple);
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
        for(int i = 0; i < numChunks; i++) recvcnt[i] = recvprfl[i*3];

        std::partial_sum(sendcnt, sendcnt+numChunks-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+numChunks-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+numChunks, static_cast<IT>(0));

        std::tuple<LIT,LIT,NT>* sendTuples = new std::tuple<LIT,LIT,NT>[totsend];
	    std::tuple<LIT,LIT,NT>* recvTuples = new std::tuple<LIT,LIT,NT>[totrecv];

        int kk=0;
        for(int i = 0; i < numChunks; i++){
            for(typename DER::SpColIter colit = sendChunks[i].begcol(); colit != sendChunks[i].endcol(); ++colit){
                for(typename DER::SpColIter::NzIter nzit = sendChunks[i].begnz(colit); nzit != sendChunks[i].endnz(colit); ++nzit){
                    NT val = nzit.value();
                    sendTuples[kk++] = std::make_tuple(nzit.rowid(), colit.colid(), nzit.value());
                }
            }
        }

        MPI_Alltoallv(sendTuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);
	    DeleteAll(sendcnt, sendprfl, sdispls, sendTuples);

        //tuple<LIT, LIT, NT> ** tempTuples = new tuple<LIT, LIT, NT>*[numChunks];
        tuple<LIT, LIT, NT> ** tempTuples = new tuple<LIT, LIT, NT>*[numChunks];
        for (int i = 0; i < numChunks; i++){
            tempTuples[i] = new tuple<LIT, LIT, NT>[recvcnt[i]];
            memcpy(tempTuples[i], recvTuples+rdispls[i], recvcnt[i]*sizeof(tuple<LIT, LIT, NT>));
        }

        for (int i = 0; i < numChunks; i++){
            recvChunks.push_back(DER(SpTuples<LIT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], tempTuples[i]), false));
        }

        // Free all memory except tempTuples; Because that memory is holding data of newly created local matrices after receiving.
        DeleteAll(recvcnt, recvprfl, rdispls, recvTuples); 
        MPI_Type_free(&MPI_tuple);

	    return;
    }

    template <class IT, class NT, class DER>
    void SpecialExchangeData_2( std::vector<DER*> & sendChunks, MPI_Comm World, IT& datasize, NT dummy, vector<SpTuples<IT,NT>*> & recvChunks){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        double vm_usage, resident_set;
        double t0, t1;
        typedef typename DER::LocalIT LIT;
        int numChunks = sendChunks.size();

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<LIT,LIT,NT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);

        int * sendcnt = new int[numChunks];
        int * sendprfl = new int[numChunks*3];
        int * sdispls = new int[numChunks]();
        int * recvcnt = new int[numChunks];
        int * recvprfl = new int[numChunks*3];
        int * rdispls = new int[numChunks]();

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        IT totsend = 0;
        for(IT i=0; i<numChunks; ++i){
            sendprfl[i*3] = sendChunks[i]->getnnz();
            sendprfl[i*3+1] = sendChunks[i]->getnrow();
            sendprfl[i*3+2] = sendChunks[i]->getncol();
            sendcnt[i] = sendprfl[i*3];
            totsend += sendcnt[i];
        }
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] Getting send profile ready: %lf\n", (t1-t0));
        }
#endif
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, World);
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] send profile alltoall: %lf\n", (t1-t0));
        }
#endif
        for(int i = 0; i < numChunks; i++) recvcnt[i] = recvprfl[i*3];

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        std::partial_sum(sendcnt, sendcnt+numChunks-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+numChunks-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+numChunks, static_cast<IT>(0));
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] getting receive profile ready: %lf\n", (t1-t0));
        }
#endif

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        std::tuple<LIT,LIT,NT>* sendTuples = new std::tuple<LIT,LIT,NT>[totsend];
	    std::tuple<LIT,LIT,NT>* recvTuples = new std::tuple<LIT,LIT,NT>[totrecv];

#pragma omp parallel for
        for(int i = 0; i < numChunks; i++){
            int kk = sdispls[i];
            for(typename DER::SpColIter colit = sendChunks[i]->begcol(); colit != sendChunks[i]->endcol(); ++colit){
                for(typename DER::SpColIter::NzIter nzit = sendChunks[i]->begnz(colit); nzit != sendChunks[i]->endnz(colit); ++nzit){
                    NT val = nzit.value();
                    sendTuples[kk++] = std::make_tuple(nzit.rowid(), colit.colid(), nzit.value());
                }
            }
        }
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] memory allocation and data copy before actual alltoallv: %lf\n", (t1-t0));
        }
#endif

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        MPI_Alltoallv(sendTuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] alltoallv: %lf\n", (t1-t0));
        }
#endif
	    DeleteAll(sendcnt, sendprfl, sdispls, sendTuples);

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
        //tuple<LIT, LIT, NT> ** tempTuples = new tuple<LIT, LIT, NT>*[numChunks];
        tuple<LIT, LIT, NT> ** tempTuples = new tuple<LIT, LIT, NT>*[numChunks];
#pragma omp parallel for
        for (int i = 0; i < numChunks; i++){
            tempTuples[i] = new tuple<LIT, LIT, NT>[recvcnt[i]];
            memcpy(tempTuples[i], recvTuples+rdispls[i], recvcnt[i]*sizeof(tuple<LIT, LIT, NT>));
        }
#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] tempTuple allocations and memcpy: %lf\n", (t1-t0));
        }
#endif

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
        }
#endif
#pragma omp parallel for
        for (int i = 0; i < numChunks; i++){
            recvChunks[i] = new SpTuples<LIT, NT>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], tempTuples[i], true, false);
        }

#ifdef TIMING
        if(dummy == 17.0){
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            if(myrank == 0) fprintf(stderr, "[SpecialExchangeData_2] creation of SpDCCols from tempTuples: %lf\n", (t1-t0));
        }
#endif
        // Free all memory except tempTuples; Because that memory is holding data of newly created local matrices after receiving.
        DeleteAll(recvcnt, recvprfl, rdispls, recvTuples); 
        MPI_Type_free(&MPI_tuple);

	    return;
    }
}
