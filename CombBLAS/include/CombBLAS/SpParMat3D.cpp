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
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool csplit)
    {
        // Save the flag whether this 3D distributed matrix is formed by splitting the 2D distributed matrix columnwise
        colsplit = csplit;
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        // Get total number of processors in the original 2D CommGrid.
        // Because number of processors in each layer of 3D grid would be determined from this number
        int nprocs = commGrid2D->GetSize();
        // Create a 3D CommGrid with all the processors involved in the 2D CommGrid
        commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0));
        
        // Total number of rows in the matrix
        IT nrows = A2D.getnrow();
        // Total number of cols in the matrix
        IT ncols = A2D.getncol();
        // Number of rows in original 2D processor grid in which matrix is distributed
        int pr2d = commGrid2D->GetGridRows();
        // Number of cols in original 2D processor grid in which matrix is distributed
        int pc2d = commGrid2D->GetGridCols();
        // On which row in original proc grid does this processor belong
        int rowrank2d = commGrid2D->GetRankInProcRow();
        // On which column in original proc grid does this processor belong
        int colrank2d = commGrid2D->GetRankInProcCol();
        // How many rows of global matrix does each processor in 2D proc grid contain
        IT m_perproc2d = nrows / pr2d;
        // How many columns of global matrix does each processor in 2D proc grid contain
        IT n_perproc2d = ncols / pc2d;
        DER* spSeq = A2D.seqptr(); // local submatrix
        // Global row index of matrix from where current processor starts to store
        IT localRowStart2d = colrank2d * m_perproc2d; // first row in this process
        // Global column index of matrix from where current processor starts to store
        IT localColStart2d = rowrank2d * n_perproc2d; // first col in this process

        LIT lrow3d, lcol3d;
        // Data structure to contain the information about how much data would be sent to each other processors from current processor.
        std::vector<int> tsendcnt(nprocs,0);
        // Iterate over all local columns in current processor
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            // Calculate global column index of corresponding local column index
            IT gcol = colit.colid() + localColStart2d;
            // Iterate over all local nonzero entries in current column
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                // Calculate global row index for corresponding nonzero
                IT grow = nzit.rowid() + localRowStart2d;
                
                //, nzit.value();
                // Calcualting owner of the particular nonzero in 3D processor grid
                // As well as what will be the row and column index of the non-zero after going to 3D
                int owner = Owner(nrows, ncols, grow, gcol, lrow3d, lcol3d); //3D owner
                // Incrementing number of data to send to the calculated owner
                tsendcnt[owner]++;
            }
        }
        
        // Calculation of how much data would be sent to which processor is done
        // Now those data need to be prepared to be sent
        // Initializing a dutu structure which would hold the data to send
        std::vector<std::vector<std::tuple<IT,IT, NT>>> sendTuples (nprocs);
        // Iterate over all local columns in current processor
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            // Calculate global column index of corresponding local column index
            IT gcol = colit.colid() + localColStart2d;
            // Iterate over all local nonzero entries in current column
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                // Calculate global row index for corresponding nonzero
                IT grow = nzit.rowid() + localRowStart2d;
                // Getting value of the non-zero 
                NT val = nzit.value();
                // Calcualting owner of the particular nonzero in 3D processor grid
                // As well as what will be the row and column index of the non-zero after going to 3D
                int owner = Owner(nrows, ncols, grow, gcol, lrow3d, lcol3d); //3D owner
                // Creating a tuple for each non-zero entry and pushing it to appropriate place of the data structure
                sendTuples[owner].push_back(std::make_tuple(lrow3d, lcol3d, val));
            }
        }
        
        for(int i = 0; i < sendTuples.size(); i++){
            cout << "Processor " << i << " would get " << sendTuples[i].size() << endl;
        }

        // Now it's known which nonzero would go to which in 3D grid. Next stage is to send those to appropriate places
        // and receive appropriate non-zeros from other places.
        std::vector<std::tuple<IT,IT,NT>> recvTuples = ExchangeData(sendTuples, commGrid2D->GetWorld());

        cout << "recvTuples : " << recvTuples.size() << endl;
        
     
        IT mdim, ndim;
        LocalDim(nrows, ncols, mdim, ndim);
        cout << "mdim , ndim : "<< mdim << " , " << ndim << endl;
        //SpTuples<IT, NT>spTuples3d(recvTuples.size(), mdim, ndim, recvTuples.data());
        //cout << "After SpTuples" << endl;
        //DER * localm3d = new DER(spTuples3d, false);
        //cout << "After DER *" << endl;
        //// not layer SpParMat
        //std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer;
        //cout << "After commGridLayer" << endl;
        //layermat = new SpParMat<IT, NT, DER>(localm3d, commGridLayer);
        //cout << "After layermat" << endl;
    }

    template <class IT, class NT, class DER>
    SpParMat3D< IT,NT,DER >::SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool csplit, bool special)
    {
        colsplit = csplit;
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        int nprocs = commGrid2D->GetSize();
        commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0, true));

        DER* spSeq = A2D.seqptr(); // local submatrix
        std::vector< DER >localChunks;
        int numChunks = (int)std::sqrt((float)nlayers);
        spSeq->ColSplit(numChunks, localChunks);
        MPI_Barrier(commGrid3D->GetWorld());
        //if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
            //printf("myrank: %d, chunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 0, localChunks[0].getnrow(), localChunks[0].getncol(), localChunks[0].getnnz());
            //printf("myrank: %d, chunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 1, localChunks[1].getnrow(), localChunks[1].getncol(), localChunks[1].getnnz());
            //printf("myrank: %d, chunk: %d, rows: %d, cols: %d, nnz: %d\n", commGrid3D->myrank, 2, localChunks[2].getnrow(), localChunks[2].getncol(), localChunks[2].getnnz());
        //}

        IT datasize;
        NT x = 0.0;
        SpecialExchangeData(localChunks, commGrid3D->specialWorld, datasize, x, commGrid3D->world3D);

        //vector< vector< tuple<IT, IT, NT> > > sendTuples(numChunks);
        //for (int i = 0; i < numChunks; i++){
            //for(typename DER::SpColIter colit = localChunks[i].begcol(); colit != localChunks[i].endcol(); ++colit){
                //for(typename DER::SpColIter::NzIter nzit = localChunks[i].begnz(colit); nzit != localChunks[i].endnz(colit); ++nzit){
                    //NT val = nzit.value();
                    //sendTuples[i].push_back(std::make_tuple(nzit.rowid(), colit.colid(), nzit.value()));
                //}
            //}
        //}
        //IT datasize;
        //std::tuple<IT,IT,NT>* recvTuples = SpecialExchangeData(sendTuples, commGrid3D->specialWorld, datasize);

        //vector< vector<LIT> > chunkEssentials(numChunks);
        //vector<LIT> chunkEssentialsBuf(numChunks * DER::esscount);
        //vector<LIT> rcvEssentialsBuf(numChunks * DER::esscount);
        //for(int i = 0; i < numChunks; i++){
            //chunkEssentials[i] = localChunks[i].GetEssentials();
            //for(int j = 0; j < DER::esscount; j++){
                //chunkEssentialsBuf[i*DER::esscount+j] = chunkEssentials[i][j];
                ////if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
                    ////printf("myrank: %d, chunk: %d, ess[%d]: %d\n", 
                            ////commGrid3D->myrank, i, j, chunkEssentials[i][j]);
                ////}
            //}
        //}
        //for(int i = 0; i < numChunks; i++){
            //for(int j = 0; j < DER::esscount; j++){
                //if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
                    //printf("myrank: %d, chunk: %d, ess[%d]: %d\n", 
                            //commGrid3D->myrank, i, j, chunkEssentialsBuf[i*DER::esscount+j]);
                //}
            //}
        //}
        //std::vector< DER >rcvChunks(numChunks);
        //MPI_Alltoall(chunkEssentialsBuf.data(), DER::esscount, MPIType< LIT >(),
                     //rcvEssentialsBuf.data(), DER::esscount, MPIType< LIT >(), 
                     //commGrid3D->specialWorld);
        //MPI_Barrier(commGrid3D->GetWorld());
        //for(int i = 0; i < numChunks; i++){
            //for(int j = 0; j < DER::esscount; j++){
                //if(commGrid3D->myrank == 0 || commGrid3D->myrank == 6 || commGrid3D->myrank == 12){
                    //printf("myrank: %d, chunk: %d, ess[%d]: %d\n", 
                            //commGrid3D->myrank, i, j, rcvEssentialsBuf[i*DER::esscount+j]);
                //}
            //}
        //}
    }
   
    // Function to calculate owner processor of a particular non-zero in 3D processor grid
    // Patameters:
    //      - total number of vertices in whole matrix
    //      - total number of columns in whole matrix
    //      - global row index of the non-zero
    //      - global column index of the non-zero
    //      - reference to a variable which holds local row index of matrix in the 3D processor grid
    //      - reference to a variable which holds local column index of matrix in the 3D processor grid
    template <class IT, class NT,class DER>
    template <typename LIT>
    int SpParMat3D<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const
    {
        // first map to Layer 0 and then split
        // We would consider distributing whole matrix on only one layer(let say layer zero or L0) of 3D CommGrid. Then split accordingly.
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer; // 2D CommGrid for my layer
        // Getting how many processor rows in this layer
        int procrows = commGridLayer->GetGridRows();
        // Getting how many processor columns in this layer
        int proccols = commGridLayer->GetGridCols();
        // Getting how many layer in the 3D CommGrid
        int nlayers = commGrid3D->gridLayers;
        
        // If distributed in this way
        // Except the processors in last row of layer, how many rows each processors are supposed to contain
        IT m_perproc_L0 = total_m / procrows;
        // Except the processors in last column of layer, how many columns each processors are supposed to contain
        IT n_perproc_L0 = total_n / proccols;
        
        int procrow_L0; // within a layer
        // If in average per process contains at least one row
        if(m_perproc_L0 != 0) // If on average per processor contains at least one row
        {
            // Calculating on which row of layer zero would this non-zero belong
            procrow_L0 = std::min(static_cast<int>(grow / m_perproc_L0), procrows-1);
        }
        else    // all owned by the last processor row
        {
            procrow_L0 = procrows -1;
        }
        int proccol_L0; // within a layer
        // If in average per process contains at least one column
        if(n_perproc_L0 != 0)
        {
            // Calculating on which column of layer zero would this non-zero belong
            proccol_L0 = std::min(static_cast<int>(gcol / n_perproc_L0), proccols-1);
        }
        else 
        {
            proccol_L0 = proccols-1;
        }
        
        // Now calculating what will be local row and column index of this non-zero if 
        // it belongs to the processor of just calculated row and column in layer zero
        IT lrow_L0 = grow - (procrow_L0 * m_perproc_L0);
        IT lcol_L0 = gcol - (proccol_L0 * n_perproc_L0);
        // Now if we split content of each processor in layer zero and scatter along layer,
        // processor row and column of that non-zero would be same independent of which layer it is going
        int proccol_layer = proccol_L0;
        int procrow_layer = procrow_L0;
        int layer;
        // next, split and scatter
        if(colsplit) // If local matrices are split along column to go to 3D
        {
            // Previously we calculated number of columns belonging to per processor in layer zero
            // Now we need to calculate number of columns belonging to per processor along layer after splitting and scattering
            IT n_perproc;
            if(proccol_L0 < commGrid3D->gridCols-1)
            {
                // If this non-zero doesn't belong to last processor, number of columns contained by the processor is the average we calcualted
                // So we are using that to calcualte for per processor along layer
                n_perproc = n_perproc_L0 / nlayers;
            }
            else
            {
                // If this non-zero doesn't belong to last processor, number of columns contained by the processor may not be the average we calcualted
                // So we are taking count of rest of the columns to calcualte for per processor along layer
                n_perproc = (total_n - (n_perproc_L0 * proccol_L0)) / nlayers;
            }
            // If in average per processor along layer contain more than zero columns
            if(n_perproc != 0)
            {
                // Calculating in which layer non-zero should belong
                layer = std::min(static_cast<int>(lcol_L0 / n_perproc), nlayers-1);
            }
            else
                layer = nlayers-1;
            
            // If splitting along column then local row index of that non-zero in 3D would be same as it is now
            lrow = lrow_L0;
            // But local column index would change
            lcol = lcol_L0 - (layer * n_perproc);
        }
        else // If local matrices are split along row to go to 3D. Logics would be similar to column split
        {
            IT m_perproc;
            if(procrow_L0 < commGrid3D->gridRows-1)
            {
                m_perproc = m_perproc_L0 / nlayers;
            }
            else
            {
                m_perproc = (total_m - (m_perproc_L0 * procrow_L0)) / nlayers;
            }
            if(m_perproc != 0)
            {
                layer = std::min(static_cast<int>(lrow_L0 / m_perproc), nlayers-1);
            }
            else
                layer = nlayers-1;
            
            lcol = lcol_L0;
            lrow = lrow_L0 - (layer * m_perproc);
        }
        // Now we know in which row and column of which layer would this non-zero go if translated to 3D
        // ID of that processor would be the owner ID, so that would be returned
        return commGrid3D->GetRank(layer, procrow_layer, proccol_layer);
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
    
    template <class IT, class NT,class DER>
    void SpParMat3D<IT,NT,DER>::LocalDim(IT total_m, IT total_n, IT &localm, IT& localn) const
    {
        // first map to Layer 0 and then split
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer; // CommGrid for my layer
        int procrows = commGridLayer->GetGridRows();
        int proccols = commGridLayer->GetGridCols();
        int nlayers = commGrid3D->gridLayers;
        
        
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
    void SpecialExchangeData( std::vector<DER> & localChunks, MPI_Comm World, IT& datasize, NT dummy, MPI_Comm secondaryWorld)
    {
        int myrank;
        MPI_Comm_rank(secondaryWorld, &myrank);
        int numChunks = localChunks.size();

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);

        int * sendcnt = new int[numChunks];
        int * sendprfl = new int[numChunks*3];
        int * recvcnt = new int[numChunks];
        int * recvprfl = new int[numChunks*3];
        int * sdispls = new int[numChunks]();
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
        if(myrank == 12){
            int i = 1;
            printf("[SENDING] Processor: %d, Chunk: %d\n", myrank, i);
            printf("nnz: %d, mdim: %d, ndim: %d\n",sendprfl[i*3], sendprfl[i*3+1], sendprfl[i*3+2]);
            printf("first tuple: < %lld, %lld, %lf >\n", 
                    get<0>(sendTuples[sdispls[i]]), 
                    get<1>(sendTuples[sdispls[i]]), 
                    get<2>(sendTuples[sdispls[i]]));
            printf("last tuple: < %lld, %lld, %lf >\n", 
                    get<0>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    get<1>(sendTuples[sdispls[i]+sendcnt[i]-1]), 
                    get<2>(sendTuples[sdispls[i]+sendcnt[i]-1]));
            //for(int j = sdispls[i]; j < sdispls[i]+sendcnt[i]; j++){
                //cout << get<0>(sendTuples[j]) << " " << get<1>(sendTuples[j]) << " " << get<2>(sendTuples[j]) << endl;
            //}
        }
        std::tuple<IT,IT,NT>* recvTuples = new std::tuple<IT,IT,NT>[totrecv];
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, World);
        //std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
        //MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        if(myrank == 6){
            int i = 2;
            printf("[RECEIVING] Processor: %d, Chunk: %d\n", myrank, i);
            printf("nnz: %d, mdim: %d, ndim: %d\n",recvprfl[i*3], recvprfl[i*3+1], recvprfl[i*3+2]);
            printf("first tuple: < %lld, %lld, %lf >\n", 
                    get<0>(recvTuples[rdispls[i]]), 
                    get<1>(recvTuples[rdispls[i]]), 
                    get<2>(recvTuples[rdispls[i]]));
            printf("last tuple: < %lld, %lld, %lf >\n", 
                    get<0>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    get<1>(recvTuples[rdispls[i]+recvcnt[i]-1]), 
                    get<2>(recvTuples[rdispls[i]+recvcnt[i]-1]));
            //for(int j = rdispls[i]; j < rdispls[i]+recvcnt[i]; j++){
                //cout << get<0>(recvTuples[j]) << " " << get<1>(recvTuples[j]) << " " << get<2>(recvTuples[j]) << endl;
            //}
            
            SpTuples<IT,NT>xt(recvprfl[i*3], recvprfl[i*3+1], recvprfl[i*3+2], recvTuples+rdispls[i]);
            //vector < tuple < IT, IT, NT > > dumv;
            //dumv.push_back(make_tuple(0,0,1));
            //dumv.push_back(make_tuple(1,1,1));
            //SpTuples<IT,NT>xt(dumv.size(), 2, 2, dumv.data());
            //tuple<IT, IT, NT> * dumv = new tuple<IT,IT,NT>[2];
            //dumv[0] = make_tuple(0,0,1);
            //dumv[1] = make_tuple(1,1,1);
            //SpTuples<IT,NT>xt(2, 2, 2, dumv);
        }

        ////IT mdim, ndim;
        ////LocalDim(nrows, ncols, mdim, ndim);
        ////cout << "mdim , ndim : "<< mdim << " , " << ndim << endl;
        ////SpTuples<IT, NT>spTuples3d(recvTuples.size(), mdim, ndim, recvTuples.data());
        ////cout << "After SpTuples" << endl;
        ////DER * localm3d = new DER(spTuples3d, false);

        //DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
        //MPI_Type_free(&MPI_tuple);
        //datasize = totrecv;
        //return recvTuples;
        return;
    }
    
    
    template <class IT, class NT, class DER>
    SpParMat<IT, NT, DER> SpParMat3D<IT,NT,DER>::Convert2D()
    {
        int nprocs = commGrid3D->GetSize();
        IT total_m = getnrow();
        IT total_n = getncol();
    
        std::shared_ptr<CommGrid> grid2d;
        grid2d.reset(new CommGrid(commGrid3D->GetWorld(), 0, 0));
        
        SpParMat<IT, NT, DER> A2D (grid2d);
        std::vector< std::vector < std::tuple<IT,IT,NT> > > data(nprocs);
        DER* spSeq = layermat->seqptr(); // local submatrix
        
        IT locsize = 0;
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            
            IT gcol = colit.colid(); //+ localColStart2d;
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid(); //+ localRowStart2d;
                NT val = nzit.value();
                
                IT lrow2d, lcol2d;
                int owner = A2D.Owner(total_m, total_n, grow, gcol, lrow2d, lcol2d);
                data[owner].push_back(std::make_tuple(lrow2d,lcol2d,val));
                locsize++;
                
            }
        }
        A2D.SparseCommon(data, locsize, total_m, total_n, maximum<NT>());
         
    }
}
