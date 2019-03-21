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
        colsplit = csplit;
        typedef typename DER::LocalIT LIT;
        auto commGrid2D = A2D.getcommgrid();
        int nprocs = commGrid2D->GetSize();
        commGrid3D.reset(new CommGrid3D(commGrid2D->GetWorld(), nlayers, 0, 0));
        
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
        std::vector<int> tsendcnt(nprocs,0);
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            IT gcol = colit.colid() + localColStart2d;
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid() + localRowStart2d;
                
                //, nzit.value();
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
        
        std::vector<std::tuple<IT,IT,NT>> recvTuples = ExchangeData(sendTuples, commGrid2D->GetWorld());
        
     
        IT mdim, ndim;
        LocalDim(nrows, ncols, mdim, ndim);
        SpTuples<IT, NT>spTuples3d(recvTuples.size(), mdim, ndim, recvTuples.data());
        DER * localm3d = new DER(spTuples3d, false);
        // not layer SpParMat
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer;
        layermat = new SpParMat<IT, NT, DER>(localm3d, commGridLayer);
    }
    
    template <class IT, class NT,class DER>
    template <typename LIT>
    int SpParMat3D<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const
    {
        
        // first map to Layer 0 and then split
        std::shared_ptr<CommGrid> commGridLayer = commGrid3D->commGridLayer; // CommGrid for my layer
        int procrows = commGridLayer->GetGridRows();
        int proccols = commGridLayer->GetGridCols();
        int nlayers = commGrid3D->gridLayers;
        
        IT m_perproc_L0 = total_m / procrows;
        IT n_perproc_L0 = total_n / proccols;
        
        int procrow_L0; // within a layer
        if(m_perproc_L0 != 0)
        {
            procrow_L0 = std::min(static_cast<int>(grow / m_perproc_L0), procrows-1);
        }
        else    // all owned by the last processor row
        {
            procrow_L0 = procrows -1;
        }
        int proccol_L0;
        if(n_perproc_L0 != 0)
        {
            proccol_L0 = std::min(static_cast<int>(gcol / n_perproc_L0), proccols-1);
        }
        else
        {
            proccol_L0 = proccols-1;
        }
        
        IT lrow_L0 = grow - (procrow_L0 * m_perproc_L0);
        IT lcol_L0 = gcol - (proccol_L0 * n_perproc_L0);
        int layer;
        // next, split and scatter
        if(colsplit)
        {
            IT n_perproc;
            if(proccol_L0 < commGrid3D->gridCols-1)
            {
                n_perproc = n_perproc_L0 / nlayers;
            }
            else
            {
                n_perproc = (total_n - (n_perproc_L0 * proccol_L0)) / nlayers;
            }
            if(n_perproc != 0)
            {
                layer = std::min(static_cast<int>(lcol_L0 / n_perproc), nlayers-1);
            }
            else
                layer = nlayers-1;
            
            lrow = lrow_L0;
            lcol = lcol_L0 - (layer * n_perproc);
        }
        else
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
        int proccol_layer = proccol_L0;
        int procrow_layer = procrow_L0;
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
    std::vector<std::tuple<IT,IT,NT>>  ExchangeData(std::vector<std::vector<std::tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World)
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
        std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
        MPI_Type_free(&MPI_tuple);
        return recvTuples;
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
