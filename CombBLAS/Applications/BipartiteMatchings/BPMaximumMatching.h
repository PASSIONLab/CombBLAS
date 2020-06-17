#ifndef BP_MAXIMUM_MATCHING_H
#define BP_MAXIMUM_MATCHING_H

#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include "MatchingDefs.h"

namespace combblas {

/**
 * Create a boolean matrix A (not necessarily a permutation matrix)
 * Input: ri: a dense vector (actual values in FullyDistVec should be IT)
 *        ncol: number of columns in the output matrix A
 * Output: a boolean matrix A with m=size(ri) and n=ncol (input)
 and  A[k,ri[k]]=1
 * This can be done by Matlab like constructor, no?
 */
template <class IT, class DER>
SpParMat<IT, bool, DER> PermMat (const FullyDistVec<IT,IT> & ri, const IT ncol)
{
    
    IT procsPerRow = ri.commGrid->GetGridCols();	// the number of processor in a row of processor grid
    IT procsPerCol = ri.commGrid->GetGridRows();	// the number of processor in a column of processor grid
    
    
    IT global_nrow = ri.TotalLength();
    IT global_ncol = ncol;
    IT m_perprocrow = global_nrow / procsPerRow;
    IT n_perproccol = global_ncol / procsPerCol;
    
    
    // The indices for FullyDistVec are offset'd to 1/p pieces
    // The matrix indices are offset'd to 1/sqrt(p) pieces
    // Add the corresponding offset before sending the data
    
    std::vector< std::vector<IT> > rowid(procsPerRow); // rowid in the local matrix of each vector entry
    std::vector< std::vector<IT> > colid(procsPerRow); // colid in the local matrix of each vector entry
    
    IT locvec = ri.arr.size();	// nnz in local vector
    IT roffset = ri.RowLenUntil(); // the number of vector elements in this processor row before the current processor
    for(typename std::vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
    {
        if(ri.arr[i]>=0 && ri.arr[i]<ncol) // this specialized for matching. TODO: make it general purpose by passing a function
        {
            IT rowrec = (n_perproccol!=0) ? std::min(ri.arr[i] / n_perproccol, procsPerRow-1) : (procsPerRow-1);
            // ri's numerical values give the colids and its local indices give rowids
            rowid[rowrec].push_back( i + roffset);
            colid[rowrec].push_back(ri.arr[i] - (rowrec * n_perproccol));
        }
        
    }
    
    
    
    int * sendcnt = new int[procsPerRow];
    int * recvcnt = new int[procsPerRow];
    for(IT i=0; i<procsPerRow; ++i)
    {
        sendcnt[i] = rowid[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, ri.commGrid->GetRowWorld()); // share the counts
    
    int * sdispls = new int[procsPerRow]();
    int * rdispls = new int[procsPerRow]();
    partial_sum(sendcnt, sendcnt+procsPerRow-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+procsPerRow-1, rdispls+1);
    IT p_nnz = accumulate(recvcnt,recvcnt+procsPerRow, static_cast<IT>(0));
    
    
    IT * p_rows = new IT[p_nnz];
    IT * p_cols = new IT[p_nnz];
    IT * senddata = new IT[locvec];
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
        std::vector<IT>().swap(rowid[i]);	// clear memory of rowid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
        std::vector<IT>().swap(colid[i]);	// clear memory of colid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    delete [] senddata;
    
    std::tuple<IT,IT,bool> * p_tuples = new std::tuple<IT,IT,bool>[p_nnz];
    for(IT i=0; i< p_nnz; ++i)
    {
        p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
    }
    DeleteAll(p_rows, p_cols);
    
    
    // Now create the local matrix
    IT local_nrow = ri.MyRowLength();
    int my_proccol = ri.commGrid->GetRankInProcRow();
    IT local_ncol = (my_proccol<(procsPerCol-1))? (n_perproccol) : (global_ncol - (n_perproccol*(procsPerCol-1)));
    
    // infer the concrete type SpMat<IT,IT>
    typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
    DER_IT * PSeq = new DER_IT();
    PSeq->Create( p_nnz, local_nrow, local_ncol, p_tuples);		// deletion of tuples[] is handled by SpMat::Create
    
    SpParMat<IT,bool,DER_IT> P (PSeq, ri.commGrid);
    //Par_DCSC_Bool P (PSeq, ri.commGrid);
    return P;
}




/***************************************************************************
// Augment a matching by a set of vertex-disjoint augmenting paths.
// The paths are explored level-by-level similar to the level-synchronous BFS
// This approach is more effecient when we have many short augmenting paths
***************************************************************************/

template <typename IT>
void AugmentLevel(FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row, FullyDistVec<IT, IT>& parentsRow, FullyDistVec<IT, IT>& leaves)
{
    
    IT nrow = mateRow2Col.TotalLength();
    IT ncol = mateCol2Row.TotalLength();
    FullyDistSpVec<IT, IT> col(leaves, [](IT leaf){return leaf!=-1;});
    FullyDistSpVec<IT, IT> row(mateRow2Col.getcommgrid(), nrow);
    FullyDistSpVec<IT, IT> nextcol(col.getcommgrid(), ncol);
    
    while(col.getnnz()!=0)
    {
        
        row = col.Invert(nrow);
        row = EWiseApply<IT>(row, parentsRow,
                                  [](IT root, IT parent){return parent;},
                                  [](IT root, IT parent){return true;},
                                  false, (IT)-1);
        
        col = row.Invert(ncol); // children array
        nextcol = EWiseApply<IT>(col, mateCol2Row,
                                      [](IT child, IT mate){return mate;},
                                      [](IT child, IT mate){return mate!=-1;},
                                      false, (IT)-1);
        mateRow2Col.Set(row);
        mateCol2Row.Set(col);
        col = nextcol;
    }
}


/***************************************************************************
// Augment a matching by a set of vertex-disjoint augmenting paths.
// An MPI processor is responsible for a complete path.
// This approach is more effecient when we have few long augmenting paths
// We used one-sided MPI. Any PGAS language should be fine as well.
// This function is not thread safe, hence multithreading is not used here
 ***************************************************************************/

template <typename IT>
void AugmentPath(FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row,FullyDistVec<IT, IT>& parentsRow, FullyDistVec<IT, IT>& leaves)
{
    MPI_Win win_mateRow2Col, win_mateCol2Row, win_parentsRow;
    MPI_Win_create((IT*)mateRow2Col.GetLocArr(), mateRow2Col.LocArrSize() * sizeof(IT), sizeof(IT), MPI_INFO_NULL, mateRow2Col.commGrid->GetWorld(), &win_mateRow2Col);
    MPI_Win_create((IT*)mateCol2Row.GetLocArr(), mateCol2Row.LocArrSize() * sizeof(IT), sizeof(IT), MPI_INFO_NULL, mateCol2Row.commGrid->GetWorld(), &win_mateCol2Row);
    MPI_Win_create((IT*)parentsRow.GetLocArr(), parentsRow.LocArrSize() * sizeof(IT), sizeof(IT), MPI_INFO_NULL, parentsRow.commGrid->GetWorld(), &win_parentsRow);
    
    
    IT* leaves_ptr = (IT*) leaves.GetLocArr();
    //MPI_Win_fence(0, win_mateRow2Col);
    //MPI_Win_fence(0, win_mateCol2Row);
    //MPI_Win_fence(0, win_parentsRow);
    
    IT row, col=100, nextrow;
    int owner_row, owner_col;
    IT locind_row, locind_col;
    int myrank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    
    for(IT i=0; i<leaves.LocArrSize(); i++)
    {
        int depth=0;
        row = *(leaves_ptr+i);
        while(row != - 1)
        {
            
            owner_row = mateRow2Col.Owner(row, locind_row);
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_parentsRow);
            MPI_Get(&col, 1, MPIType<IT>(), owner_row, locind_row, 1, MPIType<IT>(), win_parentsRow);
            MPI_Win_unlock(owner_row, win_parentsRow);
            
            owner_col = mateCol2Row.Owner(col, locind_col);
            MPI_Win_lock(MPI_LOCK_SHARED, owner_col, 0, win_mateCol2Row);
            MPI_Fetch_and_op(&row, &nextrow, MPIType<IT>(), owner_col, locind_col, MPI_REPLACE, win_mateCol2Row);
            MPI_Win_unlock(owner_col, win_mateCol2Row);
            
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_mateRow2Col);
            MPI_Put(&col, 1, MPIType<IT>(), owner_row, locind_row, 1, MPIType<IT>(), win_mateRow2Col);
            MPI_Win_unlock(owner_row, win_mateRow2Col); // we need this otherwise col might get overwritten before communication!
            row = nextrow;
            
        }
    }
    
    //MPI_Win_fence(0, win_mateRow2Col);
    //MPI_Win_fence(0, win_mateCol2Row);
    //MPI_Win_fence(0, win_parentsRow);
    
    MPI_Win_free(&win_mateRow2Col);
    MPI_Win_free(&win_mateCol2Row);
    MPI_Win_free(&win_parentsRow);
}





// Maximum cardinality matching
// Output: mateRow2Col and mateRow2Col
template <typename IT, typename NT,typename DER>
void maximumMatching(SpParMat < IT, NT, DER > & A, FullyDistVec<IT, IT>& mateRow2Col,
                     FullyDistVec<IT, IT>& mateCol2Row, bool prune=true, bool randMM = false, bool maximizeWeight = false)
{
	
	typedef VertexTypeMM <IT> VertexType;
	
    int nthreads=1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    PreAllocatedSPA<VertexType> SPA(A.seq(), nthreads*4);
    
    double tstart = MPI_Wtime();
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    IT nrow = A.getnrow();
    IT ncol = A.getncol();
    
    FullyDistSpVec<IT, VertexType> fringeRow(A.getcommgrid(), nrow);
    FullyDistSpVec<IT, IT> umFringeRow(A.getcommgrid(), nrow);
    FullyDistVec<IT, IT> leaves ( A.getcommgrid(), ncol, (IT) -1);
    
    std::vector<std::vector<double> > timing;
    std::vector<int> layers;
    std::vector<int64_t> phaseMatched;
    double t1, time_search, time_augment, time_phase;
    
    bool matched = true;
    int phase = 0;
    int totalLayer = 0;
    IT numUnmatchedCol;
    
    
    MPI_Win winLeaves;
    MPI_Win_create((IT*)leaves.GetLocArr(), leaves.LocArrSize() * sizeof(IT), sizeof(IT), MPI_INFO_NULL, A.getcommgrid()->GetWorld(), &winLeaves);
    
    
    while(matched)
    {
        time_phase = MPI_Wtime();
  
        std::vector<double> phase_timing(8,0);
        leaves.Apply ( [](IT val){return (IT) -1;});
        FullyDistVec<IT, IT> parentsRow ( A.getcommgrid(), nrow, (IT) -1);
        FullyDistSpVec<IT, VertexType> fringeCol(A.getcommgrid(), ncol);
        fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row,
                                            [](VertexType vtx, IT mate){return vtx;},
                                            [](VertexType vtx, IT mate){return mate==-1;},
                                            true, VertexType());
        
        
        if(randMM) //select rand
        {
            fringeCol.ApplyInd([](VertexType vtx, IT idx){return VertexType(idx,idx,GlobalMT.rand());});
        }
        else
        {
            fringeCol.ApplyInd([](VertexType vtx, IT idx){return VertexType(idx,idx);});
        }
        
        ++phase;
        numUnmatchedCol = fringeCol.getnnz();
        int layer = 0;
        
        
        time_search = MPI_Wtime();
        while(fringeCol.getnnz() > 0)
        {
            layer++;
            t1 = MPI_Wtime();
            
		    //TODO: think about this semiring
			if(maximizeWeight)
				SpMV<WeightMaxMMSR<NT, VertexType>>(A, fringeCol, fringeRow, false, SPA);
			else
				SpMV<Select2ndMinSR<NT, VertexType>>(A, fringeCol, fringeRow, false, SPA);
            phase_timing[0] += MPI_Wtime()-t1;
			
            
            
            // remove vertices already having parents
            
            t1 = MPI_Wtime();
            fringeRow = EWiseApply<VertexType>(fringeRow, parentsRow,
                                               [](VertexType vtx, IT parent){return vtx;},
                                               [](VertexType vtx, IT parent){return parent==-1;},
                                               false, VertexType());
            
            // Set parent pointer
            parentsRow.EWiseApply(fringeRow,
                                  [](IT dval, VertexType svtx){return svtx.parent;},
                                  [](IT dval, VertexType svtx){return true;},
                                  false, VertexType());
            
            
            umFringeRow = EWiseApply<IT>(fringeRow, mateRow2Col,
                                              [](VertexType vtx, IT mate){return vtx.root;},
                                              [](VertexType vtx, IT mate){return mate==-1;},
                                              false, VertexType());
            
            phase_timing[1] += MPI_Wtime()-t1;
            
            
            IT nnz_umFringeRow = umFringeRow.getnnz(); // careful about this timing
            
            t1 = MPI_Wtime();
            if(nnz_umFringeRow >0)
            {
                /*
                if(nnz_umFringeRow < 25*nprocs)
                {
                    leaves.GSet(umFringeRow,
                                [](IT valRoot, IT idxLeaf){return valRoot;},
                                [](IT valRoot, IT idxLeaf){return idxLeaf;},
                                winLeaves); 
                 // There might be a bug here. It does not return the same output for different number of processes
                 // e.g., check with g7jac200sc.mtx matrix
                }
                else*/
                {
                    FullyDistSpVec<IT, IT> temp1(A.getcommgrid(), ncol);
                    temp1 = umFringeRow.Invert(ncol);
                    leaves.Set(temp1);
                }
            }
            
            phase_timing[2] += MPI_Wtime()-t1;
            
            
            
            
            // matched row vertices in the the fringe
            fringeRow = EWiseApply<VertexType>(fringeRow, mateRow2Col,
                                               [](VertexType vtx, IT mate){return VertexType(mate, vtx.root);},
                                               [](VertexType vtx, IT mate){return mate!=-1;},
                                               false, VertexType());
            
            t1 = MPI_Wtime();
            if(nnz_umFringeRow>0 && prune)
            {
                fringeRow.FilterByVal (umFringeRow,[](VertexType vtx){return vtx.root;}, false);
            }
            double tprune = MPI_Wtime()-t1;
            phase_timing[3] += tprune;
            
            
            // Go to matched column from matched row in the fringe. parent is automatically set to itself.
            t1 = MPI_Wtime();
			fringeCol = fringeRow.Invert(ncol,
										 [](VertexType& vtx, const IT & index){return vtx.parent;},
										 [](VertexType& vtx, const IT & index){return vtx;},
										 [](VertexType& vtx1, VertexType& vtx2){return vtx1;});
            phase_timing[4] += MPI_Wtime()-t1;
			
			
			
			
        }
        time_search = MPI_Wtime() - time_search;
        phase_timing[5] += time_search;
        
        IT numMatchedCol = leaves.Count([](IT leaf){return leaf!=-1;});
        phaseMatched.push_back(numMatchedCol);
        time_augment = MPI_Wtime();
        if (numMatchedCol== 0) matched = false;
        else
        {
            
            if(numMatchedCol < (2* nprocs * nprocs))
                AugmentPath(mateRow2Col, mateCol2Row,parentsRow, leaves);
            else
                AugmentLevel(mateRow2Col, mateCol2Row,parentsRow, leaves);
        }
        time_augment = MPI_Wtime() - time_augment;
        phase_timing[6] += time_augment;
        
        time_phase = MPI_Wtime() - time_phase;
        phase_timing[7] += time_phase;
        timing.push_back(phase_timing);
        totalLayer += layer;
        layers.push_back(layer);
        
    }
    
    
    MPI_Win_free(&winLeaves);
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    double combTime;
    
#ifdef TIMING
    if(myrank == 0)
    {
        std::cout << "****** maximum matching runtime ********\n";
        std::cout << std::endl;
        std::cout << "========================================================================\n";
        std::cout << "                                     BFS Search                       \n";
        std::cout << "===================== ==================================================\n";
        std::cout  << "Phase Layer    Match   SpMV EWOpp CmUqL  Prun CmMC   BFS   Aug   Total\n";
        std::cout << "===================== ===================================================\n";
        
        std::vector<double> totalTimes(timing[0].size(),0);
        int nphases = timing.size();
        for(int i=0; i<timing.size(); i++)
        {
            printf(" %3d  %3d  %8lld   ", i+1, layers[i], phaseMatched[i]);
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                //timing[i][j] /= timing[i].back();
                printf("%.2lf  ", timing[i][j]);
            }
            
            printf("\n");
        }
        
        std::cout << "-----------------------------------------------------------------------\n";
        std::cout  << "Phase Layer   UnMat   SpMV EWOpp CmUqL  Prun CmMC   BFS   Aug   Total \n";
        std::cout << "-----------------------------------------------------------------------\n";
        
        combTime = totalTimes.back();
        printf(" %3d  %3d  %8lld   ", nphases, totalLayer/nphases, numUnmatchedCol);
        for(int j=0; j<totalTimes.size()-1; j++)
        {
            printf("%.2lf  ", totalTimes[j]);
        }
        printf("%.2lf\n", combTime);
    }
#endif
    
    IT nrows=A.getnrow();
    IT matchedRow = mateRow2Col.Count([](IT mate){return mate!=-1;});
#ifdef DETAIL_STATS
    if(myrank==0)
    {
        std::cout << "***Final Maximum Matching***\n";
        std::cout << "***Total-Rows Matched-Rows  Total Time***\n";
        printf("%lld %lld %lf \n",nrows, matchedRow, combTime);
        printf("matched rows: %lld , which is: %lf percent \n",matchedRow, 100*(double)matchedRow/(nrows));
        std::cout << "-------------------------------------------------------\n\n";
    }
#endif
    
}

}

#endif

