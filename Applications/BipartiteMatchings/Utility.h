#ifndef BP_UTILITY_H
#define BP_UTILITY_H

#include "CombBLAS/CombBLAS.h"

namespace combblas {

/*
 Remove isolated vertices and purmute
 */
template <typename PARMAT>
void removeIsolated(PARMAT & A)
{
	
	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	
	
	FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
	FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
	FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
	FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
	FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
	
	A.Reduce(*ColSums, Column, std::plus<int64_t>(), static_cast<int64_t>(0));
	A.Reduce(*RowSums, Row, std::plus<int64_t>(), static_cast<int64_t>(0));
	
	// this steps for general graph
	/*
	 ColSums->EWiseApply(*RowSums, plus<int64_t>()); not needed for bipartite graph
	 nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
	 nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
	 A.operator()(nonisov, nonisov, true);	// in-place permute to save memory
	 */
	
	// this steps for bipartite graph
	nonisoColV = ColSums->FindInds(bind2nd(std::greater<int64_t>(), 0));
	nonisoRowV = RowSums->FindInds(bind2nd(std::greater<int64_t>(), 0));
	delete ColSums;
	delete RowSums;
	
	
	{
		nonisoColV.RandPerm();
		nonisoRowV.RandPerm();
	}
	
	
	int64_t nrows1=A.getnrow(), ncols1=A.getncol(), nnz1 = A.getnnz();
	double avgDeg1 = (double) nnz1/(nrows1+ncols1);
	
	
	A.operator()(nonisoRowV, nonisoColV, true);
	
	int64_t nrows2=A.getnrow(), ncols2=A.getncol(), nnz2 = A.getnnz();
	double avgDeg2 = (double) nnz2/(nrows2+ncols2);
	
	
	if(myrank == 0)
	{
		std::cout << "ncol nrows  nedges deg \n";
		std::cout << nrows1 << " " << ncols1 << " " << nnz1 << " " << avgDeg1 << " \n";
		std::cout << nrows2 << " " << ncols2 << " " << nnz2 << " " << avgDeg2 << " \n";
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	
}




/*
 * Serial: Check the validity of the matching solution;
 we need a better solution using invert
 */
template <class IT, class NT>
bool isMatching(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col)
{
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    for(int i=0; i< mateRow2Col.glen ; i++)
    {
        int t = mateRow2Col[i];
        
        if(t!=-1 && mateCol2Row[t]!=i)
        {
            if(myrank == 0)
                std::cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    
    for(int i=0; i< mateCol2Row.glen ; i++)
    {
        int t = mateCol2Row[i];
        if(t!=-1 && mateRow2Col[t]!=i)
        {
            if(myrank == 0)
                std::cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    return true;
}



template <class IT>
bool CheckMatching(FullyDistVec<IT,IT> & mateRow2Col, FullyDistVec<IT,IT> & mateCol2Row)
{
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int64_t nrow = mateRow2Col.TotalLength();
    int64_t ncol = mateCol2Row.TotalLength();
    FullyDistSpVec<IT,IT> mateRow2ColSparse (mateRow2Col, [](IT mate){return mate!=-1;});
    FullyDistSpVec<IT,IT> mateCol2RowSparse (mateCol2Row, [](IT mate){return mate!=-1;});
    FullyDistSpVec<IT,IT> mateRow2ColInverted = mateRow2ColSparse.Invert(ncol);
    FullyDistSpVec<IT,IT> mateCol2RowInverted = mateCol2RowSparse.Invert(nrow);

    
    
    bool isMatching = false;
    if((mateCol2RowSparse == mateRow2ColInverted) && (mateRow2ColSparse == mateCol2RowInverted))
        isMatching = true;
    else
    {
        SpParHelper::Print("Warning: This is not a matching! Need to check the correctness of the matching (HWPM) code\n");
    }
    
    bool isPerfectMatching = false;
    if((mateRow2ColSparse.getnnz()==nrow) && (mateCol2RowSparse.getnnz() == ncol))
        isPerfectMatching = true;
    return isPerfectMatching;
}


// Gievn a matrix and matching vectors, returns the weight of the matching 
template <class IT, class NT, class DER>
NT MatchingWeight( SpParMat < IT, NT, DER > & A, FullyDistVec<IT,IT> mateRow2Col, FullyDistVec<IT,IT>& mateCol2Row)
{
	
	auto commGrid = A.getcommgrid();
	int myrank=commGrid->GetRank();
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm ColWorld = commGrid->GetColWorld();
	MPI_Comm RowWorld = commGrid->GetRowWorld();
	int nprocs = commGrid->GetSize();
	int pr = commGrid->GetGridRows();
	int pc = commGrid->GetGridCols();
	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol();
	int diagneigh = commGrid->GetComplementRank();
	
	//Information about the matrix distribution
	//Assume that A is an nrow x ncol matrix
	//The local submatrix is an lnrow x lncol matrix
	IT nrows = A.getnrow();
	IT ncols = A.getncol();
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	Dcsc<IT, NT>* dcsc = spSeq->GetDCSC();
	IT lnrow = spSeq->getnrow();
	IT lncol = spSeq->getncol();
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
	// -----------------------------------------------------------
	// replicate mate vectors for mateCol2Row
	// -----------------------------------------------------------
	int xsize = (int)  mateCol2Row.LocArrSize();
	int trxsize = 0;
	MPI_Status status;
	MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	std::vector<IT> trxnums(trxsize);
	MPI_Sendrecv(mateCol2Row.GetLocArr(), xsize, MPIType<IT>(), diagneigh, TRX, trxnums.data(), trxsize, MPIType<IT>(), diagneigh, TRX, World, &status);
	
	
	std::vector<int> colsize(pc);
	colsize[colrank] = trxsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize.data(), 1, MPI_INT, ColWorld);
	std::vector<int> dpls(pc,0);	// displacements (zero initialized pid)
	std::partial_sum(colsize.data(), colsize.data()+pc-1, dpls.data()+1);
	int accsize = std::accumulate(colsize.data(), colsize.data()+pc, 0);
	std::vector<IT> RepMateC2R(accsize);
	MPI_Allgatherv(trxnums.data(), trxsize, MPIType<IT>(), RepMateC2R.data(), colsize.data(), dpls.data(), MPIType<IT>(), ColWorld);
	// -----------------------------------------------------------
	
    /*
    if(myrank==1)
    {
        for(int i=0; i<RepMateC2R.size(); i++)
            cout << RepMateC2R[i] << ",";
    }*/
	
	NT w = 0;
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
	{
		IT lj = colit.colid(); // local numbering
		IT mj = RepMateC2R[lj]; // mate of j
		if(mj >= localRowStart && mj < (localRowStart+lnrow) )
		{
			for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
			{
				IT li = nzit.rowid();
				IT i = li + localRowStart;
				// TODO: use binary search to directly go to mj-th entry if more than 32 nonzero in this column
				if( i == mj)
				{
					w += nzit.value();
                    //cout << myrank<< ":: row: " << i << " column: "<< lj+localColStart << " weight: " <<  nzit.value() << endl;
				}
			}
		}
		
	}

    MPI_Barrier(World);
	MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPIType<NT>(), MPI_SUM, World);
    //MPI_Allreduce(&w, &gw, 1, MPIType<NT>(), MPI_SUM, World);
    //MPI_Reduce(&w, &gw, 1, MPIType<NT>(), MPI_SUM, 0, World);
     //MPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, World);
    //cout << myrank << ": " << gw << endl;
	return w;
}






template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}

}

#endif

