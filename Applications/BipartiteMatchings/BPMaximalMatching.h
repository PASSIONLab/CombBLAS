#ifndef BP_MAXIMAL_MATCHING_H
#define BP_MAXIMAL_MATCHING_H

#include "CombBLAS/CombBLAS.h"
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <limits>
#include "Utility.h"
#include "MatchingDefs.h"

#define NO_INIT 0
#define GREEDY 1
#define KARP_SIPSER 2
#define DMD 3
MTRand GlobalMT(123); // for reproducible result

namespace combblas {

// This is not tested with CSC yet
// TODO: test with CSC and Setting SPA (similar to Weighted Greedy)
template <typename Par_DCSC_Bool, typename IT>
void MaximalMatching(Par_DCSC_Bool & A, Par_DCSC_Bool & AT, FullyDistVec<IT, IT>& mateRow2Col,
            FullyDistVec<IT, IT>& mateCol2Row, FullyDistVec<IT, IT>& degColRecv, int type, bool rand=true)
{

	typedef VertexTypeML < IT, IT> VertexType;
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    FullyDistVec<IT, IT> degCol = degColRecv;
    
    //unmatched row and column vertices
    FullyDistSpVec<IT, IT> unmatchedRow(mateRow2Col, [](IT mate){return mate==-1;});
    FullyDistSpVec<IT, IT> degColSG(A.getcommgrid(), A.getncol());
    //FullyDistVec<IT, IT> degCol(A.getcommgrid());
    //A.Reduce(degCol, Column, plus<IT>(), static_cast<IT>(0)); // Reduce is not multithreaded
    
 
    FullyDistSpVec<IT, VertexType> unmatchedCol(A.getcommgrid(), A.getncol());
    // every veretx is unmatched. keep non-isolated vertices
    unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, degCol,
                                            [](VertexType vtx, IT deg){return VertexType();},
                                            [](VertexType vtx, IT deg){return deg>0;},
                                            true, VertexType());
    
    
    FullyDistSpVec<IT, VertexType> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<IT, IT> fringeRow2(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<IT, VertexType> deg1Col(A.getcommgrid(), A.getncol());
   
    
    IT curUnmatchedCol = unmatchedCol.getnnz();
    IT curUnmatchedRow = unmatchedRow.getnnz();
    IT newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    std::vector<std::vector<double> > timing;
    
#ifdef DETAIL_STATS
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        unmatchedCol.ApplyInd([](VertexType vtx, IT idx){return VertexType(idx,idx);});
        if(type==DMD)
        {
            unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, degCol,
                                                    [](VertexType vtx, IT deg){return VertexType(vtx.parent,deg);},
                                                    [](VertexType vtx, IT deg){return true;},
                                                    false, VertexType());
        }
        else if(rand)
        {
            unmatchedCol.Apply([](VertexType vtx){return VertexType(vtx.parent, static_cast<IT>((GlobalMT.rand() * 9999999)+1));});
        }
        
        // ======================== step1: One step of BFS =========================
        std::vector<double> times;
        double t1 = MPI_Wtime();
        if(type==GREEDY)
        {
            SpMV<Select2ndMinSR<bool, VertexType>>(A, unmatchedCol, fringeRow, false);
        }
        else if(type==DMD)
        {
            SpMV<Select2ndMinSR<bool, VertexType>>(A, unmatchedCol, fringeRow, false);
        }
        else //(type==KARP_SIPSER)
        {
            deg1Col = EWiseApply<VertexType>(unmatchedCol, degCol,
                                            [](VertexType vtx, IT deg){return vtx;},
                                            [](VertexType vtx, IT deg){return deg==1;},
                                            false, VertexType());

            if(deg1Col.getnnz()>9)
                SpMV<Select2ndMinSR<bool, VertexType>>(A, deg1Col, fringeRow, false);
            else
                SpMV<Select2ndMinSR<bool, VertexType>>(A, unmatchedCol, fringeRow, false);
        }
        // Remove matched row vertices
        fringeRow = EWiseApply<VertexType>(fringeRow, mateRow2Col,
                                            [](VertexType vtx, IT mate){return vtx;},
                                            [](VertexType vtx, IT mate){return mate==-1;},
                                            false, VertexType());
        
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // ======================== step2: Update matching  =========================
        
        fringeRow2  = EWiseApply<IT>(fringeRow, mateRow2Col,
                                          [](VertexType vtx, IT mate){return vtx.parent;},
                                          [](VertexType vtx, IT mate){return true;},
                                          false, VertexType());
        
        FullyDistSpVec<IT, IT> newMatchedCols = fringeRow2.Invert(A.getncol());
        FullyDistSpVec<IT, IT> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](IT mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](IT mate){return mate==-1;});
        
        if(type!=GREEDY)
        {
            // update degree
            newMatchedRows.Apply([](IT val){return 1;}); // needed if the matrix is Boolean since the SR::multiply isn't called
            SpMV< SelectPlusSR<bool, IT>>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
            // subtract degree of column vertices
            degCol.EWiseApply(degColSG,
                              [](IT old_deg, IT new_deg){return old_deg-new_deg;},
                              [](IT old_deg, IT new_deg){return true;},
                              false, static_cast<IT>(0));
            // remove isolated vertices
            unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, degCol,
                                                    [](VertexType vtx, IT deg){return vtx;},
                                                    [](VertexType vtx, IT deg){return deg>0;},
                                                    false, VertexType());
        }
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
        timing.push_back(times);
#ifdef DETAIL_STATS
        if(myrank == 0)
        {
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
#endif
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    IT cardinality = mateRow2Col.Count([](IT mate){return mate!=-1;});
    std::vector<double> totalTimes(timing[0].size(),0);
    for(int i=0; i<timing.size(); i++)
    {
        for(int j=0; j<timing[i].size(); j++)
        {
            totalTimes[j] += timing[i][j];
        }
    }

    
    if(myrank == 0)
    {
#ifdef DETAIL_STATS
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
        cout << "==========================================================\n";
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
            printf("%12.5lf ", totalTimes[i]);
        cout << endl;
#endif
        std::cout << "****** maximal matching runtime ********\n";
        std::cout << "nprocesses nthreads ncores algorithm Unmatched-Rows  Cardinality Total Time***\n";
        std::cout << nprocs << " " << nthreads << " " << nprocs * nthreads << " ";
        if(type == DMD) std::cout << "DMD";
        else if(type == GREEDY) std::cout << "Greedy";
        else if(type == KARP_SIPSER) std::cout << "Karp-Sipser";
        if(rand && (type == KARP_SIPSER || type == GREEDY) ) std::cout << "-rand";
        std::cout << " ";
        printf("%lld    %lld     %lf\n", curUnmatchedRow, cardinality, totalTimes.back());
        std::cout << "-------------------------------------------------------\n\n";
    }
    //isMatching(mateCol2Row, mateRow2Col);
}



// Special version of the greedy algorithm (works for both CSC (multithreaded) and DCSC)
// That uses WeightMaxSR semiring
// TODO: check if this is 1/2 approx of the weighted matching (probably no)
// TODO: should we remove degCol?
// TODO: can be merged with the generalized MaximalMatching
template <typename Par_MAT_Double, typename IT>
void WeightedGreedy(Par_MAT_Double & A, FullyDistVec<IT, IT>& mateRow2Col,
					 FullyDistVec<IT, IT>& mateCol2Row, FullyDistVec<IT, IT>& degCol)
{
	
	typedef VertexTypeML < IT, double> VertexType;
    int nthreads=1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    PreAllocatedSPA<VertexType> SPA(A.seq(), nthreads*4);
	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	
    
	//unmatched row and column vertices
	FullyDistSpVec<IT, IT> unmatchedRow(mateRow2Col, [](IT mate){return mate==-1;});

	
 
	FullyDistSpVec<IT, VertexType> unmatchedCol(A.getcommgrid(), A.getncol());
	// every veretx is unmatched. keep non-isolated vertices
	unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, degCol,
											[](VertexType vtx, IT deg){return VertexType();},
											[](VertexType vtx, IT deg){return deg>0;},
											true, VertexType());
	
	
	FullyDistSpVec<IT, VertexType> fringeRow(A.getcommgrid(), A.getnrow());
	FullyDistSpVec<IT, IT> fringeRow2(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<IT, VertexType> fringeRow3(A.getcommgrid(), A.getnrow());
	
	IT curUnmatchedCol = unmatchedCol.getnnz();
	IT curUnmatchedRow = unmatchedRow.getnnz();
	IT newlyMatched = 1; // ensure the first pass of the while loop
	int iteration = 0;
	
	std::vector<std::vector<double> > timing;
	
#ifdef DETAIL_STATS
	if(myrank == 0)
	{
		cout << "=======================================================\n";
		cout << "@@@@@@ Number of processes: " << nprocs << endl;
		cout << "=======================================================\n";
		cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
		cout << "=======================================================\n";
	}
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	
	
	while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
	{
		// anything is fine in the second argument
		unmatchedCol.ApplyInd([](VertexType vtx, IT idx){return VertexType(idx,idx);});
		
		
		// ======================== step1: One step of BFS =========================
		std::vector<double> times;
		double t1 = MPI_Wtime();
        
		SpMV<WeightMaxMLSR<double, VertexType>>(A, unmatchedCol, fringeRow, false, SPA);
		
		// Remove matched row vertices
		fringeRow = EWiseApply<VertexType>(fringeRow, mateRow2Col,
											[](VertexType vtx, IT mate){return vtx;},
											[](VertexType vtx, IT mate){return mate==-1;},
											false, VertexType());
		
		if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
		// ===========================================================================
		
		
		// ======================== step2: Update matching  =========================
		
		fringeRow2  = EWiseApply<IT>(fringeRow, mateRow2Col,
										  [](VertexType vtx, IT mate){return vtx.parent;},
										  [](VertexType vtx, IT mate){return true;},
										  false, VertexType());
		
		FullyDistSpVec<IT, IT> newMatchedCols = fringeRow2.Invert(A.getncol());
		FullyDistSpVec<IT, IT> newMatchedRows = newMatchedCols.Invert(A.getnrow());
		mateCol2Row.Set(newMatchedCols);
		mateRow2Col.Set(newMatchedRows);
		if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
		// ===========================================================================
		
		
		// =============== step3: Update unmatched columns and rows =================
		
		unmatchedRow.Select(mateRow2Col, [](IT mate){return mate==-1;});
		unmatchedCol.Select(mateCol2Row, [](IT mate){return mate==-1;});
		if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
		// ===========================================================================
		
		
		++iteration;
		newlyMatched = newMatchedCols.getnnz();
		times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
		timing.push_back(times);
#ifdef DETAIL_STATS
		if(myrank == 0)
		{
			printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
		}
#endif
		curUnmatchedCol = unmatchedCol.getnnz();
		curUnmatchedRow = unmatchedRow.getnnz();
		MPI_Barrier(MPI_COMM_WORLD);
		
	}
    
	IT cardinality = mateRow2Col.Count([](IT mate){return mate!=-1;});
	std::vector<double> totalTimes(timing[0].size(),0);
	for(int i=0; i<timing.size(); i++)
	{
		for(int j=0; j<timing[i].size(); j++)
		{
			totalTimes[j] += timing[i][j];
		}
	}
	
	
	if(myrank == 0)
	{
#ifdef DETAIL_STATS
		cout << "==========================================================\n";
		cout << "\n================individual timings =======================\n";
		cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
		cout << "==========================================================\n";
		for(int i=0; i<timing.size(); i++)
		{
			for(int j=0; j<timing[i].size(); j++)
			{
				printf("%12.5lf ", timing[i][j]);
			}
			cout << endl;
		}
		
		cout << "-------------------------------------------------------\n";
		for(int i=0; i<totalTimes.size(); i++)
			printf("%12.5lf ", totalTimes[i]);
		cout << endl;
#endif
#ifdef TIMING
		std::cout << "****** maximal matching runtime ********\n";
		std::cout << "Unmatched-Rows  Cardinality Total Time***\n";
		printf("%lld    %lld     %lf\n", curUnmatchedRow, cardinality, totalTimes.back());
		std::cout << "-------------------------------------------------------\n\n";
#endif
	}
	//isMatching(mateCol2Row, mateRow2Col);
}




template <class Par_DCSC_Bool, class IT, class NT>
bool isMaximalmatching(Par_DCSC_Bool & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row)
{
	typedef VertexTypeML < IT, IT> VertexType;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    FullyDistSpVec<IT, IT> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<IT, IT> fringeCol(A.getcommgrid(), A.getncol());
    FullyDistSpVec<IT, IT> unmatchedRow(mateRow2Col, [](IT mate){return mate==-1;});
    FullyDistSpVec<IT, IT> unmatchedCol(mateCol2Row, [](IT mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
	
	
    SpMV<Select2ndMinSR<bool, VertexType>>(A, unmatchedCol, fringeRow, false);
    fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (IT) -1);
    if(fringeRow.getnnz() != 0)
    {
        if(myrank == 0)
            std::cout << "Not maximal matching!!\n";
        return false;
    }
	
    Par_DCSC_Bool tA = A;
    tA.Transpose();
    SpMV<Select2ndMinSR<bool, VertexType>>(tA, unmatchedRow, fringeCol, false);
    fringeCol = EWiseMult(fringeCol, mateCol2Row, true, (IT) -1);
    if(fringeCol.getnnz() != 0)
    {
        if(myrank == 0)
            std::cout << "Not maximal matching**!!\n";
        return false;
    }
    return true;
}

}

#endif

