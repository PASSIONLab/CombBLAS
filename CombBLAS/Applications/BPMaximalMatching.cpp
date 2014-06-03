#define DETERMINISTIC
#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP
#endif
#endif
//#define DEBUG


#ifdef _OPENMP
#include <omp.h>
#endif

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
	istringstream iss(s);
	return !(iss >> f >> t).fail();
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


template<typename T>
struct positive : public std::unary_function<T, bool>
{
    bool operator()(const T& x) const
    {
        return (x>0);
    }
};



/**
 * Binary function to prune the previously discovered vertices from the current frontier 
 * When used with EWiseApply(SparseVec V, DenseVec W,...) we get the 'exclude = false' effect of EWiseMult
 **/
struct prunediscovered: public std::binary_function<int64_t, int64_t, int64_t >
{
	int64_t operator()(int64_t x, const int64_t & y) const
	{
		return ( y == -1 ) ? x: -1;
	}
};


/*
 * Check the validity of the matching solution
 */
template <class IT, class NT>
void sanityCheck(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col,
                 SpParMat < int64_t, bool, SpDCCols<int32_t,bool> >& spAdjMat, OptBuf<int32_t, int64_t> & optbuf)
{
    //mateCol2Row.DebugPrint();
    //mateRow2Col.DebugPrint();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   
    /*
    // first check if this is a matching
    for(int i=0; i< mateRow2Col.glen ; i++)
    {
        int t = mateRow2Col[i];
        
        if(t!=-1 && mateCol2Row[t]!=i)
        {
            if(myrank == 0)
                cout << "Error!!\n";
            break;
        }
    }
    
    for(int i=0; i< mateCol2Row.glen ; i++)
    {
        int t = mateCol2Row[i];
        if(t!=-1 && mateRow2Col[t]!=i)
        {
            if(myrank == 0)
                cout << "Error!!\n";
            break;
        }
    }
    */
    
    // Check if this is a maximal matching
    
    FullyDistVec<IT, NT> dvColVertices(spAdjMat.getcommgrid(), spAdjMat.getncol(), (NT) 0);
    FullyDistSpVec<IT, NT> unmatchedCol(dvColVertices);
    unmatchedCol = EWiseMult(unmatchedCol, mateCol2Row, true, (int64_t) -1);
    FullyDistSpVec<IT, NT> fringeRow(spAdjMat.getcommgrid(), spAdjMat.getnrow());
    
    fringeRow = SpMV(spAdjMat, unmatchedCol, optbuf);
    fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
    
    
    FullyDistVec<IT, NT> dvRowVertices(spAdjMat.getcommgrid(), spAdjMat.getnrow(), (NT) 0);
    FullyDistSpVec<IT, NT> unmatchedRow(dvRowVertices);
    unmatchedRow = EWiseMult(unmatchedRow, mateRow2Col, true, (int64_t) -1);
    FullyDistSpVec<IT, NT> fringeCol(spAdjMat.getcommgrid(), spAdjMat.getncol());
    
    spAdjMat.Transpose();
    fringeCol = SpMV(spAdjMat, unmatchedRow, optbuf);
    fringeCol = EWiseMult(fringeCol, mateCol2Row, true, (int64_t) -1);
    
    if(fringeRow.getnnz() != 0 || fringeCol.getnnz() != 0)
    {
        if(myrank == 0)
            cout << "Not maximam matching!!\n";
    }

}




int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./Graph500 <Force,Input> <Scale Forced | Input Name> {FastGen}" << endl;
			cout << "Example: ./Graph500 Force 25 FastGen" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int32_t> SR;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
		typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

		// Declare objects
		PSpMat_Bool A;	
		PSpMat_s32p64 Aeff;
		FullyDistVec<int64_t, int64_t> degrees;	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
        
        FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
        FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
        
		unsigned scale;
		OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits
		bool scramble = false;

		if(string(argv[1]) == string("Input")) // input option
		{
			A.ReadDistribute(string(argv[2]), 0);	// read it from file
			SpParHelper::Print("Read input\n");

			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0 
			delete G;

			//Symmetricize(A);	// A += A';
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	// plus<int64_t> matches the type of the output vector
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;
			A = A(nonisov, nonisov);
			Aeff = PSpMat_s32p64(A);
			A.FreeMemory();
			//SpParHelper::Print("Symmetricized and pruned\n");

			Aeff.OptimizeForGraph500(optbuf);               // Should be called before threading is activated
#ifdef THREADED
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);
#endif
		}
		else if(string(argv[1]) == string("Binary"))
		{
			uint64_t n, m;
			from_string(n,string(argv[3]),std::dec);
			from_string(m,string(argv[4]),std::dec);

			ostringstream outs;
			outs << "Reading " << argv[2] << " with " << n << " vertices and " << m << " edges" << endl;
			SpParHelper::Print(outs.str());
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(argv[2], n, m);
			SpParHelper::Print("Read binary input to distributed edge list\n");

			PermEdges(*DEL);
			SpParHelper::Print("Permuted Edges\n");

			RenameVertices(*DEL);	
			//DEL->Dump32bit("graph_permuted");
			SpParHelper::Print("Renamed Vertices\n");

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_Int64 * G = new PSpMat_Int64(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Int64 Sparse Matrix\n");

			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 

			A =  PSpMat_Bool(*G);			// Convert to Boolean
			delete G;
			int64_t removed  = A.RemoveLoops();

			ostringstream loopinfo;
			loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
			SpParHelper::Print(loopinfo.str());
			A.PrintInfo();

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			delete RowSums;

			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
#ifndef NOPERMUTE
			A(nonisov, nonisov, true);	// in-place permute to save memory
			SpParHelper::Print("Dropped isolated vertices from input\n");	
			A.PrintInfo();
#endif
			Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
			A.FreeMemory();

			//Symmetricize(Aeff);	// A += A';
			//SpParHelper::Print("Symmetricized\n");	
			//A.Dump("graph_symmetric");

			Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);	
#endif
		}
		else 
		{	
			if(string(argv[1]) == string("Force"))	
			{
				scale = static_cast<unsigned>(atoi(argv[2]));
				ostringstream outs;
				outs << "Forcing scale to : " << scale << endl;
				SpParHelper::Print(outs.str());

				if(argc > 3 && string(argv[3]) == string("FastGen"))
				{
					SpParHelper::Print("Using fast vertex permutations; skipping edge permutations (like v2.1)\n");	
					scramble = true;
				}
			}
			else
			{
				SpParHelper::Print("Unknown option\n");
				MPI_Finalize();
				return -1;	
			}
			// this is an undirected graph, so A*x does indeed BFS
			double initiator[4] = {.57, .19, .19, .05};

			double t01 = MPI_Wtime();
			double t02;
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
			if(!scramble)
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR);
				SpParHelper::Print("Generated edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());

				PermEdges(*DEL);
				SpParHelper::Print("Permuted Edges\n");
				//DEL->Dump64bit("edges_permuted");
				//SpParHelper::Print("Dumped\n");

				RenameVertices(*DEL);	// intermediate: generates RandPerm vector, using MemoryEfficientPSort
				SpParHelper::Print("Renamed Vertices\n");
			}
			else	// fast generation
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
				SpParHelper::Print("Generated renamed edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());
			}

			// Start Kernel #1
			MPI_Barrier(MPI_COMM_WORLD);
			double t1 = MPI_Wtime();

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_s32p64_Int * G = new PSpMat_s32p64_Int(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");

			MPI_Barrier(MPI_COMM_WORLD);
			double redts = MPI_Wtime();
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
			MPI_Barrier(MPI_COMM_WORLD);
			double redtf = MPI_Wtime();

			ostringstream redtimeinfo;
			redtimeinfo << "Calculated degrees in " << redtf-redts << " seconds" << endl;
			SpParHelper::Print(redtimeinfo.str());
			A =  PSpMat_Bool(*G);			// Convert to Boolean
			delete G;
			//int64_t removed  = A.RemoveLoops();

			//ostringstream loopinfo;
			//loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
			//SpParHelper::Print(loopinfo.str());
			//A.PrintInfo();
            
            // =========================================================================================================
            // Remove Isolated vertice
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			SpParHelper::Print("Reductions done\n");
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
            
            
            //nonisoRowV.iota(A.getnrow(), 0);  // this is not working! making the randpermutation load imbalanced
            //nonisoColV = ColSums->FindInds(bind2nd(greater<int64_t>(), -1)); // not working either
            //nonisoColV.iota(A.getncol(), 0);  //
            //nonisoRowV.RandPerm();
            //nonisoColV.RandPerm();
            delete ColSums;
            delete RowSums;

			//SpParHelper::Print("Found (and permuted) non-isolated vertices\n");
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
            
#ifndef NOPERMUTE
            
			A(nonisov, nonisov, true);	// in-place permute to save memory
            //A(nonisoRowV, nonisoColV, true); // there is a problem with this in multi-processor !!
			//SpParHelper::Print("Dropped isolated vertices from input\n");
			A.PrintInfo();
#endif
            // =========================================================================================================
            
            
			Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
			A.FreeMemory();
			//Symmetricize(Aeff);	// A += A';   ######  BFS depends on this ????
			//SpParHelper::Print("Symmetricized\n");	

			Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);	
#endif
			//Aeff.PrintInfo();

			MPI_Barrier(MPI_COMM_WORLD);
			double t2=MPI_Wtime();

			ostringstream k1timeinfo;
			k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
			SpParHelper::Print(k1timeinfo.str());
		}

		//Aeff.PrintInfo();
		float balance = Aeff.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		// Now that every remaining vertex is non-isolated, randomly pick ITERS many of them as starting vertices
#ifndef NOPERMUTE
		degrees = degrees(nonisov);	// fix the degrees array too
		degrees.PrintInfo("Degrees array");
#endif
        
        
        double tStart;
        
        {
            
            // find non isolated row and column vertices
            FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
            FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
           
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(Aeff.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(Aeff.getcommgrid());
			Aeff.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
			Aeff.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
			
            FullyDistSpVec<int64_t, int64_t> unmatchedRow(*RowSums, positive<int64_t>());
            FullyDistSpVec<int64_t, int64_t> unmatchedCol(*ColSums, positive<int64_t>());
            unmatchedRow.setNumToInd();
            unmatchedCol.setNumToInd();
            
            delete ColSums;
            delete RowSums;
            
            
            //matching vector (dense)
            FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
            FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
            
            //fringe vector (sparse)
            FullyDistSpVec<int64_t, int64_t> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
            FullyDistSpVec<int64_t, int64_t> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
            
            //row and column indices... dummy vectors used for permutation. #todo: think better alternatives
            FullyDistVec<int64_t, int64_t> dvColVertices;
            dvColVertices.iota(Aeff.getncol(), 0);
            FullyDistVec<int64_t, int64_t> dvRowVertices;
            dvRowVertices.iota(Aeff.getnrow(), 0);
            //FullyDistVec<int64_t, int64_t> dvColVertices(Aeff.getcommgrid(), Aeff.getncol(), (int64_t) 0); // just for
            
           

            MPI_Pcontrol(1,"BFS");
            
            int64_t curUnmatchedCol = unmatchedCol.getnnz();
            int64_t curUnmatchedRow = unmatchedRow.getnnz();
            int64_t newlyMatched = 1; // ensure the first pass of the while loop
            int iteration = 0;
            tStart = MPI_Wtime();
            vector<vector<double> > timing;
            if(myrank == 0)
            {
                cout << "=======================================================\n";
                cout << "@@@@@@ Number of processes: " << nprocs << endl;
                cout << "=======================================================\n";
                cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
                cout << "=======================================================\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
            

            //dvRowVertices.DebugPrint();
            while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
            {
                vector<double> times;
                double t1 = MPI_Wtime();
                // step1: Find adjacent row vertices (col vertices parent, row vertices child)
                //unmatchedCol.DebugPrint();
                fringeRow = SpMV(Aeff, unmatchedCol, optbuf);
                
                // step2: Remove matched row vertices
                fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
                if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
                
                
                
                // step3: Remove duplicate row vertices
                fringeRow = fringeRow.Uniq();
                if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
                
                // step4: Update mateRow2Col with the newly matched row vertices
                mateRow2Col.Set(fringeRow);
                
                // step5: Update mateCol2Row with the newly matched col vertices
                //FullyDistSpVec<int64_t, int64_t> temp = dvRowVertices(fringeRow);
                auto temp = dvRowVertices(fringeRow);
                //mateCol2Row.Set(dvColVertices(fringeRow)); // does not work !!
                mateCol2Row.Set(temp);
                if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
                
                // step6: Update unmatchedCol/unmatchedRow by removing newly matched columns/rows
                unmatchedCol = EWiseMult(unmatchedCol, mateCol2Row, true, (int64_t) -1);
                unmatchedRow = EWiseMult(unmatchedRow, mateRow2Col, true, (int64_t) -1);
                if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
                
                
                ++iteration;
                newlyMatched = fringeRow.getnnz();
                if(myrank == 0)
                {
                    times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
                    timing.push_back(times);
                    cout  << iteration <<  "  " << curUnmatchedRow  <<  "  " << curUnmatchedCol  <<  "  " << newlyMatched <<  "  " << times.back() << endl;
                }
                
                curUnmatchedCol = unmatchedCol.getnnz();
                curUnmatchedRow = unmatchedRow.getnnz();
                MPI_Barrier(MPI_COMM_WORLD);
                
            }
        
        
        
            //mateRow2Col.DebugPrint();
            //mateCol2Row.DebugPrint();
            
            //Check if this is a maximal matching
 
            fringeRow = SpMV(Aeff, unmatchedCol, optbuf);
            fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
            if(fringeRow.getnnz() != 0)
            {
                if(myrank == 0)
                    cout << "Not maximal matching!!\n";
            }
            //Aeff.Transpose();
            //fringeCol = SpMV(Aeff, unmatchedRow, optbuf);
            //fringeCol = EWiseMult(fringeCol, mateCol2Row, true, (int64_t) -1);
            
            // print statistics
            if(myrank == 0)
            {
                cout << "============================================================\n";
                cout << "\n================individual timings =========================\n";
                 cout  << "SpMV  |  Uniq   |  Permute   |  Update matching  |  Total "<< endl;
                cout << "============================================================\n";
                
                vector<double> totalTimes(timing[0].size(),0);
                for(int i=0; i<timing.size(); i++)
                {
                    for(int j=0; j<timing[i].size(); j++)
                    {
                        totalTimes[j] += timing[i][j];
                        cout << timing[i][j] << "  ";
                    }
                    cout << endl;
                }
                
                cout << "=================== total timing ===========================\n";
                for(int i=0; i<totalTimes.size(); i++)
                    cout<<totalTimes[i] << " ";
                cout << endl;
            }
        
    }
    
		

        
        
        /// stats
        double tEnd = MPI_Wtime();
        //FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
		//parentsp.Apply(myset<int64_t>(1));
		// we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
		int64_t nedges; //= EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
		

		MPI_Pcontrol(-1,"BFS");
		ostringstream outnew;
		outnew << endl << "==============================================================" << endl;
		outnew << "Number of edges traversed= " << nedges;
		outnew << " BFS time: " << tEnd-tStart << " seconds";
		outnew << " MTEPS: " << static_cast<double>(nedges) / (tEnd-tStart) / 1000000.0 << endl;
		SpParHelper::Print(outnew.str());

	}
	MPI_Finalize();
	return 0;
}



/*
 * Check the validity of the matching solution
 */
/*
template <class IT, class NT>
void sanityCheck(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col,
                 SpParMat < int64_t, bool, SpDCCols<int32_t,bool> >& spAdjMat, OptBuf<int32_t, int64_t> & optbuf)
{
    //mateCol2Row.DebugPrint();
    //mateRow2Col.DebugPrint();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    // first check if this is a matching
    for(int i=0; i< mateRow2Col.glen ; i++)
    {
        int t = mateRow2Col[i];
        
        if(t!=-1 && mateCol2Row[t]!=i)
        {
            if(myrank == 0)
                cout << "Error!!\n";
            break;
        }
    }
    
    for(int i=0; i< mateCol2Row.glen ; i++)
    {
        int t = mateCol2Row[i];
        if(t!=-1 && mateRow2Col[t]!=i)
        {
            if(myrank == 0)
                cout << "Error!!\n";
            break;
        }
    }
    
    
    // Check if this is a maximal matching
    
    FullyDistVec<IT, NT> dvColVertices(spAdjMat.getcommgrid(), spAdjMat.getncol(), (NT) 0);
    FullyDistSpVec<IT, NT> unmatchedCol(dvColVertices);
    unmatchedCol = EWiseMult(unmatchedCol, mateCol2Row, true, (int64_t) -1);
    FullyDistSpVec<IT, NT> fringeRow(spAdjMat.getcommgrid(), spAdjMat.getnrow());
    
    fringeRow = SpMV(spAdjMat, unmatchedCol, optbuf);
    fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
    
    
    FullyDistVec<IT, NT> dvRowVertices(spAdjMat.getcommgrid(), spAdjMat.getnrow(), (NT) 0);
    FullyDistSpVec<IT, NT> unmatchedRow(dvRowVertices);
    unmatchedRow = EWiseMult(unmatchedRow, mateRow2Col, true, (int64_t) -1);
    FullyDistSpVec<IT, NT> fringeCol(spAdjMat.getcommgrid(), spAdjMat.getncol());
    
    spAdjMat.Transpose();
    fringeCol = SpMV(spAdjMat, unmatchedRow, optbuf);
    fringeCol = EWiseMult(fringeCol, mateCol2Row, true, (int64_t) -1);
    
    if(fringeRow.getnnz() != 0 || fringeCol.getnnz() != 0)
    {
        if(myrank == 0)
            cout << "Not maximam matching!!\n";
    }
    
}
*/
