/*
 * Dumps the largest connected component of a given graph to a file with the 
 * given filename as prefix and .cc as suffix
 * */
#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "../CC.h"

using namespace std;
using namespace combblas;

/**
 ** Connected components based on Awerbuch-Shiloach algorithm
 **/


class Dist
{
public:
    typedef SpDCCols < int64_t, double > DCCols;
    typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
};



int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./cc --infile-type <mm|triples> --infile <FILENAME_MATRIX_MARKET> --outfile <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "--infile-type <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples. default:mm)\n";
            cout << "--base <BASE OF MATRIX MARKET> (default:1)\n";
        }
        MPI_Finalize();
        return -1;
    }
    {
        string ifilename = "";
        string ofilename = "";
        int base = 1;
        int randpermute = 1;
        bool isMatrixMarket = true;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"--infile-type")==0)
            {
                string ifiletype = string(argv[i+1]);
                if(ifiletype == "triples") isMatrixMarket = false;
            }
            else if (strcmp(argv[i],"--infile")==0)
            {
                ifilename = string(argv[i+1]);
                if(myrank == 0) printf("Input filename: %s\n",ifilename.c_str());
            }
            else if (strcmp(argv[i],"--outfile")==0)
            {
                ofilename = string(argv[i+1]);
                if(myrank == 0) printf("Output filename: %s\n",ofilename.c_str());
            }
            else if (strcmp(argv[i],"--base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
            else if (strcmp(argv[i],"--rand")==0)
            {
                randpermute = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nRandomly permute the matrix? (1 or 0):%d",randpermute);
            }
        }

		typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        
        double tIO = MPI_Wtime();
        Dist::MPI_DCCols A(MPI_COMM_WORLD);	// construct object
        Dist::MPI_DCCols M(MPI_COMM_WORLD);	// construct object
        
        if(isMatrixMarket)
            A.ParallelReadMM(ifilename, base, maximum<double>());
        else
            A.ReadGeneralizedTuples(ifilename,  maximum<double>());
        A.PrintInfo();
		M = Dist::MPI_DCCols(A);
        
        Dist::MPI_DCCols AT = A;
        AT.Transpose();
        if(!(AT == A))
        {
            SpParHelper::Print("Symmatricizing an unsymmetric input matrix.\n");
            A += AT;
        }
        A.PrintInfo();
        
        
        ostringstream outs;
        outs << "File Read time: " << MPI_Wtime() - tIO << endl;
        SpParHelper::Print(outs.str());
        
        if(randpermute && isMatrixMarket) // no need for GeneralizedTuples
        {
            // randomly permute for load balance
            if(A.getnrow() == A.getncol())
            {
                FullyDistVec<int64_t, int64_t> p( A.getcommgrid());
                p.iota(A.getnrow(), 0);
                //p.RandPerm();
                (A)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
        }
        
        double t1 = MPI_Wtime();
        int64_t nCC = 0;
        FullyDistVec<int64_t, int64_t> ccLabels = CC(A, nCC);
        
        double t2 = MPI_Wtime();
        //string outname = ifilename + ".components";
        //ccLabels.ParallelWrite(outname, base);
        int64_t nclusters = ccLabels.Reduce(maximum<int64_t>(), (int64_t) 0 ) ;
        nclusters ++; // because of zero based indexing for clusters

        stringstream s2;
        s2 << "Number of connected components: " << nclusters << endl;
        s2 << "Total time: " << (t2-t1) << endl;
        s2 <<  "=================================================\n" << endl ;
        SpParHelper::Print(s2.str());
        
        /* Find the largest connected component */
        std::vector<int64_t> ccLabelsLocal = ccLabels.GetLocVec();
        int64_t* ccSizesLocal = new int64_t[nclusters];
        int64_t* ccSizesGlobal = new int64_t[nclusters];
        memset(ccSizesLocal, 0, sizeof(int64_t) * nclusters);
        memset(ccSizesGlobal, 0, sizeof(int64_t) * nclusters);

#ifdef THREADED
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < nclusters; i++){
            ccSizesLocal[i] = std::count_if( ccLabelsLocal.begin(), ccLabelsLocal.end(), [i](int64_t val){return val == i;});
        }
        MPI_Allreduce(ccSizesLocal, ccSizesGlobal, (int)nclusters, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); // Type casting because MPI does not take count as something other than integer

		int64_t largestCC = 0;
        int64_t largestCCSize = ccSizesGlobal[largestCC];
        if(myrank == 0){
            fprintf(stderr, "LargestCC: %lld, LargestCCSize: %lld\n", largestCC, largestCCSize);
        }

		for(int64_t i = 1; i < nclusters; i++){
            if (ccSizesGlobal[i] > largestCCSize){
                largestCC = i;
                largestCCSize = ccSizesGlobal[i];
                if(myrank == 0){
                    fprintf(stderr, "LargestCC: %lld, LargestCCSize: %lld\n", largestCC, largestCCSize);
                }
            }
        }
		
		//delete ccSizesLocal;
        //delete ccSizesGlobal;

		if(myrank == 0) printf("Largest connected component is %dth component, size %d\n",largestCC, largestCCSize);
		
		FullyDistVec<int64_t,int64_t> isov = ccLabels.FindInds(
		    [largestCC](int64_t val){return val == largestCC;}
		    );
		Dist::MPI_DCCols MCC = M.SubsRef_SR<PTNTBOOL,PTBOOLNT>(isov, isov, false);
        MCC.PrintInfo();
        MCC.ParallelWriteMM(ofilename, true);
    }
    
    MPI_Finalize();
    return 0;
}
