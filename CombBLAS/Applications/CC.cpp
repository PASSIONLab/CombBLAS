/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
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
#include "../CombBLAS.h"

using namespace std;

/**
 ** Connected components based on Awerbuch-Shiloach algorithm
 **/


class Dist
{
public:
    typedef SpDCCols < int64_t, double > DCCols;
    typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
};


template <typename T1, typename T2>
struct Select2ndMinSR
{
    typedef typename promote_trait<T1,T2>::T_promote T_promote;
    static T_promote id(){ return numeric_limits<T_promote>::max(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T_promote multiply(const T1 & arg1, const T2 & arg2)
    {
        return static_cast<T_promote> (arg2);
    }
    
    static void axpy(const T1 a, const T2 & x, T_promote & y)
    {
        y = add(y, multiply(a, x));
    }
};





FullyDistVec<int64_t,short> StarCheck(const Dist::MPI_DCCols & A, FullyDistVec<int64_t, int64_t> & father)
{
    // FullyDistVec doesn't support "bool" due to crippled vector<bool> semantics
    //TODO: change the value of star entries to grandfathers so it will be int64_t as well.
    
    FullyDistVec<int64_t,short> star(A.getcommgrid(), A.getnrow(), 1);    // all initialized to true
    FullyDistVec<int64_t, int64_t> grandfather = father(father); // find grandparents
    grandfather.EWiseOut(father,
                         [](int64_t pv, int64_t gpv) -> short { return static_cast<short>(pv == gpv); },
                         star); // remove some stars
    
    // FullyDistSpVec FullyDistVec::Find() requires no communication
    // because FullyDistSpVec (the return object) is distributed based on length, not nonzero counts
    return star;
}

void ConditionalHook(Dist::MPI_DCCols & A, FullyDistVec<int64_t, int64_t> & father)
{
    FullyDistVec<int64_t,short> stars = StarCheck(A, father);
    
    FullyDistVec<int64_t, int64_t> minNeighborFather ( A.getcommgrid());
    minNeighborFather = SpMV<Select2ndMinSR<double, int64_t>>(A, father); // value is the minimum of all neighbors' fatthers
    
    FullyDistSpVec<int64_t, pair<int64_t, int64_t>> hooks(A.getcommgrid(), A.getnrow());
    // create entries belonging to stars
    hooks = EWiseApply<pair<int64_t, int64_t>>(hooks, stars,
                                               [](pair<int64_t, int64_t> x, short isStar){return make_pair(0,0);},
                                               [](pair<int64_t, int64_t> x, short isStar){return isStar==1;},
                                               true, {0,0});
    
    
    // include father information
    
    hooks = EWiseApply<pair<int64_t, int64_t>>(hooks, father,
                                               [](pair<int64_t, int64_t> x, short f){return make_pair(f,0);},
                                               [](pair<int64_t, int64_t> x, short f){return true;},
                                               false, {0,0});
    
    //keep entries with father>minNeighborFather and insert minNeighborFather information
    hooks = EWiseApply<pair<int64_t, int64_t>>(hooks,  minNeighborFather,
                                               [](pair<int64_t, int64_t> x, short mnf){return make_pair(get<0>(x), mnf);},
                                               [](pair<int64_t, int64_t> x, short mnf){return get<0>(x) > mnf;},
                                               false, {0,0});
    //Invert
    FullyDistSpVec<int64_t, pair<int64_t, int64_t>> starhooks= hooks.Invert(hooks.TotalLength(),
                                                                            [](pair<int64_t, int64_t> val, int64_t ind){return get<0>(val);},
                                                                            [](pair<int64_t, int64_t> val, int64_t ind){return make_pair(ind, get<0>(val));},
                                                                            [](pair<int64_t, int64_t> val1, pair<int64_t, int64_t> val2){return val1;} );
    
    
    // drop the send field of the par
    FullyDistSpVec<int64_t, int64_t> finalhooks = EWiseApply<int64_t>(starhooks,  father,
                                                                      [](pair<int64_t, int64_t> x, int64_t f){return get<1>(x);},
                                                                      [](pair<int64_t, int64_t> x, int64_t f){return true;},
                                                                      false, {0,0});
    father.Set(finalhooks);
}

void UnconditionalHook(Dist::MPI_DCCols & A)
{
}

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
            cout << "Usage: ./cc -M <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-rand <RANDOMLY PERMUTE VERTICES> (default:0)\n";
            cout << "-phases <NUM PHASES in SPGEMM> (default:1)\n";
            cout << "Example (0-indexed mtx with random permutation): ./cc -M input.mtx -base 0 -rand 1" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        string ifilename = "";
        int base = 1;
        int randpermute = 0;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-M")==0)
            {
                ifilename = string(argv[i+1]);
                if(myrank == 0) printf("filename: %s",ifilename.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
            else if (strcmp(argv[i],"-rand")==0)
            {
                randpermute = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nRandomly permute the matrix? (1 or 0):%d",randpermute);
            }
        }
        
        double tIO = MPI_Wtime();
        Dist::MPI_DCCols A;	// construct object
        A.ParallelReadMM(ifilename, base, maximum<bool>());	// if base=0, then it is implicitly converted to Boolean false
        
        
        ostringstream outs;
        outs << "File Read time: " << MPI_Wtime() - tIO << endl;
        SpParHelper::Print(outs.str());
        
        if(randpermute)
        {
            // randomly permute for load balance
            if(A.getnrow() == A.getncol())
            {
                FullyDistVec<int64_t, int64_t> p( A.getcommgrid());
                p.iota(A.getnrow(), 0);
                p.RandPerm();
                (A)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
        }
        float balance = A.LoadImbalance();
        int64_t nnz = A.getnnz();
        outs.str("");
        outs.clear();
        outs << "Load balance: " << balance << endl;
        outs << "Nonzeros: " << nnz << endl;
        SpParHelper::Print(outs.str());
        
        A.AddLoops(1.0);    // the loop value doesn't really matter anyway
        SpParHelper::Print("Added loops");
        A.PrintInfo();
        
        FullyDistVec<int64_t,int64_t> father(A.getcommgrid());
        father.iota(A.getnrow(), 0);    // father(i)=i initially
        ConditionalHook(A, father);
    }	
    
    MPI_Finalize();
    return 0;
}
