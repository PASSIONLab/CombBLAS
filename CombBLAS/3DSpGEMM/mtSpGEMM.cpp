#include <cstdlib>
#include <parallel/algorithm>
#include "../CombBLAS.h"


// multithreaded
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    
    double t01 = MPI_Wtime();
    
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IT, NTO>(0, mdim, ndim);
    }
    
    Dcsc<IT,NT1> Adcsc = *(A.GetDCSC());
    Dcsc<IT,NT2> Bdcsc = *(B.GetDCSC());
    IT nA = A.getncol();
    IT cnzmax = Adcsc.nz + Bdcsc.nz;	// estimate on the size of resulting matrix C
    float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    
  
    Adcsc.ConstructAux(nA, aux); // this is fast
    


    
    // *************** Creating global space to store result, used by all threads *********************
    
    IT* maxnnzc = new IT[Bdcsc.nzc]; // maximum number of nnz in each column of C
    IT flops = 0; // total flops (multiplication) needed to generate C
#pragma omp parallel
    {
        IT tflops=0; //thread private flops
#pragma omp for
        for(int i=0; i < Bdcsc.nzc; ++i)
        {
            IT locmax = 0;
            IT nnzcol = Bdcsc.cp[i+1] - Bdcsc.cp[i];
            vector< pair<IT,IT> > colinds(nnzcol);
            Adcsc.FillColInds(Bdcsc.ir + Bdcsc.cp[i], nnzcol, colinds, aux, csize);
            
            for(IT j = 0; (unsigned)j < nnzcol; ++j)		// create the initial heap
            {
                locmax = locmax + (colinds[j].second - colinds[j].first);
            }
            
            maxnnzc[i] = locmax;
            tflops += locmax;
        }
#pragma omp critical
        {
            flops += tflops;
        }
    }
 

    int numThreads;
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    IT flopsPerThread = flops/numThreads; // amount of work that will be assigned to each thread
    IT colPerThread [numThreads + 1]; // thread i will process columns from colPerThread[i] to colPerThread[i+1]-1
    
    IT* colStart = new IT[Bdcsc.nzc]; //start index in the global array for storing ith column of C
    IT* colEnd = new IT[Bdcsc.nzc]; //end index in the global array for storing ith column of C
    colStart[0] = 0;
    colEnd[0] = 0;
    
    int curThread = 0;
    colPerThread[curThread++] = 0;
    IT nextflops = flopsPerThread;
    
    // TODO: the following prefix sum can be parallelized, e.g., see
    // http://stackoverflow.com/questions/21352816/prefix-sums-taking-too-long-openmp
    // not a dominating term at this moment
    for(int i=0; i < (Bdcsc.nzc-1); ++i)
    {
        colStart[i+1] = colStart[i] + maxnnzc[i];
        colEnd[i+1] = colStart[i+1];
        if(nextflops < colStart[i+1])
        {
            colPerThread[curThread++] = i+1;
            nextflops += flopsPerThread;
        }
    }
    while(curThread < numThreads)
        colPerThread[curThread++] = Bdcsc.nzc;
    colPerThread[numThreads] = Bdcsc.nzc;

    IT size = colEnd[Bdcsc.nzc-1] + maxnnzc[Bdcsc.nzc-1];
    //IT * RowIdsofC = new IT[size];
    //NT * ValuesofC = new NT[size];
    //vector<StackEntry< NTO, pair<IT,IT>>> * colsC = new vector<StackEntry< NTO, pair<IT,IT>>>[Bdcsc.nzc];
    
    StackEntry< NTO, pair<IT,IT> > * colsC = static_cast<StackEntry< NTO, pair<IT,IT> > *> (::operator new (sizeof(StackEntry<NTO, pair<IT,IT> >[size])));
    delete [] maxnnzc;
    // ************************ End Creating global space *************************************
    
    
    // *************** Creating global heap space to be used by all threads *********************
    IT threadHeapSize[numThreads];
#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        IT localmax = -1;
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        {
            IT colnnz = Bdcsc.cp[i+1]-Bdcsc.cp[i];
            if(colnnz > localmax) localmax = colnnz;
        }
        threadHeapSize[thisThread] = localmax;
    }
    
    IT threadHeapStart[numThreads+1];
    threadHeapStart[0] = 0;
    for(int i=0; i<numThreads; i++)
        threadHeapStart[i+1] = threadHeapStart[i] + threadHeapSize[i];
    HeapEntry<IT,NT1> * globalheap = new HeapEntry<IT,NT1>[threadHeapStart[numThreads]];
    
    // ************************ End Creating global heap space *************************************
   

    

#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        HeapEntry<IT,NT1> * wset = globalheap + threadHeapStart[thisThread]; // thread private heap space
       
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        //for(IT i=0; i< Bdcsc.nzc; ++i)		// for all the columns of B
        {
            
        
            IT nnzcol = Bdcsc.cp[i+1] - Bdcsc.cp[i];
            //HeapEntry<IT, NT1> * wset = new HeapEntry<IT, NT1>[nnzcol];
            // heap keys are just row indices (IT)
            // heap values are <numvalue, runrank>
            // heap size is nnz(B(:,i)
            
            
            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
            vector< pair<IT,IT> > colinds(nnzcol);
            Adcsc.FillColInds(Bdcsc.ir + Bdcsc.cp[i], nnzcol, colinds, aux, csize); // can be done multithreaded
            IT hsize = 0;
            
            for(IT j = 0; (unsigned)j < nnzcol; ++j)		// create the initial heap
            {
                if(colinds[j].first != colinds[j].second)	// current != end
                {
                     wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
                }
            }
            make_heap(wset, wset+hsize);
            
            
            
            // No need to keep redefining key and hentry with each iteration of the loop
            while(hsize > 0)
            {
                pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
                IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column
                
                 NTO mrhs = SR::multiply(wset[hsize-1].num, Bdcsc.numx[Bdcsc.cp[i]+locb]);
                if (!SR::returnedSAID())
                {
                     if( (colEnd[i] > colStart[i]) && colsC[colEnd[i]-1].key.second == wset[hsize-1].key)
                    {
                        colsC[colEnd[i]-1].value = SR::add(colsC[colEnd[i]-1].value, mrhs);
                    }
                    else
                    {
                        StackEntry< NTO, pair<IT,IT>> se;
                        se.value = mrhs;
                        se.key = make_pair(Bdcsc.jc[i], wset[hsize-1].key);
                        
                        colsC[colEnd[i]]= se ;
                        colEnd[i] ++;
                    }
                }
                
                if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
                {
                    // runr stays the same !
                    wset[hsize-1].key = Adcsc.ir[colinds[locb].first];
                    wset[hsize-1].num = Adcsc.numx[colinds[locb].first];
                    push_heap(wset, wset+hsize);
                }
                else
                {
                    --hsize;
                }
            }

        }

    }
    delete [] aux;
    delete [] globalheap;
    
    
    vector<IT> colptrC(Bdcsc.nzc+1);
    colptrC[0] = 0;
    for(IT i=0; i< Bdcsc.nzc; ++i)  // insignificant
    {
        //colptrC[i+1] = colptrC[i] + colsC[i].size();
        colptrC[i+1] = colptrC[i] +colEnd[i]-colStart[i];
    }
    IT nzc = colptrC[Bdcsc.nzc];
    
    
    //StackEntry< NTO, pair<IT,IT> > * multstack = new StackEntry<NTO, pair<IT,IT> >[nzc];
    StackEntry< NTO, pair<IT,IT> > * multstack = static_cast<StackEntry< NTO, pair<IT,IT> > *> (::operator new (sizeof(StackEntry<NTO, pair<IT,IT> >[nzc])));
    
#pragma omp parallel for
    for(IT i=0; i< Bdcsc.nzc; ++i)        // combine step
    {
        //copy(colsC[i].begin(), colsC[i].end(), multstack + colptrC[i]);
        copy(&colsC[colStart[i]], &colsC[colEnd[i]], multstack + colptrC[i]);
    }
    
    delete [] colsC;
    
    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    cout << " local SpGEMM " << MPI_Wtime()-t01 << " seconds" << endl;
    return new SpTuples<IT, NTO> (nzc, mdim, ndim, multstack);
}








template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM1
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA = false, bool clearB = false)
{
    
    
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IT, NTO>(0, mdim, ndim);
    }
    
    Dcsc<IT,NT1> Adcsc = *(A.GetDCSC());
    Dcsc<IT,NT2> Bdcsc = *(B.GetDCSC());
    IT nA = A.getncol();
    IT cnzmax = Adcsc.nz + Bdcsc.nz;	// estimate on the size of resulting matrix C
    float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc.ConstructAux(nA, aux);
    
    
    IT cnz = 0;
    vector<StackEntry< NTO, pair<IT,IT>>> * colsC = new vector<StackEntry< NTO, pair<IT,IT>>>[Bdcsc.nzc];
    
    
//#pragma omp parallel for
    for(IT i=0; i< Bdcsc.nzc; ++i)		// for all the columns of B
    {
        IT prevcnz = cnz;
        IT nnzcol = Bdcsc.cp[i+1] - Bdcsc.cp[i];
        HeapEntry<IT, NT1> * wset = new HeapEntry<IT, NT1>[nnzcol];
        // heap keys are just row indices (IT)
        // heap values are <numvalue, runrank>
        // heap size is nnz(B(:,i)
        
        // colnums vector keeps column numbers requested from A
        vector<IT> colnums(nnzcol);
        
        // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
        // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
        vector< pair<IT,IT> > colinds(nnzcol);
        copy(Bdcsc.ir + Bdcsc.cp[i], Bdcsc.ir + Bdcsc.cp[i+1], colnums.begin());
        
        Adcsc.FillColInds(&colnums[0], colnums.size(), colinds, aux, csize); // can be done multithreaded
        IT maxnnz = 0;	// max number of nonzeros in C(:,i)
        IT hsize = 0;
        
        for(IT j = 0; (unsigned)j < colnums.size(); ++j)		// create the initial heap
        {
            if(colinds[j].first != colinds[j].second)	// current != end
            {
                wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
                maxnnz += colinds[j].second - colinds[j].first;
            }
        }
        make_heap(wset, wset+hsize);
        
        
        
        // No need to keep redefining key and hentry with each iteration of the loop
        while(hsize > 0)
        {
            pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
            IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column
            
            // type promotion done here:
            // static T_promote multiply(const T1 & arg1, const T2 & arg2)
            //	return (static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2) );
            NTO mrhs = SR::multiply(wset[hsize-1].num, Bdcsc.numx[Bdcsc.cp[i]+locb]);
            if (!SR::returnedSAID())
            {
                if( (!colsC[i].empty()) && colsC[i].back().key.second == wset[hsize-1].key)
                {
                    colsC[i].back().value = SR::add(colsC[i].back().value, mrhs);
                }
                else
                {
                    StackEntry< NTO, pair<IT,IT>> se;
                    se.value = mrhs;
                    se.key = make_pair(Bdcsc.jc[i], wset[hsize-1].key);
                    colsC[i].push_back( se);
                }
            }
            
            if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
            {
                // runr stays the same !
                wset[hsize-1].key = Adcsc.ir[colinds[locb].first];
                wset[hsize-1].num = Adcsc.numx[colinds[locb].first];
                push_heap(wset, wset+hsize);
            }
            else
            {
                --hsize;
            }
        }
        delete [] wset;
    }
    delete [] aux;
    
    
    
    vector<IT> colptrC(Bdcsc.nzc+1);
    colptrC[0] = 0;
    for(IT i=0; i< Bdcsc.nzc; ++i)
    {
        colptrC[i+1] = colptrC[i] + colsC[i].size();
    }
    IT nzc = colptrC[Bdcsc.nzc];
    
    StackEntry< NTO, pair<IT,IT> > * multstack = new StackEntry<NTO, pair<IT,IT> >[nzc];
#pragma omp parallel for
    for(IT i=0; i< Bdcsc.nzc; ++i)        // combine step
    {
        copy(colsC[i].begin(), colsC[i].end(), multstack + colptrC[i]);
    }
    
    delete [] colsC;
    
    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    return new SpTuples<IT, NTO> (nzc, mdim, ndim, multstack);
}





template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * LocalSpGEMM2
(const SpDCCols<IU, NU1> & A,
 const SpDCCols<IU, NU2> & B,
 bool clearA = false, bool clearB = false)
{
    IU mdim = A.getnrow();
    IU ndim = B.getncol();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IU, NUO>(0, mdim, ndim);
    }
    StackEntry< NUO, pair<IU,IU> > * multstack;
    IU cnz = SpHelper::SpColByCol< SR > (*(A.GetDCSC()), *(B.GetDCSC()), A.getncol(),  multstack);
    
    if(clearA)
        delete const_cast<SpDCCols<IU, NU1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IU, NU2> *>(&B);
    
    return new SpTuples<IU, NUO> (cnz, mdim, ndim, multstack);
}



// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IU, class NU>
tuple<IU, IU, NU>* multiwayMerge1( const vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar = 0, IU nstar = 0, bool delarrs = false )
{
    
    double t01 = MPI_Wtime();
    int nArrSpTups =  ArrSpTups.size();
    mstar = ArrSpTups[0]->getnrow();
    nstar = ArrSpTups[0]->getncol();
    
    /*
    if(nArrSpTups == 0)
    {
        return SpTuples<IU,NU>(0, mstar,nstar);
    }
    else if(nArrSpTups == 1)
    {
        SpTuples<IU,NU> ret = *ArrSpTups[0];
        if(delarrs)
            delete ArrSpTups[0];
        return ret;
    }
    
    
    for(int i=1; i< nArrSpTups; ++i)
    {
        if((mstar != ArrSpTups[i]->getnrow()) || nstar != ArrSpTups[i]->getncol())
        {
            cerr << "Dimensions do not match on MergeAll()" << endl;
            return SpTuples<IU,NU>(0,0,0);
        }
    }
    
    */
    
    
    //cout << mstar << " x " <<nstar << endl;
    
    int numThreads;
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }

    int nsplits = 2*numThreads;
    while(nsplits > nstar)
    {
        nsplits = nsplits/2;
    }
    // now 1<=nsplits<=nstar
    
    //cout << "Number of splits: " << nsplits << endl;

    vector<vector<IU> > colPtrs(nsplits+1, vector<IU>(nArrSpTups));
    vector<IU> split_tuple_estnnz (nsplits);
    std::fill (colPtrs[0].begin(), colPtrs[0].end(), 0);
    std::fill (split_tuple_estnnz.begin(), split_tuple_estnnz.end(), 0);

    ColLexiCompare<IU,NU> comp;
#pragma omp parallel for  //schedule(dynamic)
    for(int i=1; i< nsplits; i++)
    {
        IU cur_col = i * (nstar/nsplits);
        std::tuple<IU, IU, NU> search_tuple(0, cur_col, 0); // (rowindex, colindex, val)
        
        for(int j=0; j< nArrSpTups; j++)  // parallelize this if nthreads < nArrSpTups, then we can use finger search
        {
            tuple<IU, IU, NU>* it;
            it = std::lower_bound (ArrSpTups[j]->tuples, ArrSpTups[j]->tuples+ArrSpTups[j]->getnnz(), search_tuple, comp);
            colPtrs[i][j] = it - ArrSpTups[j]->tuples;
            split_tuple_estnnz[i-1] += (colPtrs[i][j] - colPtrs[i-1][j]);
        }
    }
    for(int j=0; j< nArrSpTups; j++)
    {
        colPtrs[nsplits][j] = ArrSpTups[j]->getnnz();
        split_tuple_estnnz[nsplits-1] += (colPtrs[nsplits][j] - colPtrs[nsplits-1][j]);
    }
    
    
    
    ColLexiCompare<IU,int> heapcomp;
    vector<tuple<IU, IU, NU>* > split_tuples (nsplits);
    
    for(int i=0; i< nsplits; i++)
    {
        //split_tuples[i] = new tuple<IU,IU,NU>[split_tuple_estnnz[i]];
        split_tuples[i] = static_cast<tuple<IU, IU, NU>*> (::operator new (sizeof(tuple<IU, IU, NU>[split_tuple_estnnz[i]])));
    }
    
    
    vector<IU> split_tuple_nnz (nsplits);
    
#pragma omp parallel
    {
        vector<tuple<IU, IU, int>> heap(nArrSpTups); //(rowindex, colindex, source-id)  // moving out of threaded region??
        vector<IU> curptr(nArrSpTups);
#pragma omp parallel for
        for(int i=0; i< nsplits; i++)
        {
            for(int j=0; j< nArrSpTups; j++)
            {
                curptr[j] = colPtrs[i][j];
            }
            IU hsize = 0;
            for(int j=0; j< nArrSpTups; ++j)
            {
                IU curnnz= colPtrs[i+1][j] - colPtrs[i][j];
                if(curnnz>0)
                {
                    heap[hsize++] = make_tuple(get<0>(ArrSpTups[j]->tuples[colPtrs[i][j]]), get<1>(ArrSpTups[j]->tuples[colPtrs[i][j]]), j);
                }
            }
            make_heap(heap.begin(), heap.begin()+hsize, not2(heapcomp));
            tuple<IU, IU, NU> * ntuples = split_tuples[i];
            IU cnz = 0;
            
            while(hsize > 0)
            {
                pop_heap(heap.begin(), heap.begin() + hsize, not2(heapcomp));         // result is stored in heap[hsize-1]
                int source = get<2>(heap[hsize-1]);
                
                if( (cnz != 0) &&
                   ((get<0>(ntuples[cnz-1]) == get<0>(heap[hsize-1])) && (get<1>(ntuples[cnz-1]) == get<1>(heap[hsize-1]))) )
                {
                    get<2>(ntuples[cnz-1])  = SR::add(get<2>(ntuples[cnz-1]), ArrSpTups[source]->numvalue(curptr[source]++));
                }
                else
                {
                    ntuples[cnz++] = ArrSpTups[source]->tuples[curptr[source]++];
                }
                
                if(curptr[source] != colPtrs[i+1][source])	// That array has not been depleted
                    
                {
                    heap[hsize-1] = make_tuple(get<0>(ArrSpTups[source]->tuples[curptr[source]]),
                                               get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
                    push_heap(heap.begin(), heap.begin()+hsize, not2(heapcomp));
                }
                else
                {
                    --hsize;
                }
            }
            split_tuple_nnz[i] = cnz;
        }
    }
    
    
    
    
    if(delarrs)
    {
        for(size_t i=0; i<ArrSpTups.size(); ++i)
            delete ArrSpTups[i];
    }
    
    IU cnz = 0;
    vector<IU> partial_sum (nsplits+1);
    partial_sum [0] = 0;
    for(int i=0; i< nsplits; i++)
    {
        cnz += split_tuple_nnz[i];
        partial_sum[i+1] = partial_sum[i] + split_tuple_nnz[i];
    }
    //tuple<IU, IU, NU> * merge_tuples = new tuple<IU,IU,NU>[cnz];
    tuple<IU, IU, NU>* merge_tuples = static_cast<tuple<IU, IU, NU>*> (::operator new (sizeof(tuple<IU, IU, NU>[cnz])));
    
#pragma omp parallel for
    for(int i=0; i< nsplits; i++)
    {
        std::copy(split_tuples[i], split_tuples[i] + split_tuple_nnz[i], merge_tuples+partial_sum[i]);
    }
    
    for(int i=0; i< nsplits; i++)
    {
        delete split_tuples[i];
    }
    
    cout << " entries merged in " << MPI_Wtime()-t01 << " seconds" << endl;
    //return SpTuples<IU,NU> (cnz, mstar, nstar, merge_tuples);
    
    return merge_tuples;
    
    
}

typedef struct
{
    int i;
    int j;
    double val;
}tup;

template <class IT, class NT>
struct ColLexiCompare1:  // struct instead of class so that operator() is public
public binary_function< tup, tup, bool >  // (par1, par2, return_type)
{
    inline bool operator()(const tup & lhs, const tup & rhs) const
    {
        if(lhs.j == rhs.j)
        {
            return lhs.i < rhs.i;
        }
        else
        {
            return lhs.j < rhs.j;
        }
    }
};



template<class IU, class NU>
//tuple<IU, IU, NU>*  multiwayMerge( const vector<SpTuples<IU,NU> *> & ArrSpTups, IU& mergedListSize, bool delarrs = false )
tuple<IU, IU, NU>*  multiwayMerge( const vector<tuple<IU, IU, NU>*> & ArrSpTups, const vector<IU> & listSizes, IU& mergedListSize, bool delarrs = false )
{
    double t01 = MPI_Wtime();
    int nlists =  ArrSpTups.size();
    IU totSize = 0;
    
    vector<pair<tuple<IU, IU, NU>*, tuple<IU, IU, NU>* > > seqs;
    
    for(int i = 0; i < nlists; ++i)
    {
        seqs.push_back(make_pair(ArrSpTups[i], ArrSpTups[i] + listSizes[i]));
        totSize += listSizes[i];
    }
    
    
    
    ColLexiCompare<IU,NU> comp;
    tuple<IU, IU, NU>* mergedData = static_cast<tuple<IU, IU, NU>*> (::operator new (sizeof(tuple<IU, IU, NU>[totSize])));
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), mergedData, totSize , comp);
    
    
    if(delarrs)
    {
        for(size_t i=0; i<ArrSpTups.size(); ++i)
            delete ArrSpTups[i];
    }

    cout << totSize << " entries merged in " << MPI_Wtime()-t01 << " seconds" << endl;
    t01 = MPI_Wtime();

    int totThreads;
#pragma omp parallel
    {
        totThreads = omp_get_num_threads();
    }
    
    vector <IU> tstart(totThreads);
    vector <IU> tend(totThreads);
    vector <IU> tdisp(totThreads+1);
#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        IU start = threadID * (totSize / totThreads);
        IU end = (threadID + 1) * (totSize / totThreads);
        if(threadID == (totThreads-1)) end = totSize;
        
        //cout << "thread: " << threadID << " start " << start << " end " << end << "  "<< totSize/(IU)totThreads << endl;
        
        IU curpos = start;
        for (IU i = start+1; i < end; ++i)
        {
            if((get<0>(mergedData[i]) == get<0>(mergedData[curpos])) && (get<1>(mergedData[i]) == get<1>(mergedData[curpos])))
            {
                get<2>(mergedData[curpos]) += get<2>(mergedData[i]);
            }
            else
            {
                mergedData[++curpos] = mergedData[i];
            }
        }
        tstart[threadID] = start;
        if(end>start) tend[threadID] = curpos+1;
        else tend[threadID] = end; // start=end
#pragma omp barrier
#pragma omp single
        {
            // serial
            for(int t=totThreads-1; t>0; --t)
            {
                if(tend[t] > tstart[t] && tend[t-1] > tstart[t-1])
                {
                    if((get<0>(mergedData[tstart[t]]) == get<0>(mergedData[tend[t-1]-1])) && (get<1>(mergedData[tstart[t]]) == get<1>(mergedData[tend[t-1]-1])))
                    {
                        get<2>(mergedData[tend[t-1]-1]) += get<2>(mergedData[tstart[t]]);
                        tstart[t] ++;
                    }
                }
            }
            
            tdisp[0] = 0;
            for(int t=0; t<totThreads; ++t)
            {
                tdisp[t+1] = tdisp[t] + tend[t] - tstart[t];
            }
        }
        
        // shrink space in parallel
        std::copy(mergedData + tstart[threadID], mergedData + tend[threadID], mergedData + tdisp[threadID]);
        
    }
    

    mergedListSize = tdisp[totThreads];
    cout << mergedListSize << " entries reduced in " << MPI_Wtime()-t01 << " seconds" << endl;
    //return SpTuples<IU,NU> (reducedSize, ArrSpTups[0]->getnrow(), ArrSpTups[0]->getncol(), mergedData);
    return mergedData;
}


