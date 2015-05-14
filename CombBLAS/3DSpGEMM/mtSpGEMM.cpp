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
    for(IT i=0; i< Bdcsc.nzc; ++i)
    {
        copy(&colsC[colStart[i]], &colsC[colEnd[i]], multstack + colptrC[i]);
    }
    
    delete [] colsC;
    
    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    cout << " local SpGEMM " << MPI_Wtime()-t01 << " seconds" << endl;
    return new SpTuples<IT, NTO> (nzc, mdim, ndim, multstack, true);
}




template<class IU, class NU>
tuple<IU, IU, NU>*  multiwayMerge( const vector<tuple<IU, IU, NU>*> & listTuples, const vector<IU> & listSizes, IU& mergedListSize, bool delarrs = false )
{
    double t01 = MPI_Wtime();
    int nlists =  listTuples.size();
    IU totSize = 0;
    
    vector<pair<tuple<IU, IU, NU>*, tuple<IU, IU, NU>* > > seqs;
    
    for(int i = 0; i < nlists; ++i)
    {
        seqs.push_back(make_pair(listTuples[i], listTuples[i] + listSizes[i]));
        totSize += listSizes[i];
    }
    
    
    
    ColLexiCompare<IU,NU> comp;
    tuple<IU, IU, NU>* mergedData = static_cast<tuple<IU, IU, NU>*> (::operator new (sizeof(tuple<IU, IU, NU>[totSize])));
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), mergedData, totSize , comp);
    
    
    if(delarrs)
    {
        for(size_t i=0; i<listTuples.size(); ++i)
            delete listTuples[i];
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
    }
    
    
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


    
#pragma omp parallel
    {
        int threadID = omp_get_thread_num();
        std::copy(mergedData + tstart[threadID], mergedData + tend[threadID], mergedData + tdisp[threadID]);
    }
    

    mergedListSize = tdisp[totThreads];
    cout << mergedListSize << " entries reduced in " << MPI_Wtime()-t01 << " seconds" << endl;
    return mergedData;
}


