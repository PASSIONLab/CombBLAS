//#define DETERMINISTIC 1

#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP // should be defined before any COMBBLAS header is included
#endif
#include <omp.h>
#endif

#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


#define EDGEFACTOR 16
#define RAND_PERMUTE 1

#ifdef DETERMINISTIC
MTRand GlobalMT(1);
#else
MTRand GlobalMT;	// generate random numbers with Mersenne Twister
#endif

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_localspmvtime;
double cblas_mergeconttime;
double cblas_transvectime;



using namespace std;



template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    // boolean addition is practically a "logical or"
    // therefore this doesn't destruct any links
    PARMAT AT = A;
    AT.Transpose();
    AT.RemoveLoops(); // not needed for boolean matrices, but no harm in keeping it
    A += AT;
}


struct VertexType
{
public:
    VertexType(int64_t ord=-1, int64_t deg=-1){order=ord; degree = deg;};
    
    friend bool operator<(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.order==vtx2.order) return vtx1.degree < vtx2.degree;
        else return vtx1.order<vtx2.order;
    };
    friend bool operator<=(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.order==vtx2.order) return vtx1.degree <= vtx2.degree;
        else return vtx1.order<vtx2.order;
    };
    friend bool operator>(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.order==vtx2.order) return vtx1.degree > vtx2.degree;
        else return vtx1.order>vtx2.order;
    };
    friend bool operator>=(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.order==vtx2.order) return vtx1.degree >= vtx2.degree;
        else return vtx1.order>vtx2.order;
        
        //if(vtx1.order==vtx2.order) return vtx1.degree <= vtx2.degree;
        //else return vtx1.order<vtx2.order;
    };
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.order==vtx2.order & vtx1.degree==vtx2.degree;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.order << "," << vertex.degree << ")"; return os;};
    //private:
    int64_t order;
    int64_t degree;
};



struct SelectMinSR
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T_promote multiply(const bool & arg1, const T_promote & arg2)
    {
        return arg2;
    }
    
    static void axpy(bool a, const T_promote & x, T_promote & y)
    {
        y = std::min(y, x);
    }
};


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;




FullyDistSpVec<int64_t, int64_t> getOrder(FullyDistSpVec<int64_t, VertexType> &fringeRow)
{

    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    
    vector<int64_t> lind = fringeRow.GetLocalInd ();
    vector<VertexType> lnum = fringeRow.GetLocalNum ();
    int64_t ploclen = lind.size();
    int64_t localmin = (int64_t)INT_MAX;
    int64_t localmax = (int64_t) 0;
    for(int i=0; i<ploclen; i++)
        if(localmin > lnum[i].order)  localmin = lnum[i].order;
    for(int i=0; i<ploclen; i++)
        if(localmax < lnum[i].order)  localmax = lnum[i].order;
    
    //MPI_Reduce;
    int64_t globalmin = (int64_t) INT_MAX;
    int64_t globalmax = (int64_t) 0;
    MPI_Allreduce( &localmin, &globalmin, 1, MPIType<int64_t>(), MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce( &localmax, &globalmax, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
    
    int64_t nparents = globalmax - globalmin + 1;
    int64_t perproc = nparents/nprocs;
    
    int * rdispls = new int[nprocs+1];
    int * recvcnt = new int[nprocs];
    int * sendcnt = new int[nprocs](); // initialize to 0
    int * sdispls = new int[nprocs+1];
    
    MPI_Barrier(MPI_COMM_WORLD);
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t k=0; k < ploclen; ++k)
    {
        int64_t temp = lnum[k].order-globalmin;
        int owner;
        if(perproc==0 || temp/perproc > nprocs-1)
            owner = nprocs-1;
        else
            owner = temp/perproc;
        
#ifdef _OPENMP
        __sync_fetch_and_add(&sendcnt[owner], 1);
#else
        sendcnt[owner]++;
#endif
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD);  // share the request counts
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    
    int64_t * datbuf1 = new int64_t[ploclen];
    int64_t * datbuf2 = new int64_t[ploclen];
    int64_t * indbuf = new int64_t[ploclen];
    int *count = new int[nprocs](); //current position
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t i=0; i < ploclen; ++i)
    {
        
        int64_t temp = lnum[i].order-globalmin;
        int owner;
        if(perproc==0 || temp/perproc > nprocs-1)
            owner = nprocs-1;
        else
            owner = temp/perproc;
        
        
        int id;
#ifdef _OPENMP
        id = sdispls[owner] + __sync_fetch_and_add(&count[owner], 1);
#else
        id = sdispls[owner] + count[owner];
        count[owner]++;
#endif
        
        datbuf1[id] = temp;
        datbuf2[id] = lnum[i].degree;
        indbuf[id] = lind[i] + fringeRow.LengthUntil();
    }
    delete [] count;

    int64_t totrecv = rdispls[nprocs];
    int64_t * recvdatbuf1 = new int64_t[totrecv];
    int64_t * recvdatbuf2 = new int64_t[totrecv];
    MPI_Alltoallv(datbuf1, sendcnt, sdispls, MPIType<int64_t>(), recvdatbuf1, recvcnt, rdispls, MPIType<int64_t>(), MPI_COMM_WORLD);
    delete [] datbuf1;
    MPI_Alltoallv(datbuf2, sendcnt, sdispls, MPIType<int64_t>(), recvdatbuf2, recvcnt, rdispls, MPIType<int64_t>(), MPI_COMM_WORLD);
    delete [] datbuf2;
    
    int64_t * recvindbuf = new int64_t[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<int64_t>(), recvindbuf, recvcnt, rdispls, MPIType<int64_t>(), MPI_COMM_WORLD);
    delete [] indbuf;
    
   tuple<int64_t,int64_t, int64_t>* tosort = static_cast<tuple<int64_t,int64_t, int64_t>*> (::operator new (sizeof(tuple<int64_t,int64_t, int64_t>)*totrecv));
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = make_tuple(recvdatbuf1[i], recvdatbuf2[i], recvindbuf[i]);
    }
    
    
#if defined(GNU_PARALLEL) && defined(_OPENMP)
    __gnu_parallel::sort(tosort, tosort+totrecv);
#else
    std::sort(tosort, tosort+totrecv);
#endif
    
    //cout << "I am here " << endl;
    //FullyDistSpVec<int64_t,int64_t> order(fringeRow.getcommgrid(), fringeRow.TotalLength());
    // send order back
    int * sendcnt1 = new int[nprocs]();
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t k=0; k < totrecv; ++k)
    {
        int64_t locind;
        int owner = fringeRow.Owner(get<2>(tosort[k]), locind);
#ifdef _OPENMP
        __sync_fetch_and_add(&sendcnt1[owner], 1);
#else
        sendcnt1[owner]++;
#endif
    }
    
    MPI_Alltoall(sendcnt1, 1, MPI_INT, recvcnt, 1, MPI_INT, MPI_COMM_WORLD);  // share the request counts
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt1[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }

    
    
    
    vector<int64_t> sortperproc (nprocs);
    sortperproc[myrank] = totrecv;
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<int64_t>(), sortperproc.data(), 1, MPIType<int64_t>(), MPI_COMM_WORLD);
    
    vector<int64_t> disp(nprocs+1);
    disp[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        disp[i+1] = disp[i] + sortperproc[i];
    }

    
    
    ploclen = totrecv;
    
    int64_t * datbuf = new int64_t[ploclen];
    indbuf = new int64_t[ploclen];
    count = new int[nprocs](); //current position
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t i=0; i < ploclen; ++i)
    {
        int64_t locind;
        int owner = fringeRow.Owner(get<2>(tosort[i]), locind);
        int id;
#ifdef _OPENMP
        id = sdispls[owner] + __sync_fetch_and_add(&count[owner], 1);
#else
        id = sdispls[owner] + count[owner];
        count[owner]++;
#endif
        datbuf[id] = i + disp[myrank] + globalmax + 1;
        indbuf[id] = locind;
    }
    delete [] count;
    
    
    totrecv = rdispls[nprocs];
    vector<int64_t> recvdatbuf3 (totrecv);
    MPI_Alltoallv(datbuf, sendcnt1, sdispls, MPIType<int64_t>(), recvdatbuf3.data(), recvcnt, rdispls, MPIType<int64_t>(), MPI_COMM_WORLD);
    delete [] datbuf;
    
    vector<int64_t> recvindbuf3 (totrecv);
    MPI_Alltoallv(indbuf, sendcnt1, sdispls, MPIType<int64_t>(), recvindbuf3.data(), recvcnt, rdispls, MPIType<int64_t>(), MPI_COMM_WORLD);
    delete [] indbuf;
    

    FullyDistSpVec<int64_t, int64_t> order(fringeRow.getcommgrid(), fringeRow.TotalLength(), recvindbuf3, recvdatbuf3);
    
    
    
    DeleteAll(recvindbuf, recvdatbuf1, recvdatbuf2);
    DeleteAll(sdispls, rdispls, sendcnt, sendcnt1, recvcnt);
    ::operator delete(tosort);
    
    return order;
}

double torderSpMV=0, torderSort=0, torderOther=0;
// perform ordering from a pseudo peripheral vertex
template <typename PARMAT>
void RCMOrder(PARMAT & A, int64_t source, FullyDistVec<int64_t, int64_t>& order, int64_t startOrder, FullyDistVec<int64_t, int64_t> degrees, PreAllocatedSPA<int64_t,bool,int64_t>& SPA)
{
    
    double tSpMV=0, tOrder, tOther, tSpMV1, tsort=0, tsort1;
    tOrder = MPI_Wtime();
    
    int64_t nv = A.getnrow();
    FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  nv );
    order.SetElement(source, startOrder);
    fringe.SetElement(source, startOrder);
    int64_t curOrder = startOrder+1;
    
    

    
    while(fringe.getnnz() > 0) // continue until the frontier is empty
    {
        
        fringe = EWiseApply<int64_t>(fringe, order,
                                     [](int64_t parent_order, int64_t ord){return ord;},
                                     [](int64_t parent_order, int64_t ord){return true;},
                                     false, (int64_t) -1);
        
        tSpMV1 = MPI_Wtime();
        SpMV<SelectMinSR>(A, fringe, fringe, false, SPA);
        //fringe = SpMV(A, fringe, optbuf);
        tSpMV += MPI_Wtime() - tSpMV1;
        fringe = EWiseMult(fringe, order, true, (int64_t) -1);
        
        
        //fringe.DebugPrint();
        FullyDistSpVec<int64_t, VertexType> fringeRow = EWiseApply<VertexType>(fringe, degrees,
                                                                               [](int64_t parent_order, int64_t degree){return VertexType(parent_order, degree);},
                                                                               [](int64_t parent_order, int64_t degree){return true;},
                                                                               false, (int64_t) -1);
        //fringeRow.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(vtx.order, vtx.degree, idx);});
        
        
        //FullyDistSpVec::sort returns (i,j) index pairs such that
        // jth entry before sorting becomes ith entry after sorting.
        // Here i/j is the index of the elements relative to the dense containter
        // Alternatively, j's consist a permutation that would premute the unsorted vector to sorted vector
        /*
        // Currently sort works up to 10k MPI processes
        // We can change the hardcoded limit in SpParHelper::MemoryEfficientPSort
        
        //FullyDistSpVec<int64_t, int64_t> sorted =  fringe.sort();
        FullyDistSpVec<int64_t, int64_t> sorted =  fringeRow.sort();
        tsort += MPI_Wtime()-tsort1;
        // idx is the index  of fringe in sorted order
        FullyDistVec<int64_t, int64_t> idx = sorted.FindVals([](int64_t x){return true;});
        //sorted.DebugPrint();
        //idx.DebugPrint();
        //MPI_Barrier(order.getcommgrid()->GetWorld());
        //cout << "done ordering.... " << endl;
        //MPI_Barrier(order.getcommgrid()->GetWorld());
        FullyDistVec<int64_t, int64_t> val(idx.getcommgrid());
        // val is the index  of fringe in sorted order (relative to each other starting with  1)
        val.iota(idx.TotalLength(),curOrder);
        curOrder += idx.TotalLength();
         FullyDistSpVec<int64_t, int64_t> levelOrder (fringe.TotalLength(), idx, val);
        */
        tsort1 = MPI_Wtime();
        FullyDistSpVec<int64_t, int64_t> levelOrder = getOrder(fringeRow);
        tsort += MPI_Wtime()-tsort1;
        order.Set(levelOrder);
        //order.DebugPrint();
        //MPI_Barrier(order.getcommgrid()->GetWorld());
        //cout << "return ordering.... " << endl;
        //MPI_Barrier(order.getcommgrid()->GetWorld());
        
    }
    
    tOrder = MPI_Wtime() - tOrder;
    tOther = tOrder - tSpMV - tsort;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "==================Ordering time =======================\n";
        cout << "SpMV time " << " Sort time "<< " Other time" << endl;
        cout << tSpMV << "        "<< tsort << "       "<< tOther << endl;
        cout << "Total time: " <<  tOrder << " seconds." << endl;
        cout << "=======================================================\n";
    }
    
    torderSpMV+=tSpMV; torderSort+=tsort; torderOther+=tOther;
    //order.DebugPrint();
    
}



int threads, processors;
string base_filename;
template <typename PARMAT>
FullyDistVec<int64_t, int64_t> RCM(PARMAT & A, FullyDistVec<int64_t, int64_t> degrees, PreAllocatedSPA<int64_t,bool,int64_t>& SPA)
{
#ifdef TIMING
    cblas_allgathertime = 0;
    cblas_alltoalltime = 0;
    cblas_mergeconttime = 0;
    cblas_transvectime = 0;
    cblas_localspmvtime = 0;
#endif
    /*
     unvisitedVertices: list of current unvisited vertices.
     Each entry is a (degree, vertex index) pair.
     I am keeping index as a value so that we can call reduce to finds the vertex with the minimum/maximum degree.
     TODO: alternatively, we can create a MinIdx and MaxIdx function (will not be faster)
     After discovering a pseudo peripheral vertex in the ith connected component
     all vertices in the ith component are removed from this vector.
     Degrees can be replaced by random numbers or something else.
     */
    FullyDistSpVec<int64_t, int64_t> unvisited ( A.getcommgrid(),  A.getnrow());
    unvisited.iota(A.getnrow(), (int64_t) 0); // index and values become the same
    FullyDistSpVec<int64_t, pair<int64_t, int64_t>> unvisitedVertices =
    EWiseApply<pair<int64_t, int64_t>>(unvisited, degrees,
                                       [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                       [](int64_t vtx, int64_t deg){return true;},
                                       false, (int64_t) -1);
    
    
    // final RCM order
    FullyDistVec<int64_t, int64_t> rcmorder ( A.getcommgrid(),  A.getnrow(), (int64_t) -1);
    // current connected component
    int cc = 1;
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    int64_t numUnvisited = unvisitedVertices.getnnz();
    

    double tpvSpMV=0, tpvOther=0;
    
    while(numUnvisited>0) // for each connected component
    {
        if(myrank == 0)
        {
            cout << "\n*********** Connected component # " << cc << " *****************" << endl;
            cout << "Discovering Pseudo-peripheral vertex\n";
        }
        double tstart = MPI_Wtime();
        // Select a minimum-degree unvisited vertex as the initial source
        pair<int64_t, int64_t> mindegree_vertex = unvisitedVertices.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
        int64_t source = mindegree_vertex.second;
        
        int64_t prevLevel=-1, curLevel=0; // initialized just to make the first iteration going
        // level structure in the current BFS tree
        // we are not using this information. Currently it is serving as visited flag
        FullyDistVec<int64_t, int64_t> level ( A.getcommgrid(),  A.getnrow(), (int64_t) -1);
        
        int iterations = 0;
        double tSpMV=0, tOther=0, tSpMV1;
        while(curLevel > prevLevel)
        {
            double tItr = MPI_Wtime();
            prevLevel = curLevel;
            FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  A.getnrow() );
            level = (int64_t)-1; // reset level structure in every iteration
            level.SetElement(source, 1); // place source at level 1
            fringe.SetElement(source, 1); // include source to the initial fringe
            curLevel = 2;
            while(fringe.getnnz() > 0) // continue until the frontier is empty
            {
                //fringe.setNumToInd(); // unncessary since we don't care about the parent
                //cout << "vector nnz: " << fringe.getnnz() << endl;
                tSpMV1 = MPI_Wtime();
                
                //SpMV<SelectMinSR>(A, fringe, fringe, false, SPA);
                SpMV<SelectMinSR>(A, fringe, fringe, false, SPA);
                
                //fringe = SpMV(A, fringe);
                tSpMV += MPI_Wtime() - tSpMV1;
                fringe = EWiseMult(fringe, level, true, (int64_t) -1);
                // set value to the current level
                fringe=curLevel;
                curLevel++;
                level.Set(fringe);
            }
            curLevel = curLevel-2;
            
            
            // last non-empty level
            fringe = level.Find(curLevel); // we can avoid this by keeping the last nonempty fringe
            fringe.setNumToInd();
            
            // find a minimum degree vertex in the last level
            FullyDistSpVec<int64_t, pair<int64_t, int64_t>> fringe_degree =
            EWiseApply<pair<int64_t, int64_t>>(fringe, degrees,
                                               [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                               [](int64_t vtx, int64_t deg){return true;},
                                               false, (int64_t) -1);
            
            
            mindegree_vertex = fringe_degree.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
            if (curLevel > prevLevel)
                source = mindegree_vertex.second;
            iterations++;
            
            
            if(myrank == 0)
            {
                cout <<" iteration: "<<  iterations << " BFS levels: " << curLevel << " Time: "  << MPI_Wtime() - tItr << " seconds." << endl;
            }
            
        }
        
        tOther = MPI_Wtime() - tstart - tSpMV;
        tpvSpMV += tSpMV;
        tpvOther += tOther;
        if(myrank == 0)
        {
            cout << "==================Overall Stats =======================\n";
            cout << "vertex " << source << " is a pseudo peripheral vertex" << endl;
            cout << "pseudo diameter: " << curLevel << " iterations: "<< iterations <<  endl;
            cout << "SpMV time " << " Other time" << endl;
            cout << tSpMV << "          "<< tOther << endl;
            cout << "Total time: " <<  MPI_Wtime() - tstart << " seconds." << endl;
            cout << "======================================================\n";
            
        }
        cc++;
        
        // order vertices in this connected component
        int64_t curOrder =  A.getnrow() - numUnvisited;
        RCMOrder(A, source, rcmorder, curOrder, degrees, SPA);
        
        // remove vertices in the current connected component
        //unvisitedVertices = EWiseMult(unvisitedVertices, level, true, (int64_t) -1);
        unvisitedVertices = EWiseApply<pair<int64_t, int64_t>>(unvisitedVertices, level,
                                                               [](pair<int64_t, int64_t> vtx, int64_t visited){return vtx;},
                                                               [](pair<int64_t, int64_t> vtx, int64_t visited){return visited==-1;},
                                                               false, make_pair((int64_t)-1, (int64_t)0));
        numUnvisited = unvisitedVertices.getnnz();
    }
    
#ifdef TIMING
    double *td_ag_all, *td_a2a_all, *td_tv_all, *td_mc_all, *td_spmv_all;
    if(myrank == 0)
    {
        td_ag_all = new double[nprocs];
        td_a2a_all = new double[nprocs];
        td_tv_all = new double[nprocs];
        td_mc_all = new double[nprocs];
        td_spmv_all = new double[nprocs];
    }
    
    MPI_Gather(&cblas_allgathertime, 1, MPI_DOUBLE, td_ag_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&cblas_alltoalltime, 1, MPI_DOUBLE, td_a2a_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&cblas_transvectime, 1, MPI_DOUBLE, td_tv_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&cblas_mergeconttime, 1, MPI_DOUBLE, td_mc_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&cblas_localspmvtime, 1, MPI_DOUBLE, td_spmv_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    
    double td_ag_all1=0, td_a2a_all1=0, td_tv_all1=0, td_mc_all1=0,td_spmv_all1 = 0;
    
    
    if(myrank == 0)
    {
        
        vector<double> total_time(nprocs, 0);
        for(int i=0; i< nprocs; ++i) 				// find the mean performing guy
            total_time[i] += td_ag_all[i] +  td_a2a_all[i] + td_tv_all[i] + td_mc_all[i] + td_spmv_all[i];
        
        // order
        vector<pair<double, int>> tosort;
        for(int i=0; i<nprocs; i++) tosort.push_back(make_pair(total_time[i], i));
        sort(tosort.begin(), tosort.end());
        //vector<int> permutation = SpHelper::order(total_time);
        vector<int> permutation(nprocs);
        for(int i=0; i<nprocs; i++) permutation[i] = tosort[i].second;
        
        int smallest = permutation[0];
        int largest = permutation[nprocs-1];
        int median = permutation[nprocs/2];
        
        cout << "TOTAL (accounted) MEAN: " << accumulate( total_time.begin(), total_time.end(), 0.0 )/ static_cast<double> (nprocs) << endl;
        cout << "TOTAL (accounted) MAX: " << total_time[0] << endl;
        cout << "TOTAL (accounted) MIN: " << total_time[nprocs-1]  << endl;
        cout << "TOTAL (accounted) MEDIAN: " << total_time[nprocs/2] << endl;
        cout << "-------------------------------" << endl;
        
        cout << "allgather median: " << td_ag_all[median] << endl;
        cout << "all2all median: " << td_a2a_all[median] << endl;
        cout << "transposevector median: " << td_tv_all[median] << endl;
        cout << "mergecontributions median: " << td_mc_all[median] << endl;
        cout << "spmsv median: " << td_spmv_all[median] << endl;
        cout << "-------------------------------" << endl;
        td_ag_all1=td_ag_all[median]; td_a2a_all1=td_a2a_all[median];
        td_tv_all1=td_tv_all[median]; td_mc_all1=td_mc_all[median];
        td_spmv_all1 = td_spmv_all[median];
       
        cout << "allgather fastest: " << td_ag_all[smallest] << endl;
        cout << "all2all fastest: " << td_a2a_all[smallest] << endl;
        cout << "transposevector fastest: " << td_tv_all[smallest] << endl;
        cout << "mergecontributions fastest: " << td_mc_all[smallest] << endl;
        cout << "spmsv fastest: " << td_spmv_all[smallest] << endl;
        cout << "-------------------------------" << endl;
        
        
        cout << "allgather slowest: " << td_ag_all[largest] << endl;
        cout << "all2all slowest: " << td_a2a_all[largest] << endl;
        cout << "transposevector slowest: " << td_tv_all[largest] << endl;
        cout << "mergecontributions slowest: " << td_mc_all[largest] << endl;
        cout << "spmsv slowest: " << td_spmv_all[largest] << endl;
    }
#endif

    
    if(myrank == 0)
    {
        
        cout << "summary statistics" << endl;
        cout << base_filename << " " << processors << " " << threads << " " << processors * threads << " "<< tpvSpMV << " "<< tpvOther << " "<< torderSpMV <<  " "<< torderSort<<  " "<<  torderOther<<  " "<< td_ag_all1 << " "<<  td_a2a_all1 << " "<<  td_tv_all1 << " "<<  td_mc_all1 << " "<< td_spmv_all1 << " "<<  endl;
        
    }

    
    return rcmorder;
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
            cout << "Usage: ./rcm <rmat|er|input> <scale|filename> <splitPerThread>" << endl;
            cout << "Example: mpirun -np 4 ./rcm rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm er 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm input a.mtx" << endl;
            
        }
        MPI_Finalize();
        return -1;
    }
    {
        Par_DCSC_Bool * ABool;
        ostringstream tinfo;
        
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new Par_DCSC_Bool();
            string filename(argv[2]);
            tinfo.str("");
            tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
            
            base_filename = filename.substr(filename.find_last_of("/\\") + 1);
            
            SpParHelper::Print(tinfo.str());
            double t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename, false, maximum<bool>());
            double t02 = MPI_Wtime();
            int64_t bw = ABool->Bandwidth();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            tinfo << "Bandwidth before random permutation " << bw << endl;
            SpParHelper::Print(tinfo.str());
            
           
#ifdef RAND_PERMUTE
            if(ABool->getnrow() == ABool->getncol())
            {
                FullyDistVec<int64_t, int64_t> p( ABool->getcommgrid());
                p.iota(ABool->getnrow(), 0);
                p.RandPerm();
                (*ABool)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
#endif
            // ::ParallelReadMM should already create symmetric matrix if the file is symmetric as described in header
        }
        else if(string(argv[1]) == string("rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else if(string(argv[1]) == string("er"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.25, .25, .25, .25};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
        
        
        /*
        MPI_Comm com = ABool->getcommgrid()->GetWorld();
        double gtime = MPI_Wtime();
        SpParHelper::GatherMatrix(com, ABool->seq(), (int)0);
        if(myrank==0)
        {
            cout << "gathertime " << MPI_Wtime() - gtime << endl;
        }
        */
        
        
        ABool->RemoveLoops();
        int64_t bw = ABool->Bandwidth();
        float balance = ABool->LoadImbalance();
        
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degrees ( ABool->getcommgrid());
        ABool->Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
       
        /*
        FullyDistVec<int64_t, double> kth ( ABool->getcommgrid());
        Par_DCSC_Double AInt = *ABool;
        AInt.Apply([](double x){return GlobalMT.rand() + 1;});
        AInt.PrintInfo();
        double ktime = MPI_Wtime();
        AInt.Kselect(kth, static_cast<int64_t>(2));
        if(myrank==0)
        {
            cout << "kselect " << MPI_Wtime() - ktime << endl;
        }
        kth.DebugPrint();
        AInt.PruneColumn(kth, less<double>(), true);
        AInt.PrintInfo();
        */
        
        
        
        Par_CSC_Bool * ABoolCSC = new Par_CSC_Bool(*ABool);
        //ABoolCSC->PrintInfo();
        
        
        
        int nthreads = 1;
        int splitPerThread = 1;
        if(argc==4)
            splitPerThread = atoi(argv[3]);
        int cblas_splits = splitPerThread;
        
        
        
#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
            cblas_splits = nthreads * splitPerThread;
        }
        tinfo.str("");
        tinfo << "Threading activated with " << nthreads << endl;
        SpParHelper::Print(tinfo.str());
#endif
        
        threads = nthreads;
        processors = nprocs;
        
        ostringstream outs;
        outs << "--------------------------------------" << endl;
        outs << "Number of MPI proceses: " << nprocs << endl;
        outs << "Number of threads per procese: " << nthreads << endl;
        outs << "Number of splits of the matrix: " << cblas_splits << endl;
        outs << "Load balance: " << balance << endl;
        outs << "Bandwidth after random permutation " << bw << endl;
        outs << "--------------------------------------" << endl;
        SpParHelper::Print(outs.str());
        
        // compute bandwidth
        if(cblas_splits>=1)
        {
            // ABool->ActivateThreading(cblas_splits); // note: crash on empty matrix
            //ABoolCSC->ActivateThreading(cblas_splits);
        }
        
        // Compute RCM ordering
        // FullyDistVec<int64_t, int64_t> rcmorder = RCM(*ABool, degrees);
        
        SpParHelper::Print("Pre allocating SPA\n");
        PreAllocatedSPA<int64_t,bool,int64_t> SPA(ABoolCSC->seq(), nthreads*4);
        SpParHelper::Print("Pre allocated SPA\n");
        
        FullyDistVec<int64_t, int64_t> rcmorder = RCM(*ABoolCSC, degrees, SPA);

        
        // note: threaded matrix can not be permuted
        // That is why I am not a supporter of split matrix.
        // ABAB: Any suggestions in lieu of split matrix?
        
        
        // compute bandwidth of the permuted matrix
        // using DCSC version here which is not split
        //if(cblas_splits==1)
        
            // Ariful: sort returns permutation from ordering
            // and make the original vector a sequence (like iota)
            // I actually need an invert to convert ordering a permutation
    
            FullyDistVec<int64_t, int64_t>rcmorder1 = rcmorder.sort();
            (*ABool)(rcmorder1,rcmorder1,true);// in-place permute to save memory
            bw = ABool->Bandwidth();
            ostringstream outs1;
            outs1 << "Bandwidth after RCM " << bw << endl;
            SpParHelper::Print(outs1.str());
            
        
        delete ABool;
        delete ABoolCSC;
        
    }
    MPI_Finalize();
    return 0;
}

