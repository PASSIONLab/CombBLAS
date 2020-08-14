/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/



#include <mpi.h>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"

namespace combblas {

class HipMCLClusterSaveHandler
{
public:
    // no reader
    template <typename c, typename t, typename VT>
    void save(std::basic_ostream<c,t>& os, std::vector<VT> & strvec, int64_t index)
    {
        for (auto it = strvec.begin() ; it != strvec.end(); ++it)
            os << *it << " ";
    }
};


/**
 * Write clusters to file: vertices belonging to a cluster are written in a single line separated by space.
 * TODO: sort clusters by their sizes
 * @param[in] ofName {output file name}
 * @param[in] clustIdForVtx {the ith entry stores the cluster id of the ith vertex}
 * @param[in] vtxLabels {labels of vertices}
 */

template <class IT>
void WriteMCLClusters(std::string ofName, FullyDistVec<IT, IT> clustIdForVtx, FullyDistVec<IT, std::array<char, MAXVERTNAME> > vtxLabels)
{
    auto commGrid = clustIdForVtx.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    
    // find the number of clusters
    IT nclusters = clustIdForVtx.Reduce(maximum<IT>(), (IT) 0 ) ;
    nclusters ++; // because of zero based indexing for clusters
    
    std::vector<int> rdispls(nprocs+1);
    std::vector<int> recvcnt(nprocs);
    std::vector<int> sendcnt(nprocs,0);
    std::vector<int> sdispls(nprocs+1);
    IT ploclen = clustIdForVtx.LocArrSize();
    
    
    const IT* larr = clustIdForVtx.GetLocArr(); // local part of cluster ids for vertices
    //just to get the destination processor
    FullyDistVec<IT,IT> temp(commGrid, nclusters,0);
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        int owner = temp.Owner(larr[i], locind);
        sendcnt[owner]++;
    }
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    
    typedef std::array<char, MAXVERTNAME> STRASARRAY;
    typedef std::pair< IT, STRASARRAY> TYPE2SEND;
    const STRASARRAY* lVtxLabels = vtxLabels.GetLocArr();
    std::vector<TYPE2SEND> senddata(ploclen);
    
    
    // Pack cluster and vertex information to send
    std::vector<int> count(nprocs, 0);
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        int owner = temp.Owner(larr[i], locind);
        int idx = sdispls[owner] + count[owner];
        count[owner]++;
        senddata[idx] = TYPE2SEND(locind, lVtxLabels[i]); // sending local cluster ids for the destination processor
    }
    
    MPI_Datatype MPI_CLUST;
    MPI_Type_contiguous(sizeof(TYPE2SEND), MPI_CHAR, &MPI_CLUST);
    MPI_Type_commit(&MPI_CLUST);
    
    IT totrecv = rdispls[nprocs];
    std::vector<TYPE2SEND> recvdata(totrecv);
 
    MPI_Alltoallv(senddata.data(), sendcnt.data(), sdispls.data(), MPI_CLUST, recvdata.data(), recvcnt.data(), rdispls.data(), MPI_CLUST, World);
    
    
    // Receiver groups vertices by cluster ids
    std::vector< std::vector<std::string> > vtxGroupbyCC(temp.LocArrSize());
    for(int i=0; i<totrecv; ++i)
    {
        IT clusterID = recvdata[i].first;
        auto locnull = std::find(recvdata[i].second.begin(), recvdata[i].second.end(), '\0'); // find the null character (or string::end)
        std::string vtxstr(recvdata[i].second.begin(), locnull);
        vtxGroupbyCC[clusterID].push_back(vtxstr);
    }
    
    // in each cluster sort vertex labels
#ifdef THREADED
#pragma omp parallel for
#endif
    for(unsigned int i=0; i<vtxGroupbyCC.size(); ++i)
    {
        std::sort(vtxGroupbyCC[i].begin(), vtxGroupbyCC[i].end());
    }
    
    // Create a vector locally populate it
    FullyDistVec<IT,std::vector<std::string> > clusters(commGrid, nclusters, std::vector<std::string>{});
    for(int i=0; i<clusters.LocArrSize(); i++)
    {
        clusters.SetLocalElement(i, vtxGroupbyCC[i]);
    }
    // do not write header and 1-based
    clusters.ParallelWrite(ofName, 1, HipMCLClusterSaveHandler(), false);
    
}



/**
 * Write clusters to file: vertices belonging to a cluster are written in a single line separated by space.
 * Ids of vertices are used as labels
 * TODO: sort clusters by their sizes
 * @param[in] ofName {output file name}
 * @param[in] clustIdForVtx {the ith entry stores the cluster id of the ith vertex}
 */
template <class IT>
void WriteMCLClusters(std::string ofName, FullyDistVec<IT, IT> clustIdForVtx, int base)
{
    auto commGrid = clustIdForVtx.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    IT lenuntil = clustIdForVtx.LengthUntil();
    
    // find the number of clusters
    IT nclusters = clustIdForVtx.Reduce(maximum<IT>(), (IT) 0 ) ;
    nclusters ++; // because of zero based indexing for clusters
    
    std::vector<int> rdispls(nprocs+1);
    std::vector<int> recvcnt(nprocs);
    std::vector<int> sendcnt(nprocs,0);
    std::vector<int> sdispls(nprocs+1);
    IT ploclen = clustIdForVtx.LocArrSize();
    
    
    const IT* larr = clustIdForVtx.GetLocArr(); // local part of cluster ids for vertices
    //just to get the destination processor
    FullyDistVec<IT,IT> temp(commGrid, nclusters,0);
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        int owner = temp.Owner(larr[i], locind);
        sendcnt[owner]++;
    }
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    

    std::vector<std::pair<IT, IT>> senddata(ploclen);
    // Pack cluster and vertex information to send
    std::vector<int> count(nprocs, 0);
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        int owner = temp.Owner(larr[i], locind);
        int idx = sdispls[owner] + count[owner];
        count[owner]++;
        senddata[idx] = std::make_pair(locind, i+lenuntil+base); // sending local cluster ids for the destination processor
    }
    
    MPI_Datatype MPI_CLUST;
    MPI_Type_contiguous(sizeof(std::pair<IT, IT>), MPI_CHAR, &MPI_CLUST);
    MPI_Type_commit(&MPI_CLUST);
    
    IT totrecv = rdispls[nprocs];
    std::vector<std::pair<IT, IT>> recvdata(totrecv);
    
    MPI_Alltoallv(senddata.data(), sendcnt.data(), sdispls.data(), MPI_CLUST, recvdata.data(), recvcnt.data(), rdispls.data(), MPI_CLUST, World);
    
    
    // Receiver groups vertices by cluster ids
    std::vector< std::vector<IT> > vtxGroupbyCC(temp.LocArrSize());
    for(int i=0; i<totrecv; ++i)
    {
        IT clusterID = recvdata[i].first;
        vtxGroupbyCC[clusterID].push_back(recvdata[i].second);
    }
    
    // Create a vector locally populate it
    FullyDistVec<IT,std::vector<IT> > clusters(commGrid, nclusters, std::vector<IT>{});
    for(int i=0; i<clusters.LocArrSize(); i++)
    {
        clusters.SetLocalElement(i, vtxGroupbyCC[i]);
    }
    // do not write header and 1-based
    clusters.ParallelWrite(ofName, 1, HipMCLClusterSaveHandler(), false);
    
}

}

