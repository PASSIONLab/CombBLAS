#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif


// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
  typedef SpDCCols < int64_t, NT > DCCols;
  typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef PlusTimesSRing<double, double> PTFF;
typedef PlusTimesSRing<int64_t, int64_t> PTII;
typedef PlusTimesSRing<bool, double> PTBOOLNT;
typedef PlusTimesSRing<double, bool> PTNTBOOL;
typedef std::array<char, MAXVERTNAME> LBL;
typedef std::map<std::string, IT> KEYMAP;    
typedef std::pair<LBL, IT> TYPE2SEND;
typedef std::tuple<LBL, IT, IT, IT> TUPLE2SEND; // Label, gidx on M11, gidx in M22, gidx in Mixed

void SendLblNIdxToOwner(FullyDistVec<IT, LBL> &Lbl, 
                     std::vector<int> &sendcnt, 
                     std::vector<int> &recvcnt,
                     std::vector<int> &sdispls,
                     std::vector<int> &rdispls,
                     IT &totsend,
                     IT &totrecv,
                     std::vector<TYPE2SEND> &senddata,
                     std::vector<TYPE2SEND> &recvdata){
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    auto commGrid = Lbl.getcommgrid();

    std::vector< KEYMAP > keys(nprocs); 
    std::vector<LBL> lLbl = Lbl.GetLocVec();
    
    for(IT i = 0; i < lLbl.size(); i++){
        LBL lbl = lLbl[i];
        auto lblStart = lbl.begin();
        auto lblEnd = std::find(lbl.begin(), lbl.end(), '\0');
        std::string lblStr( lblStart, lblEnd );

        int owner = 0;
        for (int j = 0; j < lLbl[i].size(); j++){
            owner += (lLbl[i][j]*107) & (nprocs-1);
            if(lLbl[i][j] == '\0') break;
        }
        owner = owner % nprocs;
        
        IT gIdx = Lbl.LengthUntil() + i;
        keys[owner].insert( std::make_pair(lblStr, gIdx) ); 
        
        ////std::string seqLbl = std::string("iso_pr_archaea|8061901736|8061903926|");
        ////std::string seqLbl = std::string("iso_pr_archaea|8058535375|8058537160|");
        //std::string seqLbl = std::string("UPI00001A7D32");
        //auto res = std::mismatch(seqLbl.begin(), seqLbl.end(), lblStr.begin());
        //if( res.first == seqLbl.end() ){
            //printf("%s going to be processed at process %d\n", str.c_str(), owner);
        //}
    }

    for(int i=0; i<nprocs; ++i) sendcnt[i] = keys[i].size();

    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, commGrid->GetWorld()); // share the counts

    sdispls[0] = 0;
    rdispls[0] = 0;
    std::partial_sum(sendcnt.begin(), sendcnt.end(), sdispls.begin() + 1 );
    std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin() + 1 );
    totsend = static_cast<IT>(sdispls[nprocs]);
    totrecv = static_cast<IT>(rdispls[nprocs]);
    ////fprintf(stderr, "myrank %d totsend: %lld, totrecv %lld\n", myrank, totsend, totrecv);
    //if(myrank == 0){
        //printf("sendcnt[%d]\t+\tsdispls[%d]\t=\t%d\t+\t%d\t=\t%d\n", 0, -1, sendcnt[0], 0, sdispls[0]);
        //for(int p = 1; p < nprocs; p++){
            //printf("sendcnt[%d]\t+\tsdispls[%d]\t=\t%d\t+\t%d\t=\t%d\n", p, p, sendcnt[p], sdispls[p-1], sdispls[p]);
        //}
        //printf("sendcnt[%d]\t+\tsdispls[%d]\t=\t%d\t+\t%d\t=\t%d\n", 0, -1, sendcnt[0], 0, sdispls[0]);
    //}

    senddata.resize(totsend);
    recvdata.resize(totrecv);

    int sdidx = 0;
    for (int p = 0; p < nprocs; p++){
        int cnt = 0;
        for(auto const& elem : keys[p]){
            LBL lbl;
            lbl.fill('\0');
            std::copy( elem.first.begin(), elem.first.end(), lbl.begin() );  
            //if(elem.first.length() < MAXVERTNAME)  lbl[elem.first.length()] = '\0';	// null termination		
            sdidx = sdispls[p] + cnt;
            senddata[sdidx] = TYPE2SEND(lbl, elem.second);
            string str(senddata[sdidx].first.begin(), senddata[sdidx].first.end());
            cnt++;
        }
    }

    MPI_Datatype MPI_PAIR;
    MPI_Type_contiguous(sizeof(TYPE2SEND), MPI_CHAR, &MPI_PAIR);
    MPI_Type_commit(&MPI_PAIR);

    MPI_Alltoallv(senddata.data(), sendcnt.data(), sdispls.data(), MPI_PAIR, recvdata.data(), recvcnt.data(), rdispls.data(), MPI_PAIR, commGrid->GetWorld());
    
    MPI_Type_free(&MPI_PAIR);

    return;
}

int main(int argc, char* argv[])
{
  int nprocs, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(argc < 19){
  //if(argc < 7){
    if(myrank == 0)
    {
      cout << "Not enough arguments" << endl;
    }
    MPI_Finalize();
    return -1;
  }
  else {
    string TriplesM11;
    string TriplesM22;
    string TriplesM21;
    string MtxM11;
    string MtxM22;
    string MtxM12;
    string MtxM21;
    string LblM11;
    string LblM22;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"--triples-m11")==0){
            TriplesM11 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--triples-m22")==0){
            TriplesM22 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--triples-m21")==0){
            TriplesM21 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--mtx-m11")==0){
            MtxM11 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--mtx-m12")==0){
            MtxM12 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--mtx-m21")==0){
            MtxM21 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--mtx-m22")==0){
            MtxM22 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--lbl-m11")==0){
            LblM11 = string(argv[i+1]);
        }
        if (strcmp(argv[i],"--lbl-m22")==0){
            LblM22 = string(argv[i+1]);
        }
    }
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

    double t0, t1, t2;

    SpParMat <IT, NT, DER> M11(fullWorld); 
    SpParMat <IT, NT, DER> M22(fullWorld); 
    SpParMat <IT, NT, DER> Mixed(fullWorld); 
    
    FullyDistVec<IT, LBL> gM11Lbl = M11.ReadGeneralizedTuples(TriplesM11,  maximum<double>());
    FullyDistVec<IT, LBL> gM22Lbl = M22.ReadGeneralizedTuples(TriplesM22,  maximum<double>());
    FullyDistVec<IT, LBL> gMixedLbl = Mixed.ReadGeneralizedTuples(TriplesM21,  maximum<double>());

    M11.PrintInfo();
    M22.PrintInfo();
    Mixed.PrintInfo();

    std::vector<int> M11_sendcnt(nprocs); 
    std::vector<int> M11_recvcnt(nprocs);
    std::vector<int> M11_sdispls(nprocs+1);
    std::vector<int> M11_rdispls(nprocs+1);
    IT M11_totsend;
    IT M11_totrecv;
    std::vector<TYPE2SEND> M11_senddata;
    std::vector<TYPE2SEND> M11_recvdata;
    SendLblNIdxToOwner(gM11Lbl, M11_sendcnt, M11_recvcnt, M11_sdispls, M11_rdispls, M11_totsend, M11_totrecv, M11_senddata, M11_recvdata);

    std::vector<int> M22_sendcnt(nprocs); 
    std::vector<int> M22_recvcnt(nprocs);
    std::vector<int> M22_sdispls(nprocs+1);
    std::vector<int> M22_rdispls(nprocs+1);
    IT M22_totsend;
    IT M22_totrecv;
    std::vector<TYPE2SEND> M22_senddata;
    std::vector<TYPE2SEND> M22_recvdata;
    SendLblNIdxToOwner(gM22Lbl, M22_sendcnt, M22_recvcnt, M22_sdispls, M22_rdispls, M22_totsend, M22_totrecv, M22_senddata, M22_recvdata);

    std::vector<int> Mixed_sendcnt(nprocs); 
    std::vector<int> Mixed_recvcnt(nprocs);
    std::vector<int> Mixed_sdispls(nprocs+1);
    std::vector<int> Mixed_rdispls(nprocs+1);
    IT Mixed_totsend;
    IT Mixed_totrecv;
    std::vector<TYPE2SEND> Mixed_senddata;
    std::vector<TYPE2SEND> Mixed_recvdata;
    SendLblNIdxToOwner(gMixedLbl, Mixed_sendcnt, Mixed_recvcnt, Mixed_sdispls, Mixed_rdispls, Mixed_totsend, Mixed_totrecv, Mixed_senddata, Mixed_recvdata);

    KEYMAP M11_labelMap;
    for(IT i = 0; i < M11_totrecv; i++) {
        LBL lbl = M11_recvdata[i].first;
        auto lblStart = lbl.begin();
        auto lblEnd = std::find(lbl.begin(), lbl.end(), '\0');
        std::string lblStr( lblStart, lblEnd );
        IT v = M11_recvdata[i].second;
        M11_labelMap.insert({lblStr, v});
    }

    KEYMAP M22_labelMap;
    for(IT i = 0; i < M22_totrecv; i++) {
        LBL lbl = M22_recvdata[i].first;
        auto lblStart = lbl.begin();
        auto lblEnd = std::find(lbl.begin(), lbl.end(), '\0');
        std::string lblStr( lblStart, lblEnd );
        IT v = M22_recvdata[i].second;
        M22_labelMap.insert({lblStr, v});
    }
    
    std::vector< TUPLE2SEND > Mixed_mapping_send(Mixed_totrecv);
    int cnt = 0;
    for (int p = 0; p < nprocs; p++){
        for (int i = Mixed_rdispls[p]; i < Mixed_rdispls[p+1]; i++){
            LBL lbl = Mixed_recvdata[i].first;
            auto lblStart = lbl.begin();
            auto lblEnd = std::find(lbl.begin(), lbl.end(), '\0');
            std::string lblStr( lblStart, lblEnd );
            IT v = Mixed_recvdata[i].second;
            auto M11_search = M11_labelMap.find(lblStr);
            cnt++;
            if(M11_search != M11_labelMap.end()){
                Mixed_mapping_send[i] = TUPLE2SEND(lbl, M11_search->second, -1, v);
            }
            else{
                auto M22_search = M22_labelMap.find(lblStr);
                if(M22_search != M22_labelMap.end()){
                    Mixed_mapping_send[i] = TUPLE2SEND(lbl, -1, M22_search->second, v) ;
                }
                else{
                    Mixed_mapping_send[i] = TUPLE2SEND(lbl, -1, -1, v) ;
                }
            }
        }
    }

    std::vector< TUPLE2SEND > Mixed_mapping_recv(Mixed_totsend);
    MPI_Datatype MPI_TUPLE;
    MPI_Type_contiguous(sizeof(TUPLE2SEND), MPI_CHAR, &MPI_TUPLE);
    MPI_Type_commit(&MPI_TUPLE);
    MPI_Alltoallv(Mixed_mapping_send.data(), Mixed_recvcnt.data(), Mixed_rdispls.data(), MPI_TUPLE, Mixed_mapping_recv.data(), Mixed_sendcnt.data(), Mixed_sdispls.data(), MPI_TUPLE, fullWorld->GetWorld());
    MPI_Type_free(&MPI_TUPLE);

    std::vector<IT> mixed_idx1_local;
    std::vector<IT> mixed_idx2_local;
    std::vector<IT> mixed_idx3_local;
    std::vector<IT> idx1_local;
    std::vector<IT> idx2_local;
    std::vector<IT> idx3_local;
    std::vector<LBL> lbl3_local;
    for(IT i = 0; i < Mixed_totsend; i++){
        LBL lbl = std::get<0>(Mixed_mapping_recv[i]);
        IT mixed_idx1 = std::get<1>(Mixed_mapping_recv[i]);
        IT mixed_idx2 = std::get<2>(Mixed_mapping_recv[i]);
        IT mixed_idx3 = std::get<3>(Mixed_mapping_recv[i]);

        if(mixed_idx1 != -1) {
            mixed_idx1_local.push_back(mixed_idx3);
            idx1_local.push_back(mixed_idx1);
        }
        if(mixed_idx2 != -1) {
            mixed_idx2_local.push_back(mixed_idx3);
            idx2_local.push_back(mixed_idx2);
        }
        if(mixed_idx1 == -1 && mixed_idx2 == -1){
            mixed_idx3_local.push_back(mixed_idx3);
            lbl3_local.push_back(lbl);
        }
    }

    FullyDistVec<IT, IT> mixed_idx1(mixed_idx1_local, fullWorld);
    FullyDistVec<IT, IT> mixed_idx2(mixed_idx2_local, fullWorld);
    FullyDistVec<IT, IT> mixed_idx3(mixed_idx3_local, fullWorld);

    if(myrank == 0){
        printf("mixed_idx1: %lld elements\n", mixed_idx1.TotalLength());
        printf("mixed_idx2: %lld elements\n", mixed_idx2.TotalLength());
        printf("mixed_idx3: %lld elements\n", mixed_idx3.TotalLength());
    }

    SpParMat<IT, NT, DER> M21_small = Mixed.SubsRef_SR<PTNTBOOL, PTBOOLNT>(mixed_idx2, mixed_idx1, false); 
    SpParMat<IT, NT, DER> M31_small = Mixed.SubsRef_SR<PTNTBOOL, PTBOOLNT>(mixed_idx3, mixed_idx1, false); 
    SpParMat<IT, NT, DER> M32_small = Mixed.SubsRef_SR<PTNTBOOL, PTBOOLNT>(mixed_idx3, mixed_idx2, false); 

    FullyDistVec<IT, IT> idx1(idx1_local, fullWorld);
    FullyDistVec<IT, IT> idx2(idx2_local, fullWorld);
    FullyDistVec<IT, IT> idx3(fullWorld);
    idx3.iota(mixed_idx3.TotalLength(), gM22Lbl.TotalLength());
    FullyDistVec<IT, LBL> lbl3(lbl3_local, fullWorld);

    //std::ostringstream outs;
    //outs << "gM11Lbl: " << gM11Lbl.TotalLength() << ", gM22Lbl: " << gM22Lbl.TotalLength() << ", idx3: " << idx3.TotalLength() << ", lbl3: " << lbl3.TotalLength() << "\n";
    //SpParHelper::Print(outs.str());

    SpParMat<IT, NT, DER> M231_big = SpParMat<IT,NT,DER>(gM22Lbl.TotalLength() + lbl3.TotalLength(), 
                     gM11Lbl.TotalLength(), 
                     FullyDistVec<IT,IT>(fullWorld), 
                     FullyDistVec<IT,IT>(fullWorld), 
                     FullyDistVec<IT,IT>(fullWorld), true);
    {
        SpParMat<IT, NT, DER> M21_big = SpParMat<IT,NT,DER>(gM22Lbl.TotalLength() + lbl3.TotalLength(), 
                         gM11Lbl.TotalLength(), 
                         FullyDistVec<IT,IT>(fullWorld), 
                         FullyDistVec<IT,IT>(fullWorld), 
                         FullyDistVec<IT,IT>(fullWorld), true);
        M21_big.SpAsgn(idx2, idx1, M21_small);
        SpParMat<IT, NT, DER> M31_big = SpParMat<IT,NT,DER>(gM22Lbl.TotalLength() + lbl3.TotalLength(), 
                         gM11Lbl.TotalLength(), 
                         FullyDistVec<IT,IT>(fullWorld), 
                         FullyDistVec<IT,IT>(fullWorld), 
                         FullyDistVec<IT,IT>(fullWorld), true);
        M31_big.SpAsgn(idx3, idx1, M31_small);
        M231_big = M21_big;
        M231_big += M31_big;
    }
    //M231_big.PrintInfo();

    SpParMat<IT, NT, DER> M23_big = SpParMat<IT,NT,DER>(gM22Lbl.TotalLength() + lbl3.TotalLength(), 
                     gM22Lbl.TotalLength() + lbl3.TotalLength(), 
                     FullyDistVec<IT,IT>(fullWorld), 
                     FullyDistVec<IT,IT>(fullWorld), 
                     FullyDistVec<IT,IT>(fullWorld), true);
    {
        FullyDistVec<IT, IT> idx2_all(fullWorld);
        idx2_all.iota(gM22Lbl.TotalLength(), 0);
        M23_big.SpAsgn(idx2_all, idx2_all, M22);
        if (M32_small.getnnz() > 0){
            M23_big.SpAsgn(idx3, idx2, M32_small);
        }
    }
    //M23_big.PrintInfo();

    FullyDistVec<IT, LBL> gM23Lbl(fullWorld);
    {
        std::vector<FullyDistVec<IT, LBL>> toConcatenateLabels; 
        toConcatenateLabels.push_back(gM22Lbl);
        toConcatenateLabels.push_back(lbl3);
        gM23Lbl = Concatenate(toConcatenateLabels);
    }
    //gM23Lbl.ParallelWrite("gM23Lbl.triples", true);
    
    M22 = M23_big;
    SpParMat<IT, NT, DER> M21 = M231_big;
    SpParMat<IT, NT, DER> M12(M21);
    M12.Transpose();
    
    M11.PrintInfo();
    M12.PrintInfo();
    M21.PrintInfo();
    M22.PrintInfo();
    
    //printf("%s\n", MtxM11.c_str());
    M11.ParallelWriteMM(MtxM11, 1);
    M12.ParallelWriteMM(MtxM12, 1);
    M21.ParallelWriteMM(MtxM21, 1);
    M22.ParallelWriteMM(MtxM22, 1);
    gM11Lbl.ParallelWrite(LblM11, 1);
    gM22Lbl.ParallelWrite(LblM22, 1);

  }


  MPI_Finalize();
  return 0;
}
