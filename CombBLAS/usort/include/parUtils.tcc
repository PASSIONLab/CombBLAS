
/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
 */

#include "binUtils.h"
#include "seqUtils.h"
#include "dtypes.h"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstring>


#include "indexHolder.h"

#ifdef _PROFILE_SORT
  #include "sort_profiler.h"
#endif

#include "ompUtils.h"


#include <mpi.h>

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

#ifndef KWAY
		#define KWAY 64
#endif 

namespace par {

  template <typename T>
    inline int Mpi_Isend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Isend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Issend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Issend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Recv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Status* status) {

      MPI_Recv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, status);

      return 1;

    }

  template <typename T>
    inline int Mpi_Irecv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Irecv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, request);

      return 1;

    }

  template <typename T, typename S>
    inline int Mpi_Sendrecv( T* sendBuf, int sendCount, int dest, int sendTag,
        S* recvBuf, int recvCount, int source, int recvTag,
        MPI_Comm comm, MPI_Status* status) {

        MPI_Sendrecv(sendBuf, sendCount, par::Mpi_datatype<T>::value(), dest, sendTag,
            recvBuf, recvCount, par::Mpi_datatype<S>::value(), source, recvTag, comm, status);

    }

  template <typename T>
    inline int Mpi_Scan( T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
        MPI_Scan(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);
    }

  template <typename T>
    inline int Mpi_Allreduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Allreduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);
    }

  template <typename T>
    inline int Mpi_Alltoall(T* sendbuf, T* recvbuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Alltoall(sendbuf, count, par::Mpi_datatype<T>::value(),
            recvbuf, count, par::Mpi_datatype<T>::value(), comm);

    }

  template <typename T>
    inline int Mpi_Alltoallv
    (T* sendbuf, int* sendcnts, int* sdispls, 
     T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Alltoallv(
            sendbuf, sendcnts, sdispls, par::Mpi_datatype<T>::value(), 
            recvbuf, recvcnts, rdispls, par::Mpi_datatype<T>::value(), 
            comm);
        return 0;
    }

  template <typename T>
    inline int Mpi_Gather( T* sendBuffer, T* recvBuffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Gather(sendBuffer, count, par::Mpi_datatype<T>::value(),
            recvBuffer, count, par::Mpi_datatype<T>::value(), root, comm);

    }

  template <typename T>
    inline int Mpi_Bcast(T* buffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Bcast(buffer, count, par::Mpi_datatype<T>::value(), root, comm);

    }

  template <typename T>
    inline int Mpi_Reduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Reduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, root, comm);

    }

  template <typename T>
    int Mpi_Allgatherv(T* sendBuf, int sendCount, T* recvBuf, 
        int* recvCounts, int* displs, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

      int maxSendCount;
      int npes, rank;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      par::Mpi_Allreduce<int>(&sendCount, &maxSendCount, 1, MPI_MAX, comm);

      T* dummySendBuf = new T[maxSendCount*npes];
      assert(dummySendBuf);

      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < sendCount; j++) {
          dummySendBuf[(i*maxSendCount) + j] = sendBuf[j];
        }
      }

      T* dummyRecvBuf = new T[maxSendCount*npes];
      assert(dummyRecvBuf);

      par::Mpi_Alltoall<T>(dummySendBuf, dummyRecvBuf, maxSendCount, comm);

      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < recvCounts[i]; j++) {
          recvBuf[displs[i] + j] = dummyRecvBuf[(i*maxSendCount) + j];
        }
      }

      delete [] dummySendBuf;
      delete [] dummyRecvBuf;

#else

      MPI_Allgatherv(sendBuf, sendCount, par::Mpi_datatype<T>::value(),
          recvBuf, recvCounts, displs, par::Mpi_datatype<T>::value(), comm);

#endif

    }

  template <typename T>
    int Mpi_Allgather(T* sendBuf, T* recvBuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      
#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

      int npes;
      MPI_Comm_size(comm, &npes);
      T* dummySendBuf = new T[count*npes];
      assert(dummySendBuf);
      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < count; j++) {
          dummySendBuf[(i*count) + j] = sendBuf[j];
        }
      }
      par::Mpi_Alltoall<T>(dummySendBuf, recvBuf, count, comm);
      delete [] dummySendBuf;

#else

      MPI_Allgather(sendBuf, count, par::Mpi_datatype<T>::value(), 
          recvBuf, count, par::Mpi_datatype<T>::value(), comm);

#endif

    }

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
    
#ifndef ALLTOALLV_FIX
      Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls, 
         recvbuf, recvcnts, rdispls, comm);
#else

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      int commCnt = 0;

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = (rank+1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      MPI_Request* requests = new MPI_Request[commCnt];
      assert(requests);

      MPI_Status* statuses = new MPI_Status[commCnt];
      assert(statuses);

      commCnt = 0;

      //First place all recv requests. Do not recv from self.
      for(int i = 0; i < rank; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Next send the messages. Do not send to self.
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], 
              i, 1, comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Now copy local portion.
#ifdef __DEBUG_PAR__
      assert(sendcnts[rank] == recvcnts[rank]);
#endif

      #pragma omp parallel for
      for(int i = 0; i < sendcnts[rank]; i++) {
        recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
      }
      
      MPI_Waitall(commCnt, requests, statuses);

      delete [] requests;
      delete [] statuses;
#endif
    }

//*
  template <typename T>
    int Mpi_Alltoallv_dense(T* sbuff_, int* s_cnt_, int* sdisp_,
        T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c){

#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      
#ifndef ALLTOALLV_FIX
      Mpi_Alltoallv
        (sbuff_, s_cnt_, sdisp_,
         rbuff_, r_cnt_, rdisp_, c);
#else
  int kway = KWAY;
  int np, pid;
  MPI_Comm_size(c, &np);
  MPI_Comm_rank(c, &pid);
  int range[2]={0,np};
  int split_id, partner;

  std::vector<int> s_cnt(np);
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    s_cnt[i]=s_cnt_[i]*sizeof(T)+2*sizeof(int);
  }
  std::vector<int> sdisp(np); sdisp[0]=0;
  omp_par::scan(&s_cnt[0],&sdisp[0],np);

  char* sbuff=new char[sdisp[np-1]+s_cnt[np-1]];
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    ((int*)&sbuff[sdisp[i]])[0]=s_cnt[i];
    ((int*)&sbuff[sdisp[i]])[1]=pid;
    memcpy(&sbuff[sdisp[i]]+2*sizeof(int),&sbuff_[sdisp_[i]],s_cnt[i]-2*sizeof(int));
  }

  //int t_indx=0;
  int iter_cnt=0;
  while(range[1]-range[0]>1){
    iter_cnt++;
    if(kway>range[1]-range[0]) 
      kway=range[1]-range[0];

    std::vector<int> new_range(kway+1);
    for(int i=0;i<=kway;i++)
      new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
    int p_class=(std::upper_bound(&new_range[0],&new_range[kway],pid)-&new_range[0]-1);
    int new_np=new_range[p_class+1]-new_range[p_class];
    int new_pid=pid-new_range[p_class];

    //Communication.
    {
      std::vector<int> r_cnt    (new_np*kway, 0);
      std::vector<int> r_cnt_ext(new_np*kway, 0);
      //Exchange send sizes.
      for(int i=0;i<kway;i++){
        MPI_Status status;
        int cmp_np=new_range[i+1]-new_range[i];
        int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;
        assert(     (new_pid<cmp_np? true: new_range[i]+new_pid==new_range[i+1]  )); //Remove this.
        MPI_Sendrecv(&s_cnt[new_range[i]-new_range[0]], cmp_np, MPI_INT, partner, 0,
                     &r_cnt[new_np   *i ], new_np, MPI_INT, partner, 0, c, &status);

        //Handle extra communication.
        if(new_pid==new_np-1 && cmp_np>new_np){
          int partner=new_range[i+1]-1;
          std::vector<int> s_cnt_ext(cmp_np, 0);
          MPI_Sendrecv(&s_cnt_ext[       0], cmp_np, MPI_INT, partner, 0,
                       &r_cnt_ext[new_np*i], new_np, MPI_INT, partner, 0, c, &status);
        }
      }

      //Allocate receive buffer.
      std::vector<int> rdisp    (new_np*kway, 0);
      std::vector<int> rdisp_ext(new_np*kway, 0);
      int rbuff_size, rbuff_size_ext;
      char *rbuff, *rbuff_ext;
      {
        omp_par::scan(&r_cnt    [0], &rdisp    [0],new_np*kway);
        omp_par::scan(&r_cnt_ext[0], &rdisp_ext[0],new_np*kway);
        rbuff_size     = rdisp    [new_np*kway-1] + r_cnt    [new_np*kway-1];
        rbuff_size_ext = rdisp_ext[new_np*kway-1] + r_cnt_ext[new_np*kway-1];
        rbuff     = new char[rbuff_size    ];
        rbuff_ext = new char[rbuff_size_ext];
      }

      //Sendrecv data.
      //*
      int my_block=kway;
      while(pid<new_range[my_block]) my_block--;
//      MPI_Barrier(c);
      for(int i_=0;i_<=kway/2;i_++){
        int i1=(my_block+i_)%kway;
        int i2=(my_block+kway-i_)%kway;

        for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
          int i=(i_==0?i1:((j+my_block/i_)%2?i1:i2));
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;

          int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
          int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
          int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;

//          ttt=omp_get_wtime();
          MPI_Sendrecv(&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0,
                       &rbuff[rdisp[new_np  * i ]], r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ], MPI_BYTE, partner, 0, c, &status);

          //Handle extra communication.
          if(pid==new_np-1 && cmp_np>new_np){
            int partner=new_range[i+1]-1;
            std::vector<int> s_cnt_ext(cmp_np, 0);
            MPI_Sendrecv(                       NULL,                                                                       0, MPI_BYTE, partner, 0,
                         &rbuff[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i], MPI_BYTE, partner, 0, c, &status);
          }
        }
      }

      //Rearrange received data.
      {
        if(sbuff!=NULL) delete[] sbuff;
        sbuff=new char[rbuff_size+rbuff_size_ext];

        std::vector<int>  cnt_new(2*new_np*kway, 0);
        std::vector<int> disp_new(2*new_np*kway, 0);
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          cnt_new[(i*2  )*kway+j]=r_cnt    [j*new_np+i];
          cnt_new[(i*2+1)*kway+j]=r_cnt_ext[j*new_np+i];
        }
        omp_par::scan(&cnt_new[0], &disp_new[0],2*new_np*kway);

        #pragma omp parallel for
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          memcpy(&sbuff[disp_new[(i*2  )*kway+j]], &rbuff    [rdisp    [j*new_np+i]], r_cnt    [j*new_np+i]);
          memcpy(&sbuff[disp_new[(i*2+1)*kway+j]], &rbuff_ext[rdisp_ext[j*new_np+i]], r_cnt_ext[j*new_np+i]);
        }

        //Free memory.
        if(rbuff    !=NULL) delete[] rbuff    ;
        if(rbuff_ext!=NULL) delete[] rbuff_ext;

        s_cnt.clear();
        s_cnt.resize(new_np,0);
        sdisp.resize(new_np);
        for(int i=0;i<new_np;i++){
          for(int j=0;j<2*kway;j++)
            s_cnt[i]+=cnt_new[i*2*kway+j];
          sdisp[i]=disp_new[i*2*kway];
        }
      }
    }

    range[0]=new_range[p_class  ];
    range[1]=new_range[p_class+1];
  }

  //Copy data to rbuff_.
  std::vector<char*> buff_ptr(np);
  char* tmp_ptr=sbuff;
  for(int i=0;i<np;i++){
    int& blk_size=((int*)tmp_ptr)[0];
    buff_ptr[i]=tmp_ptr;
    tmp_ptr+=blk_size;
  }
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    int& blk_size=((int*)buff_ptr[i])[0];
    int& src_pid=((int*)buff_ptr[i])[1];
    assert(blk_size-2*sizeof(int)<=r_cnt_[src_pid]*sizeof(T));
    memcpy(&rbuff_[rdisp_[src_pid]],buff_ptr[i]+2*sizeof(int),blk_size-2*sizeof(int));
  }

  //Free memory.
  if(sbuff   !=NULL) delete[] sbuff;
#endif

    }

		
	template<typename T>
    unsigned int defaultWeight(const T *a){
      return 1;
    }	
		
  template<typename T>
    int partitionW(std::vector<T>& nodeList, unsigned int (*getWeight)(const T *), MPI_Comm comm){
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      int npes;

      MPI_Comm_size(comm, &npes);

      if(getWeight == NULL) {
        getWeight = par::defaultWeight<T>;
      }

      int rank;

      MPI_Comm_rank(comm, &rank);

      MPI_Request request;
      MPI_Status status;
      const bool nEmpty = nodeList.empty();

      DendroIntL  off1= 0, off2= 0, localWt= 0, totalWt = 0;

      DendroIntL* wts = NULL;
      DendroIntL* lscn = NULL;
      DendroIntL nlSize = nodeList.size();
      if(nlSize) {
        wts = new DendroIntL[nlSize];
        assert(wts);

        lscn= new DendroIntL[nlSize]; 
        assert(lscn);
      }

      // First construct arrays of id and wts.
      #pragma omp parallel for reduction(+:localWt)
      for (DendroIntL i = 0; i < nlSize; i++){
        wts[i] = (*getWeight)( &(nodeList[i]) );
        localWt+=wts[i];
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // compute the total weight of the problem ...
      par::Mpi_Allreduce<DendroIntL>(&localWt, &totalWt, 1, MPI_SUM, comm);

      // perform a local scan on the weights first ...
      DendroIntL zero = 0;
      if(!nEmpty) {
        lscn[0]=wts[0];
//        for (DendroIntL i = 1; i < nlSize; i++) {
//          lscn[i] = wts[i] + lscn[i-1];
//        }//end for
        omp_par::scan(&wts[1],lscn,nlSize);
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(lscn+nlSize-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, rank+1, 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, rank-1, 0, comm, &status );
      }
      else{
        off2 = 0; 
      }

      // add offset to local array
      #pragma omp parallel for
      for (DendroIntL i = 0; i < nlSize; i++) {
        lscn[i] = lscn[i] + off2;       // This has the global scan results now ...
      }//end for

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-2 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      int * sendSz = new int [npes];
      assert(sendSz);

      int * recvSz = new int [npes];
      assert(recvSz);

      int * sendOff = new int [npes]; 
      assert(sendOff);
      sendOff[0] = 0;

      int * recvOff = new int [npes]; 
      assert(recvOff);
      recvOff[0] = 0;

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...

      #pragma omp parallel for
      for (int i = 0; i < npes; i++) {
        sendSz[i] = 0;
      }

      // Now determine the average load ...
      DendroIntL npesLong = npes;
      DendroIntL avgLoad = (totalWt/npesLong);

      DendroIntL extra = (totalWt%npesLong);

      //The Heart of the algorithm....
      if(avgLoad > 0) {
        for (DendroIntL i = 0; i < nlSize; i++) {
          if(lscn[i] == 0) {
            sendSz[0]++;
          }else {
            int ind=0;
            if ( lscn[i] <= (extra*(avgLoad + 1)) ) {
              ind = ((lscn[i] - 1)/(avgLoad + 1));
            }else {
              ind = ((lscn[i] - (1 + extra))/avgLoad);
            }
            assert(ind < npes);
            sendSz[ind]++;
          }//end if-else
        }//end for */ 
/*
        //This is more effecient and parallelizable than the above.
        //This has a bug trying a simpler approach below.
        int ind_min,ind_max;
        ind_min=(lscn[0]*npesLong)/totalWt-1;
        ind_max=(lscn[nlSize-1]*npesLong)/totalWt+2;
        if(ind_min< 0       )ind_min=0;
        if(ind_max>=npesLong)ind_max=npesLong;
        #pragma omp parallel for
        for(int i=ind_min;i<ind_max;i++){
          DendroIntL wt1=(totalWt*i)/npesLong;
          DendroIntL wt2=(totalWt*(i+1))/npesLong;
          int end = std::upper_bound(&lscn[0], &lscn[nlSize], wt2, std::less<DendroIntL>())-&lscn[0];
          int start = std::upper_bound(&lscn[0], &lscn[nlSize], wt1, std::less<DendroIntL>())-&lscn[0];
          if(i==npesLong-1)end  =nlSize;
          if(i==         0)start=0     ;
          sendSz[i]=end-start;
        }// */

#ifdef __DEBUG_PAR__
        int tmp_sum=0;
        for(int i=0;i<npes;i++) tmp_sum+=sendSz[i];
        assert(tmp_sum==nlSize);
#endif

      }else {
        sendSz[0]+= nlSize;
      }//end if-else

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-3 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      if(rank < (npes-1)) {
        MPI_Status statusWait;
        MPI_Wait(&request, &statusWait);
      }

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

#ifdef __DEBUG_PAR__
      DendroIntL totSendToOthers = 0;
      DendroIntL totRecvFromOthers = 0;
      for (int i = 0; i < npes; i++) {
        if(rank != i) {
          totSendToOthers += sendSz[i];
          totRecvFromOthers += recvSz[i];
        }
      }
#endif

      DendroIntL nn=0; // new value of nlSize, ie the local nodes.
      #pragma omp parallel for reduction(+:nn)
      for (int i = 0; i < npes; i++) {
        nn += recvSz[i];
      }

      // compute offsets ...
      omp_par::scan(sendSz,sendOff,npes);
      omp_par::scan(recvSz,recvOff,npes);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-4 passed."<<std::endl;
      }
			MPI_Barrier(comm);
#endif

      // allocate memory for the new arrays ...
      std::vector<T > newNodes(nn);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Final alloc successful."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // perform All2All  ... 
      T* nodeListPtr = NULL;
      T* newNodesPtr = NULL;
      if(!nodeList.empty()) {
        nodeListPtr = &(*(nodeList.begin()));
      }
      if(!newNodes.empty()) {
        newNodesPtr = &(*(newNodes.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(nodeListPtr, sendSz, sendOff, 
          newNodesPtr, recvSz, recvOff, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-5 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // reset the pointer ...
      swap(nodeList, newNodes);
      newNodes.clear();

      // clean up...
      if(!nEmpty) {
        delete [] lscn;
        delete [] wts;
      }
      delete [] sendSz;
      sendSz = NULL;

      delete [] sendOff;
      sendOff = NULL;

      delete [] recvSz;
      recvSz = NULL;

      delete [] recvOff;
      recvOff = NULL;

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-6 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

    }//end function

  /* Hari Notes ....
   *
   * Avoid unwanted allocations within Hypersort ...
   * 
   * 1. try to sort in place ... no output buffer, user can create a copy if
   *    needed.
   * 2. have a std::vector<T> container for rbuff. the space required can be 
   *    reserved before doing MPI_SendRecv
   * 3. alternatively, keep a send buffer and recv into original buffer. 
   *
   */ 
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
#ifdef _PROFILE_SORT
    long bytes_comm=0;
    total_sort.start();
#endif


      // Copy communicator. 
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank, rank_;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      rank_ = myrank;

      if(npes==1){
#ifdef _PROFILE_SORT
				seq_sort.start();
#endif        
				omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
				seq_sort.stop();
				total_sort.stop();
#endif        
      }
      // buffers ... keeping all allocations together 
      std::vector<T>  commBuff;
      std::vector<T>  mergeBuff;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);


      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      long totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<long>(&nelem, &totSize, 1, MPI_SUM, comm);
      long nelem_ = nelem;

      // Local sort.
#ifdef _PROFILE_SORT
			seq_sort.start();
#endif			
      omp_par::merge_sort(&arr[0], &arr[arr.size()]);
#ifdef _PROFILE_SORT
			seq_sort.stop();
#endif			

      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
#ifdef _PROFILE_SORT
    hyper_compute_splitters.start();
#endif				
        T split_key;
        long totSize_new;
        //while(true)
        { 
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count = (1000*nelem)/totSize; 
          if (npes>1000) 
            splt_count = ( ((float)rand()/(float)RAND_MAX)*totSize < (1000*nelem) ? 1 : 0 );
          
          if ( splt_count > nelem ) 
						splt_count = nelem;
          
          std::vector<T> splitters(splt_count);
          for(size_t i=0;i<splt_count;i++) 
            splitters[i]=arr[rand()%nelem];

          // Gather all splitters. O( log(p) )
          int glb_splt_count;

          par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
         
          glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];

          std::vector<T> glb_splitters(glb_splt_count);
          
          MPI_Allgatherv(&splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                         &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                         par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<long> disp(glb_splt_count,0);
          
          if(nelem>0){
            #pragma omp parallel for
            for(size_t i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
            }
          }
          std::vector<long> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<long>::value(), MPI_SUM, comm);

          long* split_disp = &glb_disp[0];
          for(size_t i=0; i<glb_splt_count; i++)
            if ( abs(glb_disp[i] - totSize/2) < abs(*split_disp - totSize/2) ) 
							split_disp = &glb_disp[i];
          split_key = glb_splitters[split_disp - &glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }
#ifdef _PROFILE_SORT
    hyper_compute_splitters.stop();
#endif

        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
#ifdef _PROFILE_SORT
      hyper_communicate.start();
#endif				
          int new_p0 = (myrank<=split_id?0:split_id+1);
          int cmp_p0 = (myrank> split_id?0:split_id+1);
          int new_np = (myrank<=split_id? split_id+1: npes-split_id-1);
          int cmp_np = (myrank> split_id? split_id+1: npes-split_id-1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;

          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr[0], &arr[nelem], split_key)-&arr[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr[0]   :  &arr[split_indx]);
          lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr[0]   :  &arr[split_indx]);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          commBuff.reserve(rsize/sizeof(T));
          char*     rbuff = (char *)(&commBuff[0]);
          char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);
#ifdef _PROFILE_SORT
          bytes_comm += ssize;
          hyper_communicate.stop();
          hyper_merge.start();
#endif

          int nbuff_size=lsize+rsize+ext_rsize;
          mergeBuff.reserve(nbuff_size/sizeof(T));
          char* nbuff= (char *)(&mergeBuff[0]);  // new char[nbuff_size];
          omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff!=NULL){
            // XXX case not handled 
            char* nbuff1= new char[nbuff_size];
            omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
            if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          /*
          if(arr_!=NULL) delete[] arr_; 
          arr_=(T*) nbuff; nbuff=NULL;
          */
          mergeBuff.swap(arr);

          //Free memory.
          // if(    rbuff!=NULL) delete[]     rbuff;
          if(ext_rbuff!=NULL) delete[] ext_rbuff;
#ifdef _PROFILE_SORT
      hyper_merge.stop();
#endif
        }

        {// Split comm.  O( log(p) ) ??
#ifdef _PROFILE_SORT
    hyper_comm_split.start();
#endif				
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
#ifdef _PROFILE_SORT
    hyper_comm_split.stop();
#endif				
        }
      }

      // par::partitionW<T>(SortedElem, NULL , comm_);
			// par::partitionW<T>(arr, NULL , comm_);

#ifdef _PROFILE_SORT
  total_sort.stop();

      par::Mpi_Allreduce<long>(&bytes_comm, &total_bytes, 1, MPI_SUM, comm_);
      // if(!rank_) printf("Total comm is %ld bytes\n", total_comm);
#endif
    }//end function

  //--------------------------------------------------------------------------------
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
	#ifdef _PROFILE_SORT
		 		total_sort.start();
	#endif

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank, myrank_;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank); myrank_=myrank;
      if(npes==1){
        // @dhairya isn't this wrong for the !sort-in-place case ... 
#ifdef _PROFILE_SORT
		 		seq_sort.start();
#endif        
				omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
		 		seq_sort.stop();
#endif        
				SortedElem  = arr;
#ifdef _PROFILE_SORT
		 		total_sort.stop();
#endif        
      }

      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      DendroIntL totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
      DendroIntL nelem_ = nelem;

      // Local sort.
#ifdef _PROFILE_SORT
		 	seq_sort.start();
#endif			
      T* arr_=new T[nelem]; memcpy (&arr_[0], &arr[0], nelem*sizeof(T));      
			omp_par::merge_sort(&arr_[0], &arr_[arr.size()]);
#ifdef _PROFILE_SORT
		 	seq_sort.stop();
#endif
      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.start();
#endif				
        T split_key;
        DendroIntL totSize_new;
        //while(true)
        { 
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count=(1000*nelem)/totSize; 
          if(npes>1000) splt_count=(((float)rand()/(float)RAND_MAX)*totSize<(1000*nelem)?1:0);
          if(splt_count>nelem) splt_count=nelem;
          std::vector<T> splitters(splt_count);
          for(size_t i=0;i<splt_count;i++) 
            splitters[i]=arr_[rand()%nelem];

          // Gather all splitters. O( log(p) )
          int glb_splt_count;
          std::vector<int> glb_splt_cnts(npes);
          std::vector<int> glb_splt_disp(npes,0);
          par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
          glb_splt_count=glb_splt_cnts[npes-1]+glb_splt_disp[npes-1];
          std::vector<T> glb_splitters(glb_splt_count);
          MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                         &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                         par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<DendroIntL> disp(glb_splt_count,0);
          if(nelem>0){
            #pragma omp parallel for
            for(size_t i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr_[0], &arr_[nelem], glb_splitters[i])-&arr_[0];
            }
          }
          std::vector<DendroIntL> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

          DendroIntL* split_disp=&glb_disp[0];
          for(size_t i=0;i<glb_splt_count;i++)
            if( labs(glb_disp[i]-totSize/2) < labs(*split_disp-totSize/2)) split_disp=&glb_disp[i];
          split_key=glb_splitters[split_disp-&glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }
#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.stop();
#endif
			
        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
#ifdef _PROFILE_SORT
				 	hyper_communicate.start();
#endif				
					
          int new_p0=(myrank<=split_id?0:split_id+1);
          int cmp_p0=(myrank> split_id?0:split_id+1);
          int new_np=(myrank<=split_id? split_id+1: npes-split_id-1);
          int cmp_np=(myrank> split_id? split_id+1: npes-split_id-1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;
          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr_[0], &arr_[nelem], split_key)-&arr_[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr_[0]   :  &arr_[split_indx]);
          lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr_[0]   :  &arr_[split_indx]);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          char*     rbuff=              new char[    rsize]       ;
          char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);
#ifdef _PROFILE_SORT
				 	hyper_communicate.stop();
				 	hyper_merge.start();
#endif
          int nbuff_size=lsize+rsize+ext_rsize;
          char* nbuff= new char[nbuff_size];
          omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff!=NULL){
            char* nbuff1= new char[nbuff_size];
            omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
            if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          if(arr_!=NULL) delete[] arr_; 
          arr_=(T*) nbuff; nbuff=NULL;

          //Free memory.
          if(    rbuff!=NULL) delete[]     rbuff;
          if(ext_rbuff!=NULL) delete[] ext_rbuff;
#ifdef _PROFILE_SORT
				 	hyper_merge.stop();
#endif				
        }

#ifdef _PROFILE_SORT
					hyper_comm_split.start();
#endif				
        {// Split comm.  O( log(p) ) ??
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
        }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif				
      }

      SortedElem.resize(nelem);
      SortedElem.assign(arr_, &arr_[nelem]);
      if(arr_!=NULL) delete[] arr_;

#ifdef _PROFILE_SORT
		 	sort_partitionw.start();
#endif
//      par::partitionW<T>(SortedElem, NULL , comm_);
#ifdef _PROFILE_SORT
		 	sort_partitionw.stop();
#endif

#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
    }//end function
// */

  template<typename T>
    int HyperQuickSort_kway(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_) {
#ifdef _PROFILE_SORT
		total_sort.clear();
      seq_sort.clear();
      hyper_compute_splitters.clear();
      hyper_communicate.clear();
      hyper_merge.clear();
      hyper_comm_split.clear();
      sort_partitionw.clear();
      MPI_Barrier(comm_);

      total_sort.start();
#endif
      unsigned int kway = KWAY;
      int omp_p=omp_get_max_threads();

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      srand(myrank);

      // Local and global sizes. O(log p)
      size_t totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<size_t>(&nelem, &totSize, 1, MPI_SUM, comm);
      std::vector<T> arr_(nelem*2); //Extra buffer.
      std::vector<T> arr__(nelem*2); //Extra buffer.

      // Local sort.
#ifdef _PROFILE_SORT
      seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0], &arr[arr.size()]);
#ifdef _PROFILE_SORT
      seq_sort.stop();
#endif

      while(npes>1 && totSize>0){
        if(kway>npes) kway = npes;
        int blk_size=npes/kway; assert(blk_size*kway==npes);
        int blk_id=myrank/blk_size, new_pid=myrank%blk_size;

        // Determine splitters.
#ifdef _PROFILE_SORT
        hyper_compute_splitters.start();
#endif
        std::vector<T> split_key = par::Sorted_approx_Select(arr, kway-1, comm);
#ifdef _PROFILE_SORT
        hyper_compute_splitters.stop();
#endif

        {// Communication
#ifdef _PROFILE_SORT
          hyper_communicate.start();
#endif
          // Determine send_size.
          std::vector<int> send_size(kway), send_disp(kway+1); send_disp[0]=0; send_disp[kway]=arr.size();
          for(int i=1;i<kway;i++) send_disp[i]=std::lower_bound(&arr[0], &arr[arr.size()], split_key[i-1])-&arr[0];
          for(int i=0;i<kway;i++) send_size[i]=send_disp[i+1]-send_disp[i];

          // Get recv_size.
          int recv_iter=0;
          std::vector<T*> recv_ptr(kway);
          std::vector<size_t> recv_cnt(kway);
          std::vector<int> recv_size(kway), recv_disp(kway+1,0);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            MPI_Status status;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;
              MPI_Sendrecv(&send_size[     i   ], 1, MPI_INT, partner, 0,
                           &recv_size[recv_iter], 1, MPI_INT, partner, 0, comm, &status);
              recv_disp[recv_iter+1]=recv_disp[recv_iter]+recv_size[recv_iter];
              recv_ptr[recv_iter]=&arr_[recv_disp[recv_iter]];
              recv_cnt[recv_iter]=recv_size[recv_iter];
              recv_iter++;
            }
          }

          // Communicate data.
          int asynch_count=2;
          recv_iter=0;
					int merg_indx=2;
          std::vector<MPI_Request> reqst(kway*2);
          std::vector<MPI_Status> status(kway*2);
          arr_ .resize(recv_disp[kway]);
          arr__.resize(recv_disp[kway]);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;

              if(recv_iter-asynch_count-1>=0) MPI_Waitall(2, &reqst[(recv_iter-asynch_count-1)*2], &status[(recv_iter-asynch_count-1)*2]);
              par::Mpi_Irecv <T>(&arr_[recv_disp[recv_iter]], recv_size[recv_iter], partner, 1, comm, &reqst[recv_iter*2+0]);
              par::Mpi_Issend<T>(&arr [send_disp[     i   ]], send_size[     i   ], partner, 1, comm, &reqst[recv_iter*2+1]);
              recv_iter++;

              int flag[2]={0,0};
              if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-1)*2],&flag[0],&status[(merg_indx-1)*2]);
              if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-2)*2],&flag[1],&status[(merg_indx-2)*2]);
              if(flag[0] && flag[1]){
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2;merg_indx%s==0;s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
            }
          }
#ifdef _PROFILE_SORT
				hyper_communicate.stop();
				hyper_merge.start();
#endif
					// Merge remaining parts.
          while(merg_indx<=(int)kway){
              MPI_Waitall(1, &reqst[(merg_indx-1)*2], &status[(merg_indx-1)*2]);
              MPI_Waitall(1, &reqst[(merg_indx-2)*2], &status[(merg_indx-2)*2]);
              {
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2;merg_indx%s==0;s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
          }
					{// Swap buffers.
						int swap_cond=0;
            for(int s=2;kway%s==0;s*=2) swap_cond++;
						if(swap_cond%2==0) swap(arr,arr_);
						else swap(arr,arr__);
					}
				}

#ifdef _PROFILE_SORT
				hyper_merge.stop();
				hyper_comm_split.start();
#endif
				{// Split comm. kway  O( log(p) ) ??
    	     MPI_Comm scomm;
      	   MPI_Comm_split(comm, blk_id, myrank, &scomm );
					 if(comm!=comm_) MPI_Comm_free(&comm);
        	 comm = scomm;

			     MPI_Comm_size(comm, &npes);
           MPI_Comm_rank(comm, &myrank);
    	  }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif
      }
#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
			SortedElem=arr;
    }

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
	// for sc13 --	mem effecient HykSort
  template<typename T>
    int HyperQuickSort_kway(std::vector<T>& arr, MPI_Comm comm_) {
#ifdef _PROFILE_SORT
		total_sort.clear();
      seq_sort.clear();
      hyper_compute_splitters.clear();
      hyper_communicate.clear();
      hyper_merge.clear();
      hyper_comm_split.clear();
      sort_partitionw.clear();
      MPI_Barrier(comm_);

      total_sort.start();
#endif
      unsigned int kway = KWAY;
      int omp_p = omp_get_max_threads();

      // Copy communicator.
      MPI_Comm comm = comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      
			srand(myrank);

      // Local and global sizes. O(log p)
      size_t totSize, nelem = arr.size(); assert(nelem);
			par::Mpi_Allreduce<size_t>(&nelem, &totSize, 1, MPI_SUM, comm);
      
      // dummy array for now ...
			std::vector<T> arr_(128); // (nelem*2); //Extra buffer.
      std::vector<T> arr__; // (nelem*2); //Extra buffer.

      // Local sort.
#ifdef _PROFILE_SORT
      seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0], &arr[arr.size()]);
#ifdef _PROFILE_SORT
      seq_sort.stop();
#endif

      while(npes>1 && totSize>0){
        if(kway>npes) kway = npes;
        int blk_size=npes/kway; assert(blk_size*kway==npes);
        int blk_id=myrank/blk_size, new_pid=myrank%blk_size;

        // Determine splitters.
#ifdef _PROFILE_SORT
        hyper_compute_splitters.start();
#endif
        std::vector<T> split_key = par::Sorted_approx_Select(arr, kway-1, comm);
#ifdef _PROFILE_SORT
        hyper_compute_splitters.stop();
#endif

        {// Communication
#ifdef _PROFILE_SORT
          hyper_communicate.start();
#endif
          // Determine send_size.
          std::vector<int> send_size(kway), send_disp(kway+1); send_disp[0] = 0; send_disp[kway] = arr.size();
          for(int i=1;i<kway;i++) send_disp[i] = std::lower_bound(&arr[0], &arr[arr.size()], split_key[i-1]) - &arr[0];
          for(int i=0;i<kway;i++) send_size[i] = send_disp[i+1] - send_disp[i];

          // Get recv_size.
          int recv_iter=0;
          std::vector<T*> recv_ptr(kway);
          std::vector<size_t> recv_cnt(kway);
          std::vector<int> recv_size(kway), recv_disp(kway+1,0);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            MPI_Status status;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;
              MPI_Sendrecv(&send_size[     i   ], 1, MPI_INT, partner, 0,
                           &recv_size[recv_iter], 1, MPI_INT, partner, 0, comm, &status);
              recv_disp[recv_iter+1] = recv_disp[recv_iter]+recv_size[recv_iter];
              recv_ptr[recv_iter]=&arr_[0] + recv_disp[recv_iter]; //! @hari - only setting address, doesnt need to be allocated yet (except for 0)
              recv_cnt[recv_iter]=recv_size[recv_iter];
              recv_iter++;
            }
          }

          // Communicate data.
          int asynch_count=2;
          recv_iter=0;
					int merg_indx=2;
          std::vector<MPI_Request> reqst(kway*2);
          std::vector<MPI_Status> status(kway*2);
          arr_ .resize(recv_disp[kway]);
          arr__.resize(recv_disp[kway]);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;

              if(recv_iter-asynch_count-1>=0) MPI_Waitall(2, &reqst[(recv_iter-asynch_count-1)*2], &status[(recv_iter-asynch_count-1)*2]);
              
							par::Mpi_Irecv <T>(&arr_[recv_disp[recv_iter]], recv_size[recv_iter], partner, 1, comm, &reqst[recv_iter*2+0]);
              par::Mpi_Issend<T>(&arr [send_disp[     i   ]], send_size[     i   ], partner, 1, comm, &reqst[recv_iter*2+1]);
              
							recv_iter++;

              int flag[2]={0,0};
              if ( recv_iter > merg_indx ) MPI_Test(&reqst[(merg_indx-1)*2], &flag[0], &status[(merg_indx-1)*2]);
              if ( recv_iter > merg_indx ) MPI_Test(&reqst[(merg_indx-2)*2], &flag[1], &status[(merg_indx-2)*2]);
              if (flag[0] && flag[1]){
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2; merg_indx%s==0; s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]], omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
            }
          }
#ifdef _PROFILE_SORT
				hyper_communicate.stop();
				hyper_merge.start();
#endif
					// Merge remaining parts.
          while(merg_indx<=(int)kway){
              MPI_Waitall(1, &reqst[(merg_indx-1)*2], &status[(merg_indx-1)*2]);
              MPI_Waitall(1, &reqst[(merg_indx-2)*2], &status[(merg_indx-2)*2]);
              {
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2;merg_indx%s==0;s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
          }
					{// Swap buffers.
						int swap_cond=0;
            for(int s=2;kway%s==0;s*=2) swap_cond++;
						if(swap_cond%2==0) swap(arr,arr_);
						else swap(arr,arr__);
					}
				}

#ifdef _PROFILE_SORT
				hyper_merge.stop();
				hyper_comm_split.start();
#endif
				{// Split comm. kway  O( log(p) ) ??
    	     MPI_Comm scomm;
      	   MPI_Comm_split(comm, blk_id, myrank, &scomm );
					 if(comm!=comm_) MPI_Comm_free(&comm);
        	 comm = scomm;

			     MPI_Comm_size(comm, &npes);
           MPI_Comm_rank(comm, &myrank);
    	  }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif
      }
#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
} // mem effecient HykSort


  /// ----------- low mem verison - sc13 -----------------------------------
  template<typename T>
    int sampleSort(std::vector<T>& arr, MPI_Comm comm){ 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

#ifdef _PROFILE_SORT
	 		total_sort.start();
#endif

     int npes;

      MPI_Comm_size(comm, &npes);

      assert(arr.size());

      if (npes == 1) {
#ifdef _PROFILE_SORT
				seq_sort.start();
#endif
        omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
  			seq_sort.stop();
		 		total_sort.stop();
#endif      
      } 

      int myrank;
      MPI_Comm_rank(comm, &myrank);

      DendroIntL nelem = arr.size();
      DendroIntL nelemCopy = nelem;
      DendroIntL totSize;
      par::Mpi_Allreduce<DendroIntL>(&nelemCopy, &totSize, 1, MPI_SUM, comm);

      DendroIntL npesLong = npes;
      const DendroIntL FIVE = 5;

      if(totSize < (FIVE*npesLong*npesLong)) {
        if(!myrank) {
          std::cout <<" Using bitonic sort since totSize < (5*(npes^2)). totSize: "
            <<totSize<<" npes: "<<npes <<std::endl;
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        MPI_Comm new_comm;
        if(totSize < npesLong) {
          if(!myrank) {
            std::cout<<" Input to sort is small. splittingComm: "
              <<npes<<" -> "<< totSize<<std::endl;
          }
          par::splitCommUsingSplittingRank(static_cast<int>(totSize), &new_comm, comm);
        } else {
          new_comm = comm;
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-2 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        if(!arr.empty()) {
          par::bitonicSort<T>(arr, new_comm);
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif
      }// end if

#ifdef __DEBUG_PAR__
      if(!myrank) {
        std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
      }
#endif

      //Re-part arr so that each proc. has atleast p elements.
#ifdef _PROFILE_SORT
  		sort_partitionw.start();
#endif
			par::partitionW<T>(arr, NULL, comm);
#ifdef _PROFILE_SORT
  		sort_partitionw.stop();
#endif
      nelem = arr.size();

#ifdef _PROFILE_SORT
			seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
			seq_sort.stop();
#endif
      unsigned long idx; 
      // std::vector<IndexHolder<T> > sendSplits(npes-1);
      std::vector< IndexHolder<T> >  splitters;
      std::vector< std::pair<T, DendroIntL> >  splitters_pair = par::Sorted_approx_Select_skewed( arr, npes-1, comm);
            
      for (int i=0; i<splitters_pair.size(); ++i) {
        splitters.push_back ( IndexHolder<T> ( splitters_pair[i].first, splitters_pair[i].second ) );
      }

      T key_last;
      DendroIntL zero = 0;
      MPI_Bcast (&key_last, 1, par::Mpi_datatype<T>::value(), npes-1, comm);

      splitters.push_back( IndexHolder<T>(key_last, zero) );
      
      omp_par::merge_sort(&splitters[0], &splitters[splitters.size()]);
      
      IndexHolder<T> *splittersPtr = NULL;
      if(!splitters.empty()) {
        splittersPtr = &(*(splitters.begin()));
      }

      int *sendcnts = new int[npes];
      assert(sendcnts);

      int * recvcnts = new int[npes];
      assert(recvcnts);

      int * sdispls = new int[npes];
      assert(sdispls);

      int * rdispls = new int[npes];
      assert(rdispls);

      #pragma omp parallel for
      for(int k = 0; k < npes; k++){
        sendcnts[k] = 0;
      }

      {
        int omp_p=omp_get_max_threads();
        int* proc_split = new int[omp_p+1];
        DendroIntL* lst_split_indx = new DendroIntL[omp_p+1];
        proc_split[0]=0;
        lst_split_indx[0]=0;
        lst_split_indx[omp_p]=nelem;
        #pragma omp parallel for
        for(int i=1;i<omp_p;i++){
          //proc_split[i] = seq::BinSearch(&splittersPtr[0],&splittersPtr[npes-1],arr[i*nelem/omp_p],std::less<T>());
          idx = 2*myrank*nelem/npes + i*(size_t)nelem/omp_p;
          IndexHolder<T> key( arr[i*(size_t)nelem/omp_p], idx);
          proc_split[i] = std::upper_bound( &splittersPtr[0], &splittersPtr[npes-1], key, std::less<IndexHolder<T> >())  - &splittersPtr[0];
          if(proc_split[i]<npes-1){
            //lst_split_indx[i]=seq::BinSearch(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>());
            lst_split_indx[i] = std::upper_bound(&arr[0], &arr[nelem], splittersPtr[proc_split[i]].value , std::less<T>()) - &arr[0];
          }else{
            proc_split[i]     = npes-1;
            lst_split_indx[i] = nelem;
          }
        }
        idx = 2*myrank*nelem/npes;
        #pragma omp parallel for
        for (int i=0;i<omp_p;i++){
          int sendcnts_=0;
          int k=proc_split[i];
          for (DendroIntL j = lst_split_indx[i]; j < lst_split_indx[i+1]; j++) {
            if ( IndexHolder<T>(arr[j],idx+j) <= splitters[k]) {
              sendcnts_++;
            } else{
              if(sendcnts_>0)
                sendcnts[k]=sendcnts_;
              sendcnts_=0;
              k = seq::UpperBound< IndexHolder<T> >(npes-1, splittersPtr, k+1, IndexHolder<T>(arr[j],idx+j) );
              if (k == (npes-1) ){
                //could not find any splitter >= arr[j]
                sendcnts_ = (nelem - j);
                break;
              } else {
                assert(k < (npes-1));
                assert(splitters[k].value >= arr[j]);
                sendcnts_++;
              }
            }//end if-else
          }//end for j
          if(sendcnts_>0)
            sendcnts[k]=sendcnts_;
        }
        delete [] lst_split_indx;
        delete [] proc_split;
      }

      par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);

      sdispls[0] = 0; rdispls[0] = 0;

      omp_par::scan(sendcnts,sdispls,npes);
      omp_par::scan(recvcnts,rdispls,npes);

      DendroIntL nsorted = rdispls[npes-1] + recvcnts[npes-1];
      std::vector<T> SortedElem(nsorted);

      T* arrPtr = NULL;
      T* SortedElemPtr = NULL;
      if(!arr.empty()) {
        arrPtr = &(*(arr.begin()));
      }
      if(!SortedElem.empty()) {
        SortedElemPtr = &(*(SortedElem.begin()));
      }
#ifdef _PROFILE_SORT
	 		sample_prepare_scatter.stop();
#endif
				
#ifdef _PROFILE_SORT
	 		sample_do_all2all.start();
#endif							
      // par::Mpi_Alltoallv_dense<T>(arrPtr, sendcnts, sdispls, SortedElemPtr, recvcnts, rdispls, comm);
      Mpi_Alltoallv(arrPtr, sendcnts, sdispls, SortedElemPtr, recvcnts, rdispls, comm);
#ifdef _PROFILE_SORT
	 		sample_do_all2all.stop();
#endif							
      arr.swap(SortedElem);
      SortedElem.clear();
      delete [] sendcnts;
      sendcnts = NULL;

      delete [] recvcnts;
      recvcnts = NULL;

      delete [] sdispls;
      sdispls = NULL;

      delete [] rdispls;
      rdispls = NULL;

#ifdef _PROFILE_SORT
	 		seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0], &arr[nsorted]);
#ifdef _PROFILE_SORT
	 		seq_sort.stop();
#endif


#ifdef _PROFILE_SORT
	 		total_sort.stop();
#endif
    }//end function

  //------------------------------------------------------------------------

  template<typename T>
    int sampleSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm){ 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

#ifdef _PROFILE_SORT
	 		total_sort.start();
#endif

     int npes;

      MPI_Comm_size(comm, &npes);

      assert(arr.size());

      if (npes == 1) {
#ifdef _PROFILE_SORT
				seq_sort.start();
#endif
        omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
  			seq_sort.stop();
#endif        
				SortedElem  = arr;
#ifdef _PROFILE_SORT
		 		total_sort.stop();
#endif      
			} 

      std::vector<T>  splitters;
      std::vector<T>  allsplitters;

      int myrank;
      MPI_Comm_rank(comm, &myrank);

      DendroIntL nelem = arr.size();
      DendroIntL nelemCopy = nelem;
      DendroIntL totSize;
      par::Mpi_Allreduce<DendroIntL>(&nelemCopy, &totSize, 1, MPI_SUM, comm);

      DendroIntL npesLong = npes;
      const DendroIntL FIVE = 5;

      if(totSize < (FIVE*npesLong*npesLong)) {
        if(!myrank) {
          std::cout <<" Using bitonic sort since totSize < (5*(npes^2)). totSize: "
            <<totSize<<" npes: "<<npes <<std::endl;
        }
//        par::partitionW<T>(arr, NULL, comm);

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        SortedElem = arr; 
        MPI_Comm new_comm;
        if(totSize < npesLong) {
          if(!myrank) {
            std::cout<<" Input to sort is small. splittingComm: "
              <<npes<<" -> "<< totSize<<std::endl;
          }
          par::splitCommUsingSplittingRank(static_cast<int>(totSize), &new_comm, comm);
        } else {
          new_comm = comm;
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-2 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        if(!SortedElem.empty()) {
          par::bitonicSort<T>(SortedElem, new_comm);
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

      }// end if

#ifdef __DEBUG_PAR__
      if(!myrank) {
        std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
      }
#endif

      //Re-part arr so that each proc. has atleast p elements.
#ifdef _PROFILE_SORT
  		sort_partitionw.start();
#endif
//			par::partitionW<T>(arr, NULL, comm);
#ifdef _PROFILE_SORT
  		sort_partitionw.stop();
#endif
      nelem = arr.size();

#ifdef _PROFILE_SORT
			seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
			seq_sort.stop();
#endif
				
      std::vector<T> sendSplits(npes-1);
      splitters.resize(npes);

      #pragma omp parallel for
      for(int i = 1; i < npes; i++)         {
        sendSplits[i-1] = arr[i*nelem/npes];        
      }//end for i

#ifdef _PROFILE_SORT
 		  sample_sort_splitters.start();
#endif
      // sort sendSplits using bitonic ...
      par::bitonicSort<T>(sendSplits,comm);
#ifdef _PROFILE_SORT
 		  sample_sort_splitters.stop();
#endif
				
				
#ifdef _PROFILE_SORT
	 		sample_prepare_scatter.start();
#endif				
      // All gather with last element of splitters.
      T* sendSplitsPtr = NULL;
      T* splittersPtr = NULL;
      if(sendSplits.size() > static_cast<unsigned int>(npes-2)) {
        sendSplitsPtr = &(*(sendSplits.begin() + (npes -2)));
      }
      if(!splitters.empty()) {
        splittersPtr = &(*(splitters.begin()));
      }
      par::Mpi_Allgather<T>(sendSplitsPtr, splittersPtr, 1, comm);

      sendSplits.clear();

      int *sendcnts = new int[npes];
      assert(sendcnts);

      int * recvcnts = new int[npes];
      assert(recvcnts);

      int * sdispls = new int[npes];
      assert(sdispls);

      int * rdispls = new int[npes];
      assert(rdispls);

      #pragma omp parallel for
      for(int k = 0; k < npes; k++){
        sendcnts[k] = 0;
      }

      {
        int omp_p=omp_get_max_threads();
        int* proc_split = new int[omp_p+1];
        DendroIntL* lst_split_indx = new DendroIntL[omp_p+1];
        proc_split[0]=0;
        lst_split_indx[0]=0;
        lst_split_indx[omp_p]=nelem;
        #pragma omp parallel for
        for(int i=1;i<omp_p;i++){
          //proc_split[i] = seq::BinSearch(&splittersPtr[0],&splittersPtr[npes-1],arr[i*nelem/omp_p],std::less<T>());
          proc_split[i] = std::upper_bound(&splittersPtr[0],&splittersPtr[npes-1],arr[i*(size_t)nelem/omp_p],std::less<T>())-&splittersPtr[0];
          if(proc_split[i]<npes-1){
            //lst_split_indx[i]=seq::BinSearch(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>());
            lst_split_indx[i]=std::upper_bound(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>())-&arr[0];
          }else{
            proc_split[i]=npes-1;
            lst_split_indx[i]=nelem;
          }
        }
        #pragma omp parallel for
        for (int i=0;i<omp_p;i++){
          int sendcnts_=0;
          int k=proc_split[i];
          for (DendroIntL j = lst_split_indx[i]; j < lst_split_indx[i+1]; j++) {
            if (arr[j] <= splitters[k]) {
              sendcnts_++;
            } else{
              if(sendcnts_>0)
                sendcnts[k]=sendcnts_;
              sendcnts_=0;
              k = seq::UpperBound<T>(npes-1, splittersPtr, k+1, arr[j]);
              if (k == (npes-1) ){
                //could not find any splitter >= arr[j]
                sendcnts_ = (nelem - j);
                break;
              } else {
                assert(k < (npes-1));
                assert(splitters[k] >= arr[j]);
                sendcnts_++;
              }
            }//end if-else
          }//end for j
          if(sendcnts_>0)
            sendcnts[k]=sendcnts_;
        }
        delete [] lst_split_indx;
        delete [] proc_split;
      }

      par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);

      sdispls[0] = 0; rdispls[0] = 0;

      omp_par::scan(sendcnts,sdispls,npes);
      omp_par::scan(recvcnts,rdispls,npes);

      DendroIntL nsorted = rdispls[npes-1] + recvcnts[npes-1];
      SortedElem.resize(nsorted);

      T* arrPtr = NULL;
      T* SortedElemPtr = NULL;
      if(!arr.empty()) {
        arrPtr = &(*(arr.begin()));
      }
      if(!SortedElem.empty()) {
        SortedElemPtr = &(*(SortedElem.begin()));
      }
#ifdef _PROFILE_SORT
	 		sample_prepare_scatter.stop();
#endif
				
#ifdef _PROFILE_SORT
	 		sample_do_all2all.start();
#endif							
      par::Mpi_Alltoallv_dense<T>(arrPtr, sendcnts, sdispls,
          SortedElemPtr, recvcnts, rdispls, comm);
#ifdef _PROFILE_SORT
	 		sample_do_all2all.stop();
#endif							
      arr.clear();

      delete [] sendcnts;
      sendcnts = NULL;

      delete [] recvcnts;
      recvcnts = NULL;

      delete [] sdispls;
      sdispls = NULL;

      delete [] rdispls;
      rdispls = NULL;

#ifdef _PROFILE_SORT
	 		seq_sort.start();
#endif
      omp_par::merge_sort(&SortedElem[0], &SortedElem[nsorted]);
#ifdef _PROFILE_SORT
	 		seq_sort.stop();
#endif


#ifdef _PROFILE_SORT
	 		total_sort.stop();
#endif
    }//end function

  /********************************************************************/
  /*
   * which_keys is one of KEEP_HIGH or KEEP_LOW
   * partner    is the processor with which to Merge and Split.
   *
   */
  template <typename T>
    void MergeSplit( std::vector<T> &local_list, int which_keys, int partner, MPI_Comm  comm) {

      MPI_Status status;
      int send_size = local_list.size();
      int recv_size = 0;

      // first communicate how many you will send and how many you will receive ...

      int       my_rank;
      MPI_Comm_rank(comm, &my_rank);

      par::Mpi_Sendrecv<int, int>( &send_size , 1, partner, 0,
          &recv_size, 1, partner, 0, comm, &status);

      // if (!my_rank || my_rank==2)
      // std::cout << my_rank << " <--> " << partner << "  -> " << send_size << " <- " << recv_size << std::endl;
      
      std::vector<T> temp_list( recv_size, local_list[0] );

      T* local_listPtr = NULL;
      T* temp_listPtr = NULL;
      if(!local_list.empty()) {
        local_listPtr = &(*(local_list.begin()));
      }
      if(!temp_list.empty()) {
        temp_listPtr = &(*(temp_list.begin()));
      }

      par::Mpi_Sendrecv<T, T>( local_listPtr, send_size, partner,
          1, temp_listPtr, recv_size, partner, 1, comm, &status);


      MergeLists<T>(local_list, temp_list, which_keys);

      temp_list.clear();
    } // Merge_split 

  template <typename T>
    void Par_bitonic_sort_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1) );
      
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = (my_rank ^ eor_bit);
      
        if (my_rank < partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }
    }  // Par_bitonic_sort_incr 


  template <typename T>
    void Par_bitonic_sort_decr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1));
      
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = my_rank ^ eor_bit;
        
        if (my_rank > partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }

    } // Par_bitonic_sort_decr 

  template <typename T>
    void Par_bitonic_merge_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int       partner;
      int       rank, npes;

      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &npes);

      unsigned int num_left  =  binOp::getPrevHighestPowerOfTwo(npes);
      unsigned int num_right = npes - num_left;

      // 1, Do merge between the k right procs and the highest k left procs.
      if ( (static_cast<unsigned int>(rank) < num_left) &&
          (static_cast<unsigned int>(rank) >= (num_left - num_right)) ) {
        partner = static_cast<unsigned int>(rank) + num_right;
        MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
      } else if (static_cast<unsigned int>(rank) >= num_left) {
        partner = static_cast<unsigned int>(rank) - num_right;
        MergeSplit<T> ( local_list,  KEEP_HIGH, partner, comm);
      }
    }

  template <typename T>
    void bitonicSort_binary(std::vector<T> & in, MPI_Comm comm) {
      int                   proc_set_size;
      unsigned int            and_bit;
      int               rank;
      int               npes;

      MPI_Comm_size(comm, &npes);

#ifdef __DEBUG_PAR__
      assert(npes > 1);
      assert(!(npes & (npes-1)));
      assert(!(in.empty()));
#endif

      MPI_Comm_rank(comm, &rank);

      for (proc_set_size = 2, and_bit = 2;
          proc_set_size <= npes;
          proc_set_size = proc_set_size*2, 
          and_bit = and_bit << 1) {

        if ((rank & and_bit) == 0) {
          Par_bitonic_sort_incr<T>( in, proc_set_size, comm);
        } else {
          Par_bitonic_sort_decr<T>( in, proc_set_size, comm);
        }
      }//end for
    }

  template <typename T>
    void bitonicSort(std::vector<T> & in, MPI_Comm comm) {
      int               rank;
      int               npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      assert(!(in.empty()));

      omp_par::merge_sort(&in[0],&in[in.size()]);
      MPI_Barrier(comm);
      
      if(npes > 1) {

        // check if npes is a power of two ...
        bool isPower = (!(npes & (npes - 1)));

        if ( isPower ) {
          bitonicSort_binary<T>(in, comm);
        } else {
          MPI_Comm new_comm;

          // Since npes is not a power of two, we shall split the problem in two ...
          //
          // 1. Create 2 comm groups ... one for the 2^d portion and one for the
          // remainder.
          unsigned int splitter = splitCommBinary(comm, &new_comm);

          if ( static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }

          // 3. Do a special merge of the two segments. (original comm).
          Par_bitonic_merge_incr( in,  binOp::getNextHighestPowerOfTwo(npes), comm );

          splitter = splitCommBinaryNoFlip(comm, &new_comm);

          // 4. Now a final sort on the segments.
          if (static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }
        }//end if isPower of 2
      }//end if single processor
    }//end function

  template <typename T>
    void MergeLists( std::vector<T> &listA, std::vector<T> &listB,
        int KEEP_WHAT) {

      T _low, _high;

      assert(!(listA.empty()));
      assert(!(listB.empty()));

      _low  = ( (listA[0] > listB[0]) ? listA[0] : listB[0]);
      _high = ( (listA[listA.size()-1] < listB[listB.size()-1]) ?
          listA[listA.size()-1] : listB[listB.size()-1]);

      // We will do a full merge first ...
      size_t list_size = listA.size() + listB.size();
      
      std::vector<T> scratch_list(list_size);

      unsigned int  index1 = 0;
      unsigned int  index2 = 0; 

      for (size_t i = 0; i < list_size; i++) {
        //The order of (A || B) is important here, 
        //so that index2 remains within bounds
        if ( (index1 < listA.size()) && ( (index2 >= listB.size()) || (listA[index1] <= listB[index2]) ) ) {
          scratch_list[i] = listA[index1];
          index1++;
        } else {
          scratch_list[i] = listB[index2];
          index2++;        
        }
      }

      //Scratch list is sorted at this point.

      listA.clear();
      listB.clear();
      if ( KEEP_WHAT == KEEP_LOW ) {
        int ii=0;
        while ( ( (scratch_list[ii] < _low) || (ii < (list_size/2)) ) && (scratch_list[ii] <= _high) ) {
          ii++;        
        }
  
        if(ii) {
          listA.insert(listA.end(), scratch_list.begin(),
              (scratch_list.begin() + ii));
        }
      } else {
        int ii = (list_size - 1);
        while ( ( (ii >= (list_size/2)) 
              && (scratch_list[ii] >= _low) )
            || (scratch_list[ii] > _high) ) {
          ii--;        
        }
        if(ii < (list_size - 1) ) {
          listA.insert(listA.begin(), (scratch_list.begin() + (ii + 1)),
              (scratch_list.begin() + list_size));
        }
      }
      scratch_list.clear();
    }//end function

	
	template<typename T>
		std::vector<T> Sorted_Sample_Select(std::vector<T>& arr, unsigned int kway, std::vector<unsigned int>& min_idx, std::vector<unsigned int>& max_idx, std::vector<DendroIntL>& splitter_ranks, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			
			//-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			//Determine splitters. O( log(N/p) + log(p) )        
      int splt_count = (1000*kway*nelem)/totSize; 
      if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
      if (splt_count>nelem) splt_count=nelem;
      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i] = arr[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splitters(glb_splt_count);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<T>::value(), comm);

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count, 0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);
        
			splitter_ranks.clear(); splitter_ranks.resize(kway);	
			min_idx.clear(); min_idx.resize(kway);
			max_idx.clear(); max_idx.resize(kway);	
	    std::vector<T> split_keys(kway);
			#pragma omp parallel for
      for (unsigned int qq=0; qq<kway; qq++) {
				DendroIntL* _disp = &glb_disp[0];
				DendroIntL* _mind = &glb_disp[0];
				DendroIntL* _maxd = &glb_disp[0];
				DendroIntL optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
        	if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
						_disp = &glb_disp[i];
					}
        	if( (glb_disp[i] > optSplitter) && ( labs(glb_disp[i] - optSplitter) < labs(*_maxd - optSplitter))  ) {
						_maxd = &glb_disp[i];
					}
        	if( (glb_disp[i] < optSplitter) && ( labs(optSplitter - glb_disp[i]) < labs(optSplitter - *_mind))  ) {
						_mind = &glb_disp[i];
					}
				}
        split_keys[qq] = glb_splitters[_disp - &glb_disp[0]];
				min_idx[qq] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[_mind - &glb_disp[0]]) - &arr[0];
				max_idx[qq] = std::upper_bound(&arr[0], &arr[nelem], glb_splitters[_maxd - &glb_disp[0]]) - &arr[0];
				splitter_ranks[qq] = optSplitter - *_mind;
			}
			
			return split_keys;
		}	
	
  template<typename T>
    void Sorted_approx_Select_helper(std::vector<T>& arr, std::vector<size_t>& exp_rank, std::vector<T>& splt_key, int beta, std::vector<size_t>& start, std::vector<size_t>& end, size_t& max_err, MPI_Comm comm) {
  
      int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      
      size_t nelem=arr.size();
      int kway=exp_rank.size();
      std::vector<size_t> locSize(kway), totSize(kway);
      for(int i=0;i<kway;i++) locSize[i]=end[i]-start[i];
      par::Mpi_Allreduce<size_t>(&locSize[0], &totSize[0], kway, MPI_SUM, comm);
  
      //-------------------------------------------
      std::vector<T> loc_splt;
      for(int i=0;i<kway;i++){
        int splt_count = (totSize[i]==0?1:(beta*(end[i]-start[i]))/totSize[i]);
        if (npes>beta) splt_count = (((float)rand()/(float)RAND_MAX)*totSize[i]<(beta*locSize[i])?1:0);
        for(int j=0;j<splt_count;j++) loc_splt.push_back(arr[start[i]+rand()%(locSize[i]+1)]);
        std::sort(&loc_splt[loc_splt.size()-splt_count],&loc_splt[loc_splt.size()]);
      }
  
			int splt_count=loc_splt.size();
      
      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splt(glb_splt_count);
      MPI_Allgatherv(&loc_splt[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splt[0], &glb_splt_cnts[0], &glb_splt_disp[0], par::Mpi_datatype<T>::value(), comm);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////
      std::sort(&glb_splt[0],&glb_splt[glb_splt_count]);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<size_t> loc_rank(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          loc_rank[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splt[i]) - &arr[0];
        }
      }
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////
      std::vector<size_t> glb_rank(glb_splt_count, 0);
      MPI_Allreduce(&loc_rank[0], &glb_rank[0], glb_splt_count, par::Mpi_datatype<size_t>::value(), MPI_SUM, comm);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      size_t new_max_err=0;
      std::vector<T> split_keys(kway);
      // #pragma omp parallel for
      for (int i=0; i<kway; i++) {
        int ub_indx=std::upper_bound(&glb_rank[0], &glb_rank[glb_splt_count], exp_rank[i])-&glb_rank[0];
        int lb_indx=ub_indx-1; if(lb_indx<0) lb_indx=0;
        size_t err=labs(glb_rank[lb_indx]-exp_rank[i]);

        if(err<max_err){
          if(glb_rank[lb_indx]>exp_rank[i]) start[i]=0;
          else start[i] = loc_rank[lb_indx];
          if(ub_indx==glb_splt_count) end[i]=nelem;
          else end[i] = loc_rank[ub_indx];
          splt_key[i]=glb_splt[lb_indx];
          if(new_max_err<err) new_max_err=err;
        }
      }
      max_err=new_max_err;
    }

  template<typename T>
    std::vector<T> Sorted_approx_Select_recursive(std::vector<T>& arr, unsigned int kway, MPI_Comm comm) {
      int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      
      //-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);

      double tol=1e-2/kway;
      int beta=pow(1.0/tol,1.0/3.0)*3.0;
      std::vector<T> splt_key(kway);
      std::vector<size_t> start(kway,0);
      std::vector<size_t> end(kway,nelem);
      std::vector<size_t> exp_rank(kway);
      for(int i=0;i<kway;i++) exp_rank[i]=((i+1)*totSize)/(kway+1);
      
      size_t max_error=totSize;
      while(max_error>totSize*tol){
        Sorted_approx_Select_helper(arr, exp_rank, splt_key, beta, start, end, max_error, comm);
      }

      return splt_key;
    }
    
	template<typename T>
		std::vector<T> Sorted_approx_Select(std::vector<T>& arr, unsigned int kway, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			
			//-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			//Determine splitters. O( log(N/p) + log(p) )        
      int splt_count = (1000*kway*nelem)/totSize; 
      if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
      if (splt_count>nelem) splt_count=nelem;
      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i] = arr[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splitters(glb_splt_count);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<T>::value(), comm);

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count, 0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);
        
	    std::vector<T> split_keys(kway);
			#pragma omp parallel for
      for (unsigned int qq=0; qq<kway; qq++) {
				DendroIntL* _disp = &glb_disp[0];
				DendroIntL optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
        	if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
						_disp = &glb_disp[i];
					}
				}
        split_keys[qq] = glb_splitters[_disp - &glb_disp[0]];
			}
			
			return split_keys;
		}	
	
	template<typename T>
		std::vector<std::pair<T, DendroIntL> > Sorted_approx_Select_skewed (std::vector<T>& arr, unsigned int kway, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			
			//-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			//Determine splitters. O( log(N/p) + log(p) )        
      int splt_count = (1000*kway*nelem)/totSize; 
      if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
      if (splt_count>nelem) splt_count=nelem;
     
      //! this changes to a pair ?
      // long should be sufficient for some time at least 
      // 1<<63 <- 9,223,372,036,854,775,808 (9 Quintillion )  
      std::vector<T>          splitters(splt_count); 
      std::vector<DendroIntL> dup_ranks(splt_count); 
      for(size_t i=0;i<splt_count;i++) {
        dup_ranks[i] = (2*i*totSize/npes) + rand()%nelem;
        splitters[i] =  arr[rand()%nelem];
      }

      // std::cout << rank << ": got splitters and indices " << std::endl;
      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      
      std::vector<T>          glb_splitters (glb_splt_count);
      std::vector<DendroIntL> glb_dup_ranks (glb_splt_count);
      MPI_Allgatherv(&    dup_ranks[0], splt_count, par::Mpi_datatype<DendroIntL>::value(), 
                     &glb_dup_ranks[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<DendroIntL>::value(), comm);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<T>::value(), comm);


      // std::cout << rank << ": ranking splitters " << std::endl;
      // rank splitters. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count, 0);
      DendroIntL dLow, dHigh;
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          // disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
          dLow = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
          dHigh = std::upper_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
          if ( (dHigh-dLow) > 1 ) {
            DendroIntL sRank = glb_dup_ranks[i]*npes/2/totSize;
            if (sRank < rank ) {
              disp[i] = dLow;
            } else if (sRank > rank) {
              disp[i] = dHigh;
            } else {
              disp[i] = glb_dup_ranks[i] - (2*rank*totSize/npes);
            }
          } else {
            disp[i] = dLow;
          }
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count, 0);
     
      // std::cout << rank << ": all reduce " << std::endl;
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);
        
	    std::vector< std::pair<T, DendroIntL> > split_keys(kway);
      std::pair<T, DendroIntL> key_pair;
      #pragma omp parallel for
      for (unsigned int qq=0; qq<kway; qq++) {
				DendroIntL* _disp = &glb_disp[0];
				DendroIntL optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
        	if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
						_disp = &glb_disp[i];
					}
				}
        
        key_pair.first  = glb_splitters[_disp - &glb_disp[0]];
        key_pair.second = glb_dup_ranks[_disp - &glb_disp[0]];
        split_keys[qq]  = key_pair; // 
			}
			
      // std::cout << rank << ": all done" << std::endl;
			return split_keys;
		}	

}//end namespace

