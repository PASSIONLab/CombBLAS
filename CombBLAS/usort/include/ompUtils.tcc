#include <cstdlib>
#include <omp.h>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cstring>
// #include <seqUtils.h>



template <class T,class StrictWeakOrdering>
void omp_par::merge(T A_,T A_last,T B_,T B_last,T C_,int p,StrictWeakOrdering comp){
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  _DiffType N1=A_last-A_;
  _DiffType N2=B_last-B_;
  if(N1==0 && N2==0) return;
  if(N1==0 || N2==0){
    _ValType* A=(N1==0? &B_[0]: &A_[0]);
    _DiffType N=(N1==0?  N2  :  N1   );
    #pragma omp parallel for
    for(int i=0;i<p;i++){
      _DiffType indx1=( i   *N)/p;
      _DiffType indx2=((i+1)*N)/p;
      memcpy(&C_[indx1], &A[indx1], (indx2-indx1)*sizeof(_ValType));
    }
    return;
  }

  //Split both arrays ( A and B ) into n equal parts.
  //Find the position of each split in the final merged array.
  int n=10;
  _ValType* split=new _ValType[p*n*2];
  _DiffType* split_size=new _DiffType[p*n*2];
  #pragma omp parallel for
  for(int i=0;i<p;i++){
    for(int j=0;j<n;j++){
      int indx=i*n+j;
      _DiffType indx1=(indx*N1)/(p*n);
      split   [indx]=A_[indx1];
      split_size[indx]=indx1+(std::lower_bound(B_,B_last,split[indx],comp)-B_);

      indx1=(indx*N2)/(p*n);
      indx+=p*n;
      split   [indx]=B_[indx1];
      split_size[indx]=indx1+(std::lower_bound(A_,A_last,split[indx],comp)-A_);
    }
  }

  //Find the closest split position for each thread that will 
  //divide the final array equally between the threads.
  _DiffType* split_indx_A=new _DiffType[p+1];
  _DiffType* split_indx_B=new _DiffType[p+1];
  split_indx_A[0]=0;
  split_indx_B[0]=0;
  split_indx_A[p]=N1;
  split_indx_B[p]=N2;
  #pragma omp parallel for
  for(int i=1;i<p;i++){
    _DiffType req_size=(i*(N1+N2))/p;

    int j=std::lower_bound(&split_size[0],&split_size[p*n],req_size,std::less<_DiffType>())-&split_size[0];
    if(j>=p*n)
      j=p*n-1;
    _ValType  split1     =split     [j];
    _DiffType split_size1=split_size[j];

    j=(std::lower_bound(&split_size[p*n],&split_size[p*n*2],req_size,std::less<_DiffType>())-&split_size[p*n])+p*n;
    if(j>=2*p*n)
      j=2*p*n-1;
    if(abs(split_size[j]-req_size)<abs(split_size1-req_size)){
      split1     =split   [j];
      split_size1=split_size[j];
    }

    split_indx_A[i]=std::lower_bound(A_,A_last,split1,comp)-A_;
    split_indx_B[i]=std::lower_bound(B_,B_last,split1,comp)-B_;
  }
  delete[] split;
  delete[] split_size;

  //Merge for each thread independently.
  #pragma omp parallel for
  for(int i=0;i<p;i++){
    T C=C_+split_indx_A[i]+split_indx_B[i];
    //std::merge(A_+split_indx_A[i],A_+split_indx_A[i+1],B_+split_indx_B[i],B_+split_indx_B[i+1],C,comp);
    //sse<_ValType>::merge(A_+split_indx_A[i],A_+split_indx_A[i+1],B_+split_indx_B[i],B_+split_indx_B[i+1],C);
    std::merge(A_+split_indx_A[i],A_+split_indx_A[i+1],B_+split_indx_B[i],B_+split_indx_B[i+1],C);
  }
  delete[] split_indx_A;
  delete[] split_indx_B;
}

template <class T,class StrictWeakOrdering>
void omp_par::merge_sort(T A,T A_last,StrictWeakOrdering comp){
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  int p=omp_get_max_threads();
  _DiffType N=A_last-A; 
  if(N<2*p || p==1){
    std::sort(A,A_last,comp);
    return;
  }

  //Split the array A into p equal parts.
  _DiffType* split=new _DiffType[p+1];
  split[p]=N;
  #pragma omp parallel for
  for(int id=0;id<p;id++){
    split[id]=(id*N)/p;
  }

  //Sort each part independently.
  #pragma omp parallel for
  for(int id=0;id<p;id++){
    std::sort(A+split[id],A+split[id+1],comp);
  }

  //Merge two parts at a time.
  _ValType* B=new _ValType[N];
  _ValType* A_=&A[0];
  _ValType* B_=&B[0];
  for(int j=1;j<p;j=j*2){
    for(int i=0;i<p;i=i+2*j){
      if(i+j<p){
	omp_par::merge(A_+split[i],A_+split[i+j],A_+split[i+j],A_+split[(i+2*j<=p?i+2*j:p)],B_+split[i],p,comp);
      }else{
	#pragma omp parallel for
	for(int k=split[i];k<split[p];k++)
	  B_[k]=A_[k];
      }
    }
    _ValType* tmp_swap=A_;
    A_=B_;
    B_=tmp_swap;
  }

  //The final result should be in A.
  if(A_!=&A[0]){
    #pragma omp parallel for
    for(int i=0;i<N;i++)
      A[i]=A_[i];
  }

  //Free memory.
  delete[] split;
  delete[] B;
}

template <class T>
void omp_par::merge_sort(T A,T A_last){
  typedef typename std::iterator_traits<T>::value_type _ValType;
  if(sizeof(_ValType)<=8*sizeof(_ValType*))
    omp_par::merge_sort(A,A_last,std::less<_ValType>());
  else
    omp_par::merge_sort_ptrs(A,A_last);
}

template <class T>
struct DataPtr{
public:
  T* elem;

  inline bool  operator < ( DataPtr<T> const  &other) const {
    return ((*elem)<(*(other.elem)));
  }
};

template <class T>
void omp_par::merge_sort_ptrs(T A,T A_last){
  //std::cout<<"Using Pointer sort.\n";
  int p=omp_get_max_threads();
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;
  _DiffType N=A_last-A; 

  // Make copy and init pointer array to be sorted.
  DataPtr<_ValType>* B=new DataPtr<_ValType>[N];
  _ValType* C=new _ValType[N];
  #pragma omp parallel for
  for(int i=0;i<p;i++){
    _DiffType start=(N*i)/p;
    _DiffType end=(N*(i+1))/p;
    memcpy(&C[start], &A[start], (end-start)*sizeof(_ValType));
    for(_DiffType j=start;j<end;j++) B[j].elem=&C[0]+j;
  }

  // Sort pointers.
  omp_par::merge_sort(B,B+N,std::less<DataPtr<_ValType> >());

  // Copy data to its sorted position.
  #pragma omp parallel for
  for(int i=0;i<p;i++){
    _DiffType start=(N*i)/p;
    _DiffType end=(N*(i+1))/p;
    for(size_t j=start;j<end;j++) A[j]=*(B[j].elem);
  }

  delete[] B;
  delete[] C;
}

template <class T, class I>
T omp_par::reduce(T* A, I cnt){
  T sum=0;
  #pragma omp parallel for reduction(+:sum)
  for(I i = 0; i < cnt; i++)
    sum+=A[i];
  return sum;
}

template <class T, class I>
void omp_par::scan(T* A, T* B,I cnt){
  int p=omp_get_max_threads();
  if(cnt<100*p){
    for(I i=1;i<cnt;i++)
      B[i]=B[i-1]+A[i-1];
    return;
  }

  I step_size=cnt/p;

  #pragma omp parallel for
  for(int i=0; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    if(i!=0)B[start]=0;
    for(I j=start+1; j<end; j++)
      B[j]=B[j-1]+A[j-1];
  }

  T* sum=new T[p];
  sum[0]=0;
  for(int i=1;i<p;i++)
    sum[i]=sum[i-1]+B[i*step_size-1]+A[i*step_size-1];

  #pragma omp parallel for
  for(int i=1; i<p; i++){
    int start=i*step_size;
    int end=start+step_size;
    if(i==p-1) end=cnt;
    T sum_=sum[i];
    for(I j=start; j<end; j++)
      B[j]+=sum_;
  }
  delete[] sum;
}


