
/**
  @file seqUtils.txx
  @brief Definitions of the templated functions in the seq module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
 */

#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>

namespace seq {

  template <typename T>
    bool BinarySearch(const T* arr, unsigned int nelem, const T & key, unsigned int *ret_idx) {
      if(!nelem) {*ret_idx = nelem; return false;}
      unsigned int left = 0;
      unsigned int right = (nelem -1);	
      while (left <= right) {
        unsigned int mid =
          (unsigned int)( left + (unsigned int)(floor((double)(right-left)/2.0)) );

        if (key > arr[mid]) {
          left  = mid+1;
        } else if (key < arr[mid]) {
          if(mid>0) { right = mid-1; }
          else { right = 0; break;}
        } else {
          *ret_idx = mid;
          return true;
        }//end if-else-if
      }//end while
      *ret_idx = nelem;	
      return false;
    }//end function

  template <typename T>
    int UpperBound (unsigned int p,const T * splitt,unsigned int itr, const T & elem)
    {
      if (itr >= p) {
        return p;
      }
      while (itr < p){
        if (elem <= splitt[itr]) {
          return itr;
        } else {
          itr = itr + 1;
        }
      }//end while
      return itr;
    }//end function

  template <typename T>
    bool maxLowerBound(const std::vector<T>& arr, const T & key, unsigned int &ret_idx,
        unsigned int* leftIdx, unsigned int* rightIdx ) {
      unsigned int nelem = static_cast<unsigned int>(arr.size());
      ret_idx = 0;
      if(!nelem) { return false;}
      if(arr[0] > key) {	return false;   }
      if(arr[nelem-1] < key) {
        ret_idx = (nelem-1);
        return true;
      }//end if	
      //binary search
      unsigned int left = 0;
      unsigned int right = (nelem -1);	
      unsigned int mid = 0;
      if(leftIdx) {
        left = (*leftIdx);
      }
      if(rightIdx) {
        right = (*rightIdx);
      }
      while (left <= right) {
        mid = (unsigned int)( left + (unsigned int)(floor((double)(right-left)/2.0)) );
        if (key > arr[mid]) {
          left  = mid + (1u);
        } else if (key < arr[mid]){
          if(mid>0) {
            right = mid-1;
          }else {
            right=0;
            break;
          }
        } else {
          ret_idx = mid;
          return true;
        }//end if-else-if
      }//end while

      //If binary search did not find an exact match, it would have
      //stopped one element after or one element before. 

      if( (arr[mid] > key) && (mid > 0) ){ mid--; }	
      if(arr[mid] <= key ) { ret_idx = mid; return true; }
      else { ret_idx = 0; return false;}
    }//end function

  template<typename T> void makeVectorUnique(std::vector<T>& vecT, bool isSorted) {
    if(vecT.size() < 2) { return;}
    if(!isSorted) { sort(vecT.begin(),vecT.end()); }
    std::vector<T> tmp(vecT.size());
    //Using the array [] is faster than the vector version
    T* tmpPtr = (&(*(tmp.begin())));
    T* vecTptr = (&(*(vecT.begin())));
    tmpPtr[0] = vecTptr[0];
    unsigned int tmpSize=1;
    unsigned int vecTsz = static_cast<unsigned int>(vecT.size());
    for(unsigned int i = 1; i < vecTsz; i++) {	
      if(tmpPtr[tmpSize-1] != vecTptr[i]) {
        tmpPtr[tmpSize] = vecTptr[i];
        tmpSize++;
      }
    }//end for
    vecT.clear();
    vecT.insert(vecT.begin(), tmp.begin(), (tmp.begin() + tmpSize));
    tmp.clear();
  }//end function

  template <typename T>
    void flashsort(T* a, int n, int m, int *ctr)
    {
      const int THRESHOLD1 = 75;
      const int CLASS_SIZE = 75;     /* minimum value for m */

      /* declare variables */
      int *l, nmin, nmax, i, j, k, nmove, nx, mx;

      T c1,c2,flash,hold;

      /* allocate space for the l vector */
      l=(int*)calloc(m,sizeof(int));

      /***** CLASS FORMATION ****/

      nmin=nmax=0;
      for (i=0 ; i<n ; i++)
      {
        if (a[i] < a[nmin]) nmin = i;
        else if (a[i] > a[nmax]) nmax = i;
      }
        if ( (a[nmax]==a[nmin]) && (ctr==0) )
        {
          printf("All the numbers are identical, the list is sorted\n");
          return;
        }

        c1=(m-1.0)/(a[nmax]-a[nmin]) ;
        c2=a[nmin];

        l[0]=-1; /* since the base of the "a" (data) array is 0 */
        for (k=1; k<m ; k++) l[k]=0;

        for (i=0; i<n ; i++)
        {
          k=floor(c1*(a[i]-c2) );
          l[k]+=1;
        }

        for (k=1; k<m ; k++) l[k]+=l[k-1];

        hold=a[nmax];
        a[nmax]=a[0];
        a[0]=hold; 
        /**** PERMUTATION *****/

        nmove=0;
        j=0;
        k=m-1;

        while(nmove<n)
        {
          while  (j  >  l[k] )
          {
            j++;
            k=floor(c1*(a[j]-c2) ) ;
          }

          flash=a[ j ] ;

          while ( j <= l[k] )
          {
            k=floor(c1*(flash-c2));
            hold=a[ l[k] ];
            a[ l[k] ] = flash;
            l[k]--;
            flash=hold;
            nmove++;
          }
        }

        /**** Choice of RECURSION or STRAIGHT INSERTION *****/

        for (k=0;k<(m-1);k++)
          if ( (nx = l[k+1]-l[k]) > THRESHOLD1 )  /* then use recursion */
          {
            flashsort(&a[l[k]+1],nx,CLASS_SIZE,ctr);
            (*ctr)++;
          }

          else  /* use insertion sort */
            for (i=l[k+1]-1; i > l[k] ; i--)
              if (a[i] > a[i+1])
              {
                hold=a[i];
                j=i;
                while  (hold  >  a[j+1] )  a[j++]=a[j+1] ;
                a[j]=hold;
              }
        free(l);   /* need to free the memory we grabbed for the l vector */
    }
}//end namespace

/*    //Same as std::upper_limit
    template <class T,class StrictWeakOrdering>
    typename std::iterator_traits<T>::difference_type seq::BinSearch(T A,T A_last,
	const typename std::iterator_traits<T>::value_type& val,StrictWeakOrdering comp){
      typedef typename std::iterator_traits<T>::difference_type _DiffType;

      _DiffType a=0;
      _DiffType b=A_last-A-1;
      _DiffType c;

      if(!comp(val,*(A+b))) return b+1;
      if(comp(val,*(A+a))) return a;
      while(b-a>1){
        c=(a+b)/2;
        if(comp(val,*(A+c)))
          b=c;
        else
          a=c;
      }
      return b;
    }

     //Same as std::merge
    template <class T,class StrictWeakOrdering>
    void seq::Merge(T A_,T A_last,T B_,T B_last,T C_,StrictWeakOrdering comp){

      typedef typename std::iterator_traits<T>::difference_type _DiffType;

      _DiffType N1=A_last-A_;
      _DiffType N2=B_last-B_;
      T A=A_;
      T B=B_;
      T C=C_;
      if(N1+N2==0)return;

      for(_DiffType i=0;i<N1+N2;i++){
        if(comp(*A,*B)){
          if(A==A_last){
            for(_DiffType j=i;j<N1+N2;j++){
              *C=*B;
              ++C;
              ++B;
            }
            break;
          }
          *C=*A;
          ++C;
          ++A;
        }else{
          if(B==B_last){
            for(_DiffType j=i;j<N1+N2;j++)
            {
              *C=*A;
              ++C;
              ++A;
            }
            break;
          }
          *C=*B;
          ++C;
          ++B;
        }
      }
    }
*/

 
