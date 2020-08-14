#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <sys/time.h> // for gettimeofday
#include "promote.h"
#include "Semirings.h"
#include "Deleter.h"
#include <ext/numeric>
#include <tr1/array>
#include <xmmintrin.h>
#include <emmintrin.h>
//#include <smmintrin.h>	// SSE-4


#define BETA 16
#define ITER 1000000

template<typename T, typename I>
T ** allocate2D(I m, I n)
{
	T ** array = new T*[m];
	for(I i = 0; i<m; ++i) 
		array[i] = new T[n];
	return array;
}
template<typename T, typename I>
void deallocate2D(T ** array, I m)
{
	for(I i = 0; i<m; ++i) 
		delete [] array[i];
	delete [] array;
}


using namespace std;

//typedef int vpackedsi __attribute__ ((vector_size (BETA*sizeof(int))));			// 64-bytes, a full cache line !

//union ipackedvector 
//{
//  	vpackedsi v;
//  	int f[BETA];
//};

template <int D, typename T>
void saxpy(T a, T * __restrict b, T * __restrict c)
{
	for(int i=0; i<D; ++i)
	{
		c[i] +=  a* b[i];
	}	
}

template <int D, typename T>
void saxpy_array(T a, tr1::array<T,D> & b, tr1::array<T,D> & c)
{
	for(int i=0; i<D; ++i)
	{
		c[i] +=  a* b[i];
	}	
}


int main()
{
	/*
	__m128i a, b, c;
	int inp_sse1[4] __attribute__((aligned(16))) = { 2, 2, 2, 2 };
	int inp_sse2[4] __attribute__((aligned(16))) = { 0, 1, 2, 3 };
	int out_sse[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

 	__m128i res = _mm_mul_epi32(a, b); */


	int ** __restrict xx = allocate2D<int>(ITER,BETA);
	int ** __restrict yy = allocate2D<int>(ITER,BETA);
	
	for (int i=0; i< ITER; ++i)
	{
		for (int j=0; j< BETA; ++j)
		{
			xx[i][j] = j;
			yy[i][j] = 0;
		}
	}

	// tr1::array seems to be at least as fast as built-in C arrays
	int a = 2;
	tr1::array<int,BETA> * b = new tr1::array<int,BETA>[ITER];
	tr1::array<int,BETA> * c = new tr1::array<int,BETA>[ITER]();	// initialize to zero !


	timeval tim;
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	int index = 0;

	int * __restrict xxx = xx[index];
	int * __restrict yyy = yy[index];
	
	for(int i=0; i<ITER; ++i)
	{
		saxpy<BETA>(a, xxx, yyy);
	}

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for gcc vector extensions\n", t2-t1);

	copy(yy[0], yy[0]+BETA, ostream_iterator<float>( cout, " "));
	cout << endl;

	/*
	ipackedvector av, bv, cv;
	for (int i=0; i< ITER; ++i)
	{
		for (int j=0; j< BETA; ++j)
		{
			b[i][j] = j;
		}
	}
	//float * test = new float[10]();
	// note the empty set of parantheses as the initializer --> makes them default constructed
	// The C++ standard says that: A default constructed POD type is zero-initialized,
		
	//copy(test, test+10, ostream_iterator<float>( cout, " "));
	//cout << endl;
	

	//tr1::array<int,BETA> wtf = b[0] + c[0];
	
	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	for(int i=0; i<ITER; ++i)
	{
		//saxpy_array<BETA>(a, b[i], c[i]);
		c[i] = c[i] + b[i];
	}

	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for template unrolled loop\n", t2-t1);
	copy(c[0].begin(), c[0].end(), ostream_iterator<int>( cout, " "));
	cout << endl;*/
	

//	ipackedvector av, bv, cv;
//	for (int i=0; i<BETA; ++i)
//	{
//		av.f[i] = 2;
//		bv.f[i] = i;
//		cv.f[i] = 0;
//	}
	
//	gettimeofday(&tim, NULL);
//	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
//	for(int i=0; i<ITER; ++i)
//	{
//		cv.v += av.v * bv.v;	
//	}
//	gettimeofday(&tim, NULL);
//	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
//	printf("%.6lf seconds elapsed for gcc vector extensions\n", t2-t1);

//	copy(cv.f, cv.f+BETA, ostream_iterator<int>( cout, " "));
//	cout << endl;

	/*
	vector<int> tvec;
	tvec.reserve(10);
	
	tvec[5] = 5;
	cout << tvec.size() << " " << tvec.capacity() << endl;	*/
	return 0;
}
