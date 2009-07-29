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

#define BETA 16
#define ITER 1000000

using namespace std;
typedef int vpackedsi __attribute__ ((vector_size (BETA*sizeof(int))));

union ipackedvector 
{
  	vpackedsi v;
  	int f[BETA];
};

template <int D, typename T>
void saxpy(T a, T *b, T *c)
{
	for(int i=0; i<D; ++i)
	{
		c[i] +=  a* b[i];
	}	
}

int main()
{
	int a = 2;
	int b[BETA];
	int c[BETA];

	for (int i=0; i< BETA; ++i)
	{
		b[i] = i;
		c[i] = 0;
	}

	timeval tim;		
	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(int i=0; i<ITER; ++i)
	{
		saxpy<BETA>(a, b, c);
	}
	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for template unrolled loop\n", t2-t1);
	copy(c, c+BETA, ostream_iterator<int>( cout, " "));
	cout << endl;

	ipackedvector av, bv, cv;
	for (int i=0; i<BETA; ++i)
	{
		av.f[i] = 2;
		bv.f[i] = i;
		cv.f[i] = 0;
	}
	
	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(int i=0; i<ITER; ++i)
	{
		cv.v += av.v * bv.v;	
	}
	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for gcc vector extensions\n", t2-t1);
	copy(cv.f, cv.f+BETA, ostream_iterator<int>( cout, " "));
	cout << endl;


	vector<int> tvec;
	tvec.reserve(10);
	
	tvec[5] = 5;
	cout << tvec.size() << " " << tvec.capacity() << endl;
	return 0;
}
