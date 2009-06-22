
#define NDEBUG

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "SpTuples.h"
#include "SpDCCols.h"

using namespace std;

int main()
{
	ifstream inputa("matrixA.txt");
	ifstream inputb("matrixB.txt");
	if(!(inputa.is_open() && inputb.is_open()))
	{
		cerr << "One of the input files doesn't exist\n";
		exit(-1);
	}
	
	SpDCCols<int,double> A;
	inputa >> A;
	A.PrintInfo();
	
	SpDCCols<int,double> B;
	inputb >> B;		
	B.PrintInfo();

	// arguments (in this order): nnz, n, m, nzc
	SpDCCols<int,double> C(0, A.getnrow(), B.getncol(), 0);
	C.PrintInfo();

	typedef PlusTimesSRing<double, double> PT;	
	C.SpGEMM <PT> (A, B, false, false);	// C = A*B
	C.PrintInfo();

	SpDCCols<int,bool> A_bool = A.ConvertNumericType<bool>();

	A_bool.PrintInfo();

	SpTuples<int,double> C_tuples =  MultiplyReturnTuples<PT>(A_bool, B, false, false);	// D = A_bool*B
	C_tuples.PrintInfo();

	SpTuples<int,double> C_tt = MultiplyReturnTuples<PT>(B, A_bool, false, true);
	C_tt.PrintInfo();


#define BIGTEST
	// Start big timing test
#ifdef BIGTEST

	ifstream input1("p1/input1_0");
        ifstream input2("p1/input2_0");
        if(!(input1.is_open() && input2.is_open()))
        {
                cerr << "One of the input files doesn't exist\n";
                exit(-1);
        }
	SpDCCols<int,double> bigA;
        input1 >> bigA;
        bigA.PrintInfo();

        SpDCCols<int,double> bigB;
        input2 >> bigB;
        bigB.PrintInfo();

	// Cache warm-up
	SpTuples<int,double> bigC = MultiplyReturnTuples<PT>(bigA, bigB, false, false);
        bigC.PrintInfo();
	
	struct timeval tempo1, tempo2;

	double elapsed_time;    /* elapsed time in seconds */
	long elapsed_seconds;  /* diff between seconds counter */
	long elapsed_useconds; /* diff between microseconds counter */

	gettimeofday(&tempo1, NULL);
	bigC = MultiplyReturnTuples<PT>(bigA, bigB, false, false);
	gettimeofday(&tempo2, NULL);
	elapsed_seconds  = tempo2.tv_sec  - tempo1.tv_sec;
	elapsed_useconds = tempo2.tv_usec - tempo1.tv_usec;

	elapsed_time = (elapsed_seconds + ((double) elapsed_useconds)/1000000.0);
	printf("ColByCol time = %.5f seconds\n", elapsed_time);

	// Cache warm-up
        SpTuples<int,double> bigC_t = MultiplyReturnTuples<PT>(bigA, bigB, false, true);
        bigC_t.PrintInfo();

	gettimeofday(&tempo1, NULL);
        bigC_t = MultiplyReturnTuples<PT>(bigA, bigB, false, true);
        gettimeofday(&tempo2, NULL);
        elapsed_seconds  = tempo2.tv_sec  - tempo1.tv_sec;
        elapsed_useconds = tempo2.tv_usec - tempo1.tv_usec;

        elapsed_time = (elapsed_seconds + ((double) elapsed_useconds)/1000000.0);
        printf("OuterProduct time = %.5f seconds\n", elapsed_time);

#endif
}
