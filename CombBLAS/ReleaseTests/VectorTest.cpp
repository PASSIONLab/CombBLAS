#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../SpParVec.h"
#include "../FullyDistVec.h"

using namespace std;

template <class T>
struct IsOdd : public unary_function<T,bool> {
  bool operator() (T number) {return (number%2==1);}
};

int main(int argc, char* argv[])
{

	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	{
		SpParVec<int64_t, int64_t> SPV_A(1024);
		SPV_A.SetElement(2,2);
		SPV_A.SetElement(83,-83);
		SPV_A.SetElement(284,284);
		SPV_A.DebugPrint();
		
		SpParVec<int64_t, int64_t> SPV_B(1024);
		SPV_B.SetElement(2,4);
		SPV_B.SetElement(184,368);
		SPV_B.SetElement(83,-1);
		SPV_B.DebugPrint();

		FullyDistVec<int64_t, int64_t> FDV(-1);
		FDV.iota(64,0);
		FDV.DebugPrint();

		FullyDistSpVec<int64_t, int64_t> FDSV = FDV.Find(IsOdd<int64_t>());
		FDSV.DebugPrint();

		SpParVec<int64_t, int64_t> SPV_C(12);
		SPV_C.SetElement(2,2);	
		SPV_C.SetElement(4,4);	
		SPV_C.SetElement(5,-5);	
		SPV_C.SetElement(6,6);	

		SpParVec<int64_t, int64_t> SPV_D(12);
		SPV_D.SetElement(2,4);	
		SPV_D.SetElement(3,9);	
		SPV_D.SetElement(5,-25);	
		SPV_D.SetElement(7,-49);	
		
		SPV_C += SPV_D;
		SPV_D += SPV_D;
		SPV_C.DebugPrint();
		SPV_D.DebugPrint();

		SpParVec<int64_t, int64_t> SPV_E(3);
		SPV_E.SetElement(0,3);
		SPV_E.SetElement(1,7);
		SPV_E.SetElement(2,10);
		SPV_E.DebugPrint();

		SpParVec<int64_t, int64_t> SPV_F = SPV_C(SPV_E);
		SPV_F.DebugPrint();
		SpParVec<int64_t, int64_t> SPV_H = SPV_C;
		SpParVec<int64_t, int64_t> SPV_J = SPV_H(SPV_F);
		int64_t val = SPV_J[8];
		stringstream tss;
		string ss;
		if(val == SPV_J.NOT_FOUND)
		{	
			ss = "NOT_FOUND";
		}
		else
		{
			tss << val;
			ss = tss.str();
		}
		cout << ss << endl;	
		SPV_J.SetElement(8, 777);

		val = SPV_J[8];
		if(val == SPV_J.NOT_FOUND)
		{	
			ss = "NOT_FOUND";
		}
		else
		{
			tss << val;
			ss = tss.str();
		}
		cout << ss << endl;	
	}
	MPI::Finalize();
	return 0;
}

