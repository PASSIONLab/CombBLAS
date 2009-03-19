
#include "SpMat.h"

template <class IT, class NT, class DER>
void SpMat<IT, NT, DER>::MultiplyAddAssign(SpMat<IT, NT, DT> & A, 
			SpMat<IT, NT, DER> & B, bool isAT, bool isBT)
{
	IT A_m, A_n, B_m, B_n;
 
	if(isAT)
	{
		A_m = A.n;
		A_n = A.m;
	}
	else
	{
		A_m = A.m;
		A_n = A.n;
	}
	if(isBT)
	{
		B_m = B.n;
		B_n = B.m;
	}
	else
	{
		B_m = B.m;
		B_n = B.n;
	}
		
        if(m == A_m && n == B_n)                
        {
               	if(A_n == B_m)
               	{
			if(isAT && isBT)
			{
				static_cast<DER*>(this)->PlusEq_AtXBt(static_cast<DER>(A), static_cast<DER>(B));
			}
			else if(isAT && (!isBT))
			{
				static_cast<DER*>(this)->PlusEq_AtXBn(static_cast<DER>(A), static_cast<DER>(B));
			}
			else if((!isAT) && isBT)
			{
				static_cast<DER*>(this)->PlusEq_AnXBt(static_cast<DER>(A), static_cast<DER>(B));
			}
			else
			{
				static_cast<DER*>(this)->PlusEq_AnXBn(static_cast<DER>(A), static_cast<DER>(B));
			}				
		}
                else
                {
                       	cerr <<"Not multipliable: " << A_n << "!=" << B_m << endl;
                }
        }
        else
        {
		cerr<< "Not addable: "<< m << "!=" << A_m << " or " << n << "!=" << B_n << endl;
        }
};

