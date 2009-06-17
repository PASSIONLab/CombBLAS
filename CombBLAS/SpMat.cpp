
#include "SpMat.h"

template <class IT, class NT, class DER>
SpMat<IT, NT, DER> operator() (const vector<IT> & ri, const vector<IT> & ci) const
{
	return ((static_cast<DER>(*this)) (ri, ci));
}


template <class IT, class NT, class DER>
template <typename SR>
void SpMat<IT, NT, DER>::MultiplyAddAssign(SpMat<IT, NT, DER> & A, 
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
				static_cast<DER*>(this)->PlusEq_AtXBt<SR>(static_cast<DER>(A), static_cast<DER>(B));
			}
			else if(isAT && (!isBT))
			{
				static_cast<DER*>(this)->PlusEq_AtXBn<SR>(static_cast<DER>(A), static_cast<DER>(B));
			}
			else if((!isAT) && isBT)
			{
				static_cast<DER*>(this)->PlusEq_AnXBt<SR>(static_cast<DER>(A), static_cast<DER>(B));
			}
			else
			{
				static_cast<DER*>(this)->PlusEq_AnXBn<SR>(static_cast<DER>(A), static_cast<DER>(B));
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


template<typename IU, typename NU1, typename NU2, typename DER, typename SR>
SpTuples<IU, promote_trait<NU1,NU2>::T_promote> MultiplyReturnTuples
					(const SpMat<IU, NU1, DER> & A, 
					 const SpMat<IU, NU2, DER> & B, 
					 bool isAT, bool isBT, SR sring)

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
		
        if(A_n == B_m)
	{
		if(isAT && isBT)
		{
			return Tuples_AtXBt<SR>(static_cast<DER>(A), static_cast<DER>(B));
		}
		else if(isAT && (!isBT))
		{
			return Tuples_AtXBn<SR>(static_cast<DER>(A), static_cast<DER>(B));
		}
		else if((!isAT) && isBT)
		{
			return Tuples_AnXBt<SR>(static_cast<DER>(A), static_cast<DER>(B));
		}
		else
		{
			return Tuples_AnXBn<SR>(static_cast<DER>(A), static_cast<DER>(B));
		}				
	}
	else
	{
		cerr <<"Not multipliable: " << A_n << "!=" << B_m << endl;
		return SpTuples<IU, promote_trait<NU1,NU2>::T_promote> (0, 0, 0);
	}
}
