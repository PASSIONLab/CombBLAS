
#include "SpMat.h"
#include "Friends.h"

template <class IT, class NT, class DER>
SpMat<IT, NT, DER> SpMat<IT, NT, DER>::operator() (const vector<IT> & ri, const vector<IT> & ci) const
{
	return ((static_cast<DER>(*this)) (ri, ci));
}


template <class IT, class NT, class DER>
template <typename SR>
void SpMat<IT, NT, DER>::SpGEMM(SpMat<IT, NT, DER> & A, 
			SpMat<IT, NT, DER> & B, bool isAT, bool isBT)
{
	IT A_m, A_n, B_m, B_n;
 
	if(isAT)
	{
		A_m = A.getncol();
		A_n = A.getnrow();
	}
	else
	{
		A_m = A.getnrow();
		A_n = A.getncol();
	}
	if(isBT)
	{
		B_m = B.getncol();
		B_n = B.getnrow();
	}
	else
	{
		B_m = B.getnrow();
		B_n = B.getncol();
	}
		
        if(getnrow() == A_m && getncol() == B_n)                
        {
               	if(A_n == B_m)
               	{
			if(isAT && isBT)
			{
				static_cast< DER* >(this)->template PlusEq_AtXBt< SR >(static_cast< DER & >(A), static_cast< DER & >(B));
			}
			else if(isAT && (!isBT))
			{
				static_cast< DER* >(this)->template PlusEq_AtXBn< SR >(static_cast< DER & >(A), static_cast< DER & >(B));
			}
			else if((!isAT) && isBT)
			{
				static_cast< DER* >(this)->template PlusEq_AnXBt< SR >(static_cast< DER & >(A), static_cast< DER & >(B));
			}
			else
			{
				static_cast< DER* >(this)->template PlusEq_AnXBn< SR >(static_cast< DER & >(A), static_cast< DER & >(B));
			}				
		}
                else
                {
                       	cerr <<"Not multipliable: " << A_n << "!=" << B_m << endl;
                }
        }
        else
        {
		cerr<< "Not addable: "<< getnrow() << "!=" << A_m << " or " << getncol() << "!=" << B_n << endl;
        }
};


template<typename IU, typename NU1, typename NU2, typename DER, typename SR>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> MultiplyReturnTuples
					(const SpMat<IU, NU1, DER> & A, 
					 const SpMat<IU, NU2, DER> & B, 
					 bool isAT, bool isBT)

{
	IU A_m, A_n, B_m, B_n;
 
	if(isAT)
	{
		A_m = A.getncol();
		A_n = A.getnrow();
	}
	else
	{
		A_m = A.getnrow();
		A_n = A.getncol();
	}
	if(isBT)
	{
		B_m = B.getncol();
		B_n = B.getnrow();
	}
	else
	{
		B_m = B.getnrow();
		B_n = B.getncol();
	}
		
        if(A_n == B_m)
	{
		if(isAT && isBT)
		{
			return Tuples_AtXBt<SR>(static_cast< DER & >(A), static_cast< DER & >(B));
		}
		else if(isAT && (!isBT))
		{
			return Tuples_AtXBn<SR>(static_cast< DER & >(A), static_cast< DER & >(B));
		}
		else if((!isAT) && isBT)
		{
			return Tuples_AnXBt<SR>(static_cast< DER & >(A), static_cast< DER & >(B));
		}
		else
		{
			return Tuples_AnXBn<SR>(static_cast< DER & >(A), static_cast< DER & >(B));
		}				
	}
	else
	{
		cerr <<"Not multipliable: " << A_n << "!=" << B_m << endl;
		return SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> (0, 0, 0);
	}
}

template <class IT, class NT, class DER>
inline ofstream& SpMat<IT, NT, DER>::put(ofstream& outfile) const
{
	return static_cast<DER*>(this)->put(outfile);
}

template <class IT, class NT, class DER>
inline ifstream& SpMat<IT, NT, DER>::get(ifstream& infile)
{
	return static_cast<DER*>(this)->get(infile);
}


template < typename UIT, typename UNT, typename UDER >
ofstream& operator<<(ofstream& outfile, const SpMat< UIT,UNT,UDER > & s)
{
	return s.put(outfile);
}

template < typename UIT, typename UNT, typename UDER >
ifstream& operator>> (ifstream& infile, SpMat< UIT,UNT,UDER > & s)
{
	return s.get(infile);
}

