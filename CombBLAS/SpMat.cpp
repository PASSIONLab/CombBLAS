/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <cstdlib>
#include "SpMat.h"
#include "Friends.h"
using namespace std;

template <class IT, class NT, class DER>
SpMat<IT, NT, DER> SpMat<IT, NT, DER>::operator() (const vector<IT> & ri, const vector<IT> & ci) const
{
	if( (!ci.empty()) && (ci.back() > getncol()))
	{
		cerr << "Col indices out of bounds" << endl;
		abort();
	}
	if( (!ri.empty()) && (ri.back() > getnrow()))
	{
		cerr << "Row indices out of bounds" << endl;
		abort();
	}

	return ((static_cast<DER>(*this)) (ri, ci));
}

template <class IT, class NT, class DER>
bool SpMat<IT, NT, DER>::operator== (const SpMat<IT, NT, DER> & rhs) const
{
	return ((static_cast<DER &>(*this)) == (static_cast<DER &>(rhs)) );
}

template <class IT, class NT, class DER>
void SpMat<IT, NT, DER>::Split( SpMat< IT,NT,DER > & partA, SpMat< IT,NT,DER > & partB) 
{
	static_cast< DER* >(this)->Split(static_cast< DER & >(partA), static_cast< DER & >(partB));
}

template <class IT, class NT, class DER>
void SpMat<IT, NT, DER>::Merge( SpMat< IT,NT,DER > & partA, SpMat< IT,NT,DER > & partB)
{
	static_cast< DER* >(this)->Merge(static_cast< DER & >(partA), static_cast< DER & >(partB));
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


template<typename SR, typename NUO, typename IU, typename NU1, typename NU2, typename DER1, typename DER2>
SpTuples<IU, NUO> * MultiplyReturnTuples
					(const SpMat<IU, NU1, DER1> & A, 
					 const SpMat<IU, NU2, DER2> & B, 
					 bool isAT, bool isBT,
					bool clearA = false, bool clearB = false)

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
			return Tuples_AtXBt<SR, NUO>(static_cast< const DER1 & >(A), static_cast< const DER2 & >(B), clearA, clearB);
		}
		else if(isAT && (!isBT))
		{
			return Tuples_AtXBn<SR, NUO>(static_cast< const DER1 & >(A), static_cast< const DER2 & >(B), clearA, clearB);
		}
		else if((!isAT) && isBT)
		{
			return Tuples_AnXBt<SR, NUO>(static_cast< const DER1 & >(A), static_cast< const DER2 & >(B), clearA, clearB);
		}
		else
		{
			return Tuples_AnXBn<SR, NUO>(static_cast< const DER1 & >(A), static_cast< const DER2 & >(B), clearA, clearB);
		}				
	}
	else
	{
		cerr <<"Not multipliable: " << A_n << "!=" << B_m << endl;
		return new SpTuples<IU, NUO> (0, 0, 0);
	}
}

template <class IT, class NT, class DER>
inline ofstream& SpMat<IT, NT, DER>::put(ofstream& outfile) const
{
	return static_cast<const DER*>(this)->put(outfile);
}

template <class IT, class NT, class DER>
inline ifstream& SpMat<IT, NT, DER>::get(ifstream& infile)
{
	cout << "Getting... SpMat" << endl;
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

