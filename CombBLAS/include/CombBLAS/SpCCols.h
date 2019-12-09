/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
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


#ifndef _SP_CCOLS_H_
#define _SP_CCOLS_H_

#include <cmath>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "csc.h"

namespace combblas {

template <class IT, class NT>
class SpCCols: public SpMat<IT, NT, SpCCols<IT, NT> >
{
public:
    typedef IT LocalIT;
    typedef NT LocalNT;
    
    // Constructors :
    SpCCols ();
    SpCCols (IT size, IT nRow, IT nCol);
    SpCCols (const SpTuples<IT,NT> & rhs, bool transpose);
    SpCCols (const SpDCCols<IT,NT> & rhs):nnz(0), n(0), m(0), splits(0), csc(NULL)
    {
        SpTuples<IT,NT> tuples(rhs);
        SpCCols<IT,NT> object(tuples, false);
        *this = object; // its members are already initialized by the initializer list
    }

    SpCCols (const SpCCols<IT,NT> & rhs);					// Actual copy constructor
    ~SpCCols();

    // Member Functions and Operators:
    SpCCols<IT,NT> & operator= (const SpCCols<IT, NT> & rhs);
    SpCCols<IT,NT> & operator+= (const SpCCols<IT, NT> & rhs);

    void RowSplit(int numsplits);
    void ColSplit(int parts, std::vector< SpCCols<IT,NT> > & matrices); //!< \attention Destroys calling object (*this)
    
    void CreateImpl(const std::vector<IT> & essentials);
    void CreateImpl(IT size, IT nRow, IT nCol, std::tuple<IT, IT, NT> * mytuples);
    
    Arr<IT,NT> GetArrays() const;
    std::vector<IT> GetEssentials() const;
    const static IT esscount;
    
    IT getnrow() const { return m; }
    IT getncol() const { return n; }
    IT getnnz() const { return nnz; }
    int getnsplit() const { return splits; }
    
    
    auto GetInternal() const    { return GetCSC(); }
    auto GetInternal(int i) const  { return GetCSC(i); }
    
    
    class SpColIter //! Iterate over (sparse) columns of the sparse matrix
    {
    public:
        class NzIter	//! Iterate over the nonzeros of the sparse column
        {
        public:
            NzIter(IT * ir = NULL, NT * num = NULL) : rid(ir), val(num) {}
            
            bool operator==(const NzIter & other)
            {
                return(rid == other.rid);	// compare pointers
            }
            bool operator!=(const NzIter & other)
            {
                return(rid != other.rid);
            }
            NzIter & operator++()	// prefix operator
            {
                ++rid;
                ++val;
                return(*this);
            }
            NzIter operator++(int)	// postfix operator
            {
                NzIter tmp(*this);
                ++(*this);
                return(tmp);
            }
            IT rowid() const	//!< Return the "local" rowid of the current nonzero entry.
            {
                return (*rid);
            }
            NT & value()		//!< value is returned by reference for possible updates
            {
                return (*val);
            }
        private:
            IT * rid;
            NT * val;
            
        };
        
        SpColIter(IT * jc = NULL) : begcptr(jc), curcptr(jc) {}
        bool operator==(const SpColIter& other)
        {
            return(curcptr == other.curcptr);	// compare pointers
        }
        bool operator!=(const SpColIter& other)
        {
            return(curcptr != other.curcptr);
        }
        SpColIter& operator++()		// prefix operator (different across derived classes)
        {
            ++curcptr;
            return(*this);
        }
        IT colid() const	//!< Return the "local" colid of the current column.
        {
            return (curcptr-begcptr);
        }
        IT colptr() const   // only needed internally by ::begnz() below
        {
            return (*curcptr);
        }
        IT colptrnext() const   // only needed internally by ::begnz() below
        {
            return (*(curcptr+1));
        }
        IT nnz() const
        {
            return (colptrnext() - colptr());
        }
    private:
        IT * begcptr;
        IT * curcptr;
   	};
    
    SpColIter begcol()  // serial version
    {
        if( nnz > 0 )
            return SpColIter(csc->jc);
        else
            return SpColIter(NULL);
    }
    SpColIter endcol()  //serial version
    {
        if( nnz > 0 )
            return SpColIter(csc->jc + n);  // (csc->jc+n) should never execute because SpColIter::colptrnext() would point invalid
        else
            return SpColIter(NULL);
    }

    SpColIter begcol(int i)  // multithreaded version
    {
        if( cscarr[i] )
            return SpColIter(cscarr[i]->jc);
        else
            return SpColIter(NULL);
    }
    SpColIter endcol(int i)  //multithreaded version
    {
        if( cscarr[i] )
            return SpColIter(cscarr[i]->jc + n);  // (csc->jc+n) should never execute because SpColIter::colptrnext() would point invalid
        else
            return SpColIter(NULL);
    }


    typename SpColIter::NzIter begnz(const SpColIter & ccol)	//!< Return the beginning iterator for the nonzeros of the current column
    {
        return typename SpColIter::NzIter( csc->ir + ccol.colptr(), csc->num + ccol.colptr() );
    }
    
    typename SpColIter::NzIter endnz(const SpColIter & ccol)	//!< Return the ending iterator for the nonzeros of the current column
    {
        return typename SpColIter::NzIter( csc->ir + ccol.colptrnext(), NULL );
    }
    
    typename SpColIter::NzIter begnz(const SpColIter & ccol, int i)	//!< multithreaded version
    {
        return typename SpColIter::NzIter( cscarr[i]->ir + ccol.colptr(), cscarr[i]->num + ccol.colptr() );
    }
    
    typename SpColIter::NzIter endnz(const SpColIter & ccol, int i)	//!< multithreaded version
    {
        return typename SpColIter::NzIter( cscarr[i]->ir + ccol.colptrnext(), NULL );
    }
    
    void PrintInfo() const;


	template <typename UnaryOperation,
			  typename GlobalIT>
	SpCCols<IT, NT> *
	PruneI (UnaryOperation unary_op, bool inPlace,
			GlobalIT rowOffset, GlobalIT colOffset);

	
    Csc<IT, NT> *
	GetCSC() const 	// only for single threaded matrices
    {
        return csc;
    }

	
    Csc<IT, NT> *
	GetCSC(int i) const 	// only for split (multithreaded) matrices
    {
        return cscarr[i];
    }
    

	bool
	isZero() const
	{
		return (nnz == 0);
	}


	// Transpose members
	void Transpose();
	SpCCols<IT, NT> TransposeConst() const;
	SpCCols<IT, NT> * TransposeConstPtr() const;

	void
	Split (SpCCols<IT, NT> &partA, SpCCols<IT, NT> &partB);

	void
	Merge (SpCCols<IT, NT> &partA, SpCCols<IT, NT> &partB);

	std::ofstream &
	put (std::ofstream &outfile) const;


	
private:

	SpCCols (IT nRow, IT nCol, Csc<IT, NT> *mycsc);
    
    void SubPrintInfo(Csc<IT,NT> * mycsc) const;

    // Anonymous union
    union {
        Csc<IT, NT> * csc;
        Csc<IT, NT> ** cscarr;
    };
    
    IT m;
    IT n;
    IT nnz;
    
    int splits;	// for multithreading

	template <class IU, class NU>
	friend class SpTuples;

    void CopyCsc(Csc<IT,NT> * source);
    
    template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
    friend void csc_gespmv_dense (const SpCCols<IU, NU> & A, const RHS * x, LHS * y); //!< dense vector (not implemented)
    
    //<! sparse vector version
    template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
    friend int generic_gespmv_threaded (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
                                        int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c, PreAllocatedSPA<OVT> & SPA);
};


// At this point, complete type of of SpCCols is known, safe to declare these specialization (but macros won't work as they are preprocessed)
// General case #1: When both NT is the same
template <class IT, class NT> struct promote_trait< SpCCols<IT,NT> , SpCCols<IT,NT> >
{
    typedef SpCCols<IT,NT> T_promote;
};
// General case #2: First is boolean the second is anything except boolean (to prevent ambiguity)
template <class IT, class NT> struct promote_trait< SpCCols<IT,bool> , SpCCols<IT,NT>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >
{
    typedef SpCCols<IT,NT> T_promote;
};
// General case #3: Second is boolean the first is anything except boolean (to prevent ambiguity)
template <class IT, class NT> struct promote_trait< SpCCols<IT,NT> , SpCCols<IT,bool>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >
{
    typedef SpCCols<IT,NT> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,int> , SpCCols<IT,float> >
{
    typedef SpCCols<IT,float> T_promote;
};

template <class IT> struct promote_trait< SpCCols<IT,float> , SpCCols<IT,int> >
{
    typedef SpCCols<IT,float> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,int> , SpCCols<IT,double> >
{
    typedef SpCCols<IT,double> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,double> , SpCCols<IT,int> >
{
    typedef SpCCols<IT,double> T_promote;
};


// Capture everything of the form SpCCols<OIT, ONT>
// it may come as a surprise that the partial specializations can
// involve more template parameters than the primary template
template <class NIT, class NNT, class OIT, class ONT>
struct create_trait< SpCCols<OIT, ONT> , NIT, NNT >
{
    typedef SpCCols<NIT,NNT> T_inferred;
};

}

#include "SpCCols.cpp"

#endif
