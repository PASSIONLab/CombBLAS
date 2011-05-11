/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/

#ifndef _SP_DCCOLS_H
#define _SP_DCCOLS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "Isect.h"
#include "Semirings.h"
#include "MemoryPool.h"
#include "LocArr.h"
#include "Friends.h"


template <class IT, class NT>
class SpDCCols: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	// Constructors :
	SpDCCols ();
	SpDCCols (IT size, IT nRow, IT nCol, IT nzc, MemoryPool * mpool = NULL);
	SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool = NULL);
	SpDCCols (const SpDCCols<IT,NT> & rhs);					// Actual copy constructor		
	~SpDCCols();

	template <typename NNT> operator SpDCCols<IT,NNT> () const;		//!< NNT: New numeric type
	template <typename NIT, typename NNT> operator SpDCCols<NIT,NNT> () const;		//!< NNT: New numeric type, NIT: New index type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> operator() (IT ri, IT ci) const;	
	SpDCCols<IT,NT> operator() (const vector<IT> & ri, const vector<IT> & ci) const;
	bool operator== (const SpDCCols<IT, NT> & rhs) const
	{
		if(rhs.nnz == 0 && nnz == 0)
			return true;
		if(nnz != rhs.nnz || m != rhs.m || n != rhs.n)
			return false;
		return ((*dcsc) == (*(rhs.dcsc)));
	}

	class SpColIter //! Iterate over (sparse) columns of the sparse matrix
	{
	public:
		class NzIter	//! Iterate over the nonzeros of the sparse column
		{	
		public:
			NzIter(IT * ir, NT * num) : rid(ir), val(num) {}
     		
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

      		SpColIter(IT * cp, IT * jc) : cptr(cp), cid(jc) {}
     		
      		bool operator==(const SpColIter& other)
      		{
         		return(cptr == other.cptr);	// compare pointers
      		}
      		bool operator!=(const SpColIter& other)
      		{
         		return(cptr != other.cptr);
      		}
           	SpColIter& operator++()		// prefix operator
      		{
         		++cptr;
			++cid;
         		return(*this);
      		}
      		SpColIter operator++(int)	// postfix operator
      		{
         		SpColIter tmp(*this);
         		++(*this);
         		return(tmp);
      		}
      		IT colid() const	//!< Return the "local" colid of the current column. 
      		{
         		return (*cid);
      		}
		IT colptr() const
		{
			return (*cptr);
		}
		IT colptrnext() const
		{
			return (*(cptr+1));
		}
		IT nnz() const
		{
			return (colptrnext() - colptr());
		}
  	private:
      		IT * cptr;
		IT * cid;
   	};
	
	SpColIter begcol()
	{
		if( nnz > 0 )
			return SpColIter(dcsc->cp, dcsc->jc); 
		else	
			return SpColIter(NULL, NULL);
	}	
	SpColIter endcol()
	{
		if( nnz > 0 )
			return SpColIter(dcsc->cp + dcsc->nzc, NULL); 
		else
			return SpColIter(NULL, NULL);
	}

	typename SpColIter::NzIter begnz(const SpColIter & ccol)	//!< Return the beginning iterator for the nonzeros of the current column
	{
		return typename SpColIter::NzIter( dcsc->ir + ccol.colptr(), dcsc->numx + ccol.colptr() );
	}

	typename SpColIter::NzIter endnz(const SpColIter & ccol)	//!< Return the ending iterator for the nonzeros of the current column
	{
		return typename SpColIter::NzIter( dcsc->ir + ccol.colptrnext(), NULL );
	}			

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		if(nnz > 0)
			dcsc->Apply(__unary_op);	
	}
	
	template <typename _UnaryOperation>
	void Prune(_UnaryOperation __unary_op);

	template <typename _BinaryOperation>
	void UpdateDense(NT ** array, _BinaryOperation __binary_op) const
	{
		if(nnz > 0 && dcsc != NULL)
			dcsc->UpdateDense(array, __binary_op);
	}

	void EWiseScale(NT ** scaler, IT m_scaler, IT n_scaler);
	void EWiseMult (const SpDCCols<IT,NT> & rhs, bool exclude);
	
	void Transpose();				//!< Mutator version, replaces the calling object 
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object

	void RowSplit(int numsplits)
	{
		BooleanRowSplit(*this, numsplits);	// only works with boolean arrays
	}

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	void CreateImpl(const vector<IT> & essentials);
	void CreateImpl(IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);

	Arr<IT,NT> GetArrays() const;
	vector<IT> GetEssentials() const;
	const static IT esscount;

	bool isZero() const { return (nnz == 0); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	int getnsplit() const { return splits; }
	
	ofstream& put(ofstream & outfile) const;
	ifstream& get(ifstream & infile);
	void PrintInfo() const;
	void PrintInfo(ofstream & out) const;

	template <typename SR> 
	int PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);
	
	template <typename SR>
	int PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);

private:
	int splits;	// ABAB: Future multithreaded extension
	void CopyDcsc(Dcsc<IT,NT> * source);
	SpDCCols<IT,NT> ColIndex(const vector<IT> & ci) const;	//!< col indexing without multiplication	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdOutProdMult(const SpDCCols<IT,NTR> & rhs) const;	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdColByCol(const SpDCCols<IT,NTR> & rhs) const;	
	
	SpDCCols (IT size, IT nRow, IT nCol, const vector<IT> & indices, bool isRow);	// Constructor for indexing
	SpDCCols (IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc);			// Constructor for multiplication

	// Anonymous union
	union {
		Dcsc<IT, NT> * dcsc;
		Dcsc<IT, NT> ** dcscarr;
	};

	IT m;
	IT n;
	IT nnz;
	
	//! store a pointer to the memory pool, to transfer it to other matrices returned by functions like Transpose
	MemoryPool * localpool;

	template <class IU, class NU>
	friend class SpDCCols;		// Let other template instantiations (of the same class) access private members
	
	template <class IU, class NU>
	friend class SpTuples;

	template <class IU, class NU>
	friend class SpDCCols<IU, NU>::SpColIter;
	
	template<typename IU>
	friend void BooleanRowSplit(SpDCCols<IU, bool> & A, int numsplits);

	template<typename IU, typename NU1, typename NU2>
	friend SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseMult (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, bool exclude);

	template<typename IU, typename NU1, typename NU2, typename _BinaryOperation>
	friend SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseApply (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
        friend void dcsc_gespmv (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y);

	template <typename SR, typename IU, typename NUM, typename NUV>	
	friend void dcsc_gespmv (const SpDCCols<IU, NUM> & A, const IU * indx, const NUV * numx, IU nnzx, 	//!< SpMV with sparse vector
		vector<IU> & indy, vector<typename promote_trait<NUM,NUV>::T_promote>  & numy);

	template <typename SR, typename IU, typename NUM, typename NUV>	
	friend void dcsc_gespmv (const SpDCCols<IU, NUM> & A, const IU * indx, const NUV * numx, IU nnzx, 
		IU * indy, typename promote_trait<NUM,NUV>::T_promote * numy, int * cnts, int * dspls, int p_c);

	template <typename SR, typename IU, typename NUM, typename NUV>	
	friend int dcsc_gespmv_threaded (const SpDCCols<IU, NUM> & A, const IU * indx, const NUV * numx, IU nnzx, 
		IU * & sendindbuf, typename promote_trait<NUM,NUV>::T_promote * & sendnumbuf, int * & sdispls, int p_c);
		
	template <typename _BinaryOperation, typename IU, typename NUM, typename NUV>
	friend void dcsc_colwise_apply (const SpDCCols<IU, NUM> & A, const IU * indx, const NUV * numx, IU nnzx, _BinaryOperation __binary_op);
};

// At this point, complete type of of SpDCCols is known, safe to declare these specialization (but macros won't work as they are preprocessed)

// type promotion for IT=int64_t
template <> struct promote_trait< SpDCCols<int64_t,int> , SpDCCols<int64_t,int> >       
    {                                           
        typedef SpDCCols<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,bool> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,float> , SpDCCols<int64_t,float> >       
    {                                           
        typedef SpDCCols<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,double> , SpDCCols<int64_t,double> >       
    {                                           
        typedef SpDCCols<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,int64_t> >       
    {                                           
        typedef SpDCCols<int64_t,int64_t> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,int64_t> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,int64_t> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,int> >       
    {                                           
        typedef SpDCCols<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,int> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,int> , SpDCCols<int64_t,float> >       
    {                                           
        typedef SpDCCols<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,float> , SpDCCols<int64_t,int> >       
    {                                           
        typedef SpDCCols<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,int> , SpDCCols<int64_t,double> >       
    {                                           
        typedef SpDCCols<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,double> , SpDCCols<int64_t,int> >       
    {                                           
        typedef SpDCCols<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,unsigned> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,short> >       
    {                                           
        typedef SpDCCols<int64_t,short> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,short> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,short> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,unsigned> >       
    {                                           
        typedef SpDCCols<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,double> >       
    {                                           
        typedef SpDCCols<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,float> >       
    {                                           
        typedef SpDCCols<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,double> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int64_t,float> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,float> T_promote;                    
    };

// type promotion for IT=int
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,unsigned> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,short> >       
    {                                           
        typedef SpDCCols<int,short> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,short> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,short> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,unsigned> >       
    {                                           
        typedef SpDCCols<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };

// Below are necessary constructs to be able to define a SpMat<NT,IT> where
// all we know is DER (say SpDCCols<int, double>) and NT,IT
// in other words, we infer the templated SpDCCols<> type
// This is not a type conversion from an existing object, 
// but a type inference for the newly created object

template <class SPMAT, class NIT, class NNT>
struct create_trait
{
	// none
};

template <class NIT, class NNT>  struct create_trait< SpDCCols<int, double> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<int, int> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<int, bool> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<unsigned, double> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<unsigned, int> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<unsigned, bool> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, double> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, int> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, bool> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };

#include "SpDCCols.cpp"
#endif

