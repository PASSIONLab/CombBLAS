/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/


#ifndef _SP_PAR_MAT_H_
#define _SP_PAR_MAT_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>
#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/memory>	// for shared_ptr
	#include <tr1/tuple>
#endif

#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "DenseParMat.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"

using namespace std;
using namespace std::tr1;


/**
  * This class implements an asynchronous 2D algorithm, in the sense that there is no notion of stages.
  * \n The process that completes its submatrix update, requests subsequent matrices from their owners w/out waiting to sychronize with other processors
  * \n This partially remedies the severe load balancing problem in sparse matrices. 
  * \n The class uses MPI-2 to achieve one-sided asynchronous communication
  * \n The algorithm treats each submatrix as a single block
  * \n Local data structure can be any SpMat that has a constructor with array sizes and getarrs() member 
  * \todo At this point, each MPI call probably sends less than 2 billion elements, but what if
  *	we need to send more with a single call? MPI routines expect an 32-bit int sendcount/recvcount.
  *	solution #1: Replace all MPI_{foo} with MPI_{foo}L whenever MPI 2.3 becomes widely available
  *	solution #2: Exchange data in a loop 
  */
template <class IT, class NT, class DER>
class SpParMat
{
public:	
	// Constructors
	SpParMat ();
	SpParMat (DER * myseq, shared_ptr<CommGrid> grid);
		
	SpParMat (ifstream & input, MPI::Intracomm & world);
	SpParMat (DER * myseq, MPI::Intracomm & world);	

	template <class DELIT>
	SpParMat (const DistEdgeList< DELIT > & rhs, bool removeloops = true);	// conversion from distributed edge list

	SpParMat (const SpParMat< IT,NT,DER > & rhs);				// copy constructor
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const FullyDistVec<IT,NT> & );	// matlab sparse
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const NT & );	// matlab sparse
	SpParMat< IT,NT,DER > & operator=(const SpParMat< IT,NT,DER > & rhs);	// assignment operator
	SpParMat< IT,NT,DER > & operator+=(const SpParMat< IT,NT,DER > & rhs);
	~SpParMat ();

	template <typename SR>
	void Square (); 

	float LoadImbalance() const;
	void Transpose();
	void EWiseMult (const SpParMat< IT,NT,DER >  & rhs, bool exclude);
	void EWiseScale (const DenseParMat<IT,NT> & rhs);
	void DimScale (const DenseParVec<IT,NT> & v, Dim dim);
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & , FullyDistVec<IT,NT> & ) const;
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & ) const;

	template <typename _BinaryOperation>
	void DimApply(Dim dim, const FullyDistVec<IT, NT>& v, _BinaryOperation __binary_op);

	template <typename _BinaryOperation, typename _UnaryOperation >	
	DenseParVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const;

	template <typename _BinaryOperation>	
	DenseParVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const;

	template <typename VT, typename _BinaryOperation, typename _UnaryOperation >	
	void Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

	template <typename VT, typename _BinaryOperation>	
	void Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const;

	template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

	template <typename VT, typename GIT, typename _BinaryOperation>	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const;

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		spSeq->Apply(__unary_op);	
	}

	IT RemoveLoops();	// returns the number of loops removed
	
	template <typename OT>
	void OptimizeForGraph500(OptBuf<IT,OT> & optbuf);

	void ActivateThreading(int numsplits);
	
	/*
	template <typename NU2, typename UDER2> 
	template <typename _BinaryOperation>
	void EWiseApply (const SpParMat<IT,NU2,UDER2> & B, _BinaryOperation __binary_op);
	*/

	template <typename _UnaryOperation>
	void Prune(_UnaryOperation __unary_op) //<! Prune any nonzero entries for which the __unary_op evaluates to true	
	{
		spSeq->Prune(__unary_op);
	}

	template <typename _BinaryOperation>
	void UpdateDense(DenseParMat<IT, NT> & rhs, _BinaryOperation __binary_op) const;

	void Dump(string filename) const;
	void PrintInfo() const;

	template <typename NNT, typename NDER> operator SpParMat< IT,NNT,NDER > () const;	//!< Type conversion operator
	template <typename NIT, typename NNT, typename NDER> operator SpParMat< NIT,NNT,NDER > () const;	//!< Type conversion operator (for indices as well)

	IT getnrow() const;
	IT getncol() const;
	IT getnnz() const;

	SpParMat<IT,NT,DER> SubsRefCol (const vector<IT> & ci) const;				//!< Column indexing with special parallel semantics

	//! General indexing with serial semantics

	SpParMat<IT,NT,DER> operator() (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace=false);
	SpParMat<IT,NT,DER> operator() (const SpParVec<IT,IT> & ri, const SpParVec<IT,IT> & ci) const;

	bool operator== (const SpParMat<IT,NT,DER> & rhs) const;

	ifstream& ReadDistribute (ifstream& infile, int master, bool nonum=false);
	ofstream& put(ofstream& outfile) const;
	void PrintForPatoh(string filename) const;

	shared_ptr<CommGrid> getcommgrid() const { return commGrid; } 	
	IT getlocalrows() const { return spSeq->getnrow(); }
	IT getlocalcols() const { return spSeq->getncol();} 
	IT getlocalnnz() const { return spSeq->getnnz(); }
	DER & seq() { return (*spSeq); }

	//! Friend declarations
	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_DoubleBuff (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_Synch (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_ActiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_PassiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_Fence (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend SpParVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const SpParVec<IU,NUV> & x );
	
	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue);
	
	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<IU, typename promote_trait<NUM,IU>::T_promote > & optbuf);
	
	template <typename _BinaryOperation, typename IU, typename NUM, typename NUV, typename UDER> 
	friend void ColWiseApply (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x, _BinaryOperation __binary_op);

	template <typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	EWiseMult (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B , bool exclude);

	template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote>
	EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal);

private:
	int Owner(IT total_m, IT total_n, IT grow, IT gcol, IT & lrow, IT & lcol) const;
	shared_ptr<CommGrid> commGrid; 
	DER * spSeq;
	
	template <class IU, class NU>
	friend class DenseParMat;

	template <typename IU, typename NU, typename UDER> 	
	friend ofstream& operator<< (ofstream& outfile, const SpParMat<IU,NU,UDER> & s);	
};

#include "SpParMat.cpp"
#endif
