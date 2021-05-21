#ifndef _FULLY_DIST_DMAT_H_
#define _FULLY_DIST_DMAT_H_

#include <memory>
#include <vector>

#include "CombBLAS.h"
#include "CommGrid.h"
#include "FullyDist.h"

using std::shared_ptr;	using std::vector;



namespace combblas
{

// forward declarations
template <class IT,
		  class NT,
		  class DER>
class SpParMat;


template <class IT,
		  class NT>
class
FullyDistDMat : public FullyDist<IT, NT,
								 typename combblas::disable_if<
								 combblas::is_boolean<NT>::value, NT >::type>
{

public:


	// constructors
	FullyDistDMat ();

	FullyDistDMat (IT glen, IT d, NT v);

	FullyDistDMat (shared_ptr<CommGrid> cgr);
	
	FullyDistDMat (shared_ptr<CommGrid> cgr, IT glen, IT d, NT v);

	// FullyDistDMat (const vector<NT> &fillarr, shared_ptr<CommGrid> cgr);

	// operators
	FullyDistDMat<IT, NT> &
	operator=(const FullyDistDMat<IT, NT> &rhs);
	

	virtual
	IT
	MyLocLength () const;



	IT
	getncol () const
	{
		return d_;
	}



	IT
	LocArrSize() const
	{
		return arr_.size();
	}



	void
	PrintToFile (std::string prefix)
	{
		std::ofstream output;
		commGrid->OpenDebugFile(prefix, output);
		for (IT i = 0; i < arr_.size() / d_; ++i)
		{
			std::copy(arr_.begin() + (i*d_), arr_.begin() + ((i+1)*d_),
					  std::ostream_iterator<NT> (output, " "));
			output << "\n";
		}
		output << std::endl;
		output.close();
	}



	using FullyDist<IT, NT, typename combblas::disable_if<
								combblas::is_boolean<NT>::value,
								NT>::type>::glen; 
	using FullyDist<IT, NT, typename combblas::disable_if<
								combblas::is_boolean<NT>::value,
								NT>::type>::commGrid;


private:

	vector<NT>	arr_;
	IT			d_;



	// SpMM
	template <typename SR,
			  typename IU,
			  typename NUM,
			  typename NUV,
			  typename UDER> 
	friend
	FullyDistDMat<IU, typename promote_trait<NUM, NUV>::T_promote> 
	SpMM_sA (const SpParMat<IU, NUM, UDER> &A,
			 const FullyDistDMat<IU, NUV> &X,
			 spmm_stats &stats);
};
	
}


#include "FullyDistDMat.cpp"


#endif
