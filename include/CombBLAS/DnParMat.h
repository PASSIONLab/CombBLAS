#ifndef _DN_PAR_MAT_H_
#define _DN_PAR_MAT_H_

#include <memory>

#include "CombBLAS.h"
#include "CommGrid.h"

using std::shared_ptr;



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
DnParMat
{
	
public:

	// constructors
	DnParMat ();

	DnParMat (shared_ptr<CommGrid> cgr, IT g_nr, IT g_nc, NT v);



	IT
	getnrow () const
	{
		return m;
	}



	IT
	getgnrow () const
	{
		return gm;
	}

	

	IT
	getncol () const
	{
		return n;
	}



	IT
	getgncol () const
	{
		return gn;
	}


	
	IT
	getnels () const
	{
		return m * n;
	}



	void
	PrintToFile (std::string prefix)
	{
		std::ofstream output;
		cgr->OpenDebugFile(prefix, output);
		for (IT i = 0; i < arr.size() / n; ++i)
		{
			std::copy(arr.begin() + (i*n), arr.begin() + ((i+1)*n),
					  std::ostream_iterator<NT> (output, " "));
			output << "\n";
		}
		output << std::endl;
		output.close();
	}



	void
	get_local_length (int i, int j, IT &rsize, IT &csize, IT &nels) const;

	


private:

	shared_ptr<CommGrid>	cgr; 
	vector<NT>				arr;
	IT						m;
	IT						n;
	IT						gm;
	IT						gn;


	
	void
	set_local_lengths ();



	// SpMM
	template <typename SR,
			  typename IU,
			  typename NUM,
			  typename NUV,
			  typename UDER> 
	friend
	DnParMat<IU, typename promote_trait<NUM, NUV>::T_promote> 
	SpMM_sC (const SpParMat<IU, NUM, UDER> &A,
			 const DnParMat<IU, NUV> &X,
			 spmm_stats &stats);

	template <typename SR,
			  typename IU,
			  typename NUM,
			  typename NUV,
			  typename UDER> 
	friend
	DnParMat<IU, typename promote_trait<NUM, NUV>::T_promote> 
	SpMM_sA_2D (const SpParMat<IU, NUM, UDER> &A,
				const DnParMat<IU, NUV> &X,
				spmm_stats &stats);
};

}




#include "DnParMat.cpp"

#endif

