#include "DnParMat.h"



namespace combblas
{

template <class IT,
		  class NT>
DnParMat<IT, NT>::DnParMat () :
	m(0), n(0), gm(0), gn(0)
{
	cgr.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
}



template <class IT,
		  class NT>
DnParMat<IT, NT>::DnParMat
(
    shared_ptr<CommGrid>	cgr,
	IT						g_nr,
	IT						g_nc,
	NT						v
)
	: cgr(cgr), gm(g_nr), gn(g_nc)
{
	set_local_lengths();
	arr.resize(m*n, v);
}



template <class IT,
		  class NT>
void
DnParMat<IT, NT>::set_local_lengths ( )
{
	int procrows	 = cgr->GetGridRows();
	int my_procrow	 = cgr->GetRankInProcCol();
	IT	n_perprocrow = gm / procrows;
	m				 = n_perprocrow;
	if (my_procrow == procrows - 1)
		m = gm - (n_perprocrow * (procrows - 1));

	int proccols	 = cgr->GetGridCols();
	int my_proccol	 = cgr->GetRankInProcRow();
	IT	n_perproccol = gn / proccols;
	n				 = n_perproccol;
	if (my_proccol == proccols - 1)
		n = gn - (n_perproccol * (proccols - 1));
}
	


template <class IT,
		  class NT>
void
DnParMat<IT, NT>::get_local_length (int i, int j,
									IT &rsize, IT &csize, IT &nels) const
{
	int procrows	 = cgr->GetGridRows();
	IT	n_perprocrow = gm / procrows;
	rsize			 = n_perprocrow;
	if (i == procrows - 1)
		rsize = gm - (n_perprocrow * (procrows - 1));

	int proccols	 = cgr->GetGridCols();
	IT	n_perproccol = gn / proccols;
	csize			 = n_perproccol;
	if (j == proccols - 1)
		csize = gn - (n_perproccol * (proccols - 1));

	nels = rsize * csize;
}


	
}
