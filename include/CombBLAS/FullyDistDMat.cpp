#include "FullyDistDMat.h"



namespace combblas
{

template <class IT,
		  class NT>
FullyDistDMat<IT, NT>::FullyDistDMat()
	: FullyDist<IT, NT, typename
				combblas::disable_if<combblas::is_boolean<NT>::value,
	                                 NT>::type>()
{
}



template <class IT,
		  class NT>
FullyDistDMat<IT, NT>::FullyDistDMat
(
    IT	glen,
	IT	d,
	NT	v
)
	: FullyDist<IT, NT, typename
				combblas::disable_if<combblas::is_boolean<NT>::value,
	                                 NT>::type>(glen),
	d_(d)
{
	arr_.resize(MyLocLength(), v);
}



template <class IT,
		  class NT>
FullyDistDMat<IT, NT>::FullyDistDMat
(
    shared_ptr<CommGrid> cgr
)
	: FullyDist<IT, NT, typename
				combblas::disable_if<combblas::is_boolean<NT>::value,
	                                 NT>::type>(cgr)
{
}



template <class IT,
		  class NT>
FullyDistDMat<IT, NT>::FullyDistDMat
(
    shared_ptr<CommGrid>	cgr,
    IT						glen,
	IT						d,
	NT						v
)
	: FullyDist<IT, NT, typename
				combblas::disable_if<combblas::is_boolean<NT>::value,
	                                 NT>::type>(cgr, glen),
	d_(d)
{
	arr_.resize(MyLocLength(), v);
}



template <class IT,
		  class NT>
FullyDistDMat<IT, NT> &
FullyDistDMat<IT, NT>::operator=
(
    const FullyDistDMat<IT, NT> &rhs
)
{
	if (this != &rhs)
	{
		FullyDist<IT,NT,typename combblas::disable_if<
							combblas::is_boolean<NT>::value, NT >::type>::
			operator= (rhs);
		arr_ = rhs.arr_;
		d_	 = rhs.d_;
	}

	return *this;
}



template <class IT,
		  class NT>
IT
FullyDistDMat<IT, NT>::MyLocLength ( ) const
{
	int procrows	 = commGrid->GetGridRows();
	int my_procrow	 = commGrid->GetRankInProcCol();
	IT	n_perprocrow = glen / procrows;
	IT	n_thisrow	 = n_perprocrow;
	if (my_procrow == procrows - 1)
		n_thisrow = glen - (n_perprocrow * (procrows - 1));

	int proccols   = commGrid->GetGridCols();
	int my_proccol = commGrid->GetRankInProcRow();
	IT	n_perproc  = n_thisrow / proccols;

	// extend by each dense dim
	if (my_proccol == proccols - 1)
		return (n_thisrow - (n_perproc * (proccols - 1))) * d_;
	else
		return n_perproc * d_;
}


}
