#ifndef _FULLY_DIST_H
#define _FULLY_DIST_H

#include <iostream>
#include <algorithm>

using namespace std;

template <class IT, class NT>
class FullyDist
{
public:
	FullyDist():glen(0)
	{
		commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
	}
	FullyDist(IT globallen): glen(globallen) 
	{
		commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
	}
	FullyDist( shared_ptr<CommGrid> grid):glen(0)
	{
		commGrid.reset(new CommGrid(*grid));		
	}
	FullyDist( shared_ptr<CommGrid> grid, IT globallen): glen(globallen)
	{
		commGrid.reset(new CommGrid(*grid));		
	}

	IT LengthUntil() const;
	IT RowLenUntil() const;
	IT MyLocLength() const;
	IT MyRowLength() const;
	IT TotalLength() const { return glen; }
	int Owner(IT gind, IT & lind) const;
	int OwnerWithinRow(IT n_thisrow, IT ind_withinrow, IT & lind) const;

protected:
	shared_ptr<CommGrid> commGrid;
	IT glen;		// global length (actual "length" including zeros)
};


//! Given global index gind,
//! Return the owner processor id, and
//! Assign the local index to lind
template <class IT, class NT>
int FullyDist<IT,NT>::Owner(IT gind, IT & lind) const
{
	int procrows = commGrid->GetGridRows();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	IT n_thisrow;		// length assigned to owner's processor row	
	int own_procrow;	// owner's processor row
	if(n_perprocrow != 0)
	{
		own_procrow = std::min(static_cast<int>(gind / n_perprocrow), procrows-1);	// owner's processor row
	}
	else	// all owned by the last processor row
	{
		own_procrow = procrows -1;
	}

	IT ind_withinrow = gind - (own_procrow * n_perprocrow);
	if(own_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	

	int proccols = commGrid->GetGridCols();
	IT n_perproc = n_thisrow / proccols;	// length on a typical processor

	int own_proccol;
	if(n_perproc != 0)
	{
		own_proccol = std::min(static_cast<int>(ind_withinrow / n_perproc), proccols-1);
	}
	else
	{
		own_proccol = proccols-1;
	}
	lind = ind_withinrow - (own_proccol * n_perproc);

	// GetRank(int rowrank, int colrank) { return rowrank * grcols + colrank;}
	return commGrid->GetRank(own_procrow, own_proccol);
}

/**
 * @param[in] ind_withinrow {index within processor row}
 * @param[in] n_thisrow {length within this row}
 * @param[out] lind {index local to owning processor}
 * Return the owner processor id (within processor row)
 **/
template <class IT, class NT>
int FullyDist<IT,NT>::OwnerWithinRow(IT n_thisrow, IT ind_withinrow, IT & lind) const
{
	int proccols = commGrid->GetGridCols();
	IT n_perproc = n_thisrow / proccols;	// length on a typical processor
	
	int own_proccol;
	if(n_perproc != 0)
	{
		own_proccol = std::min(static_cast<int>(ind_withinrow / n_perproc), proccols-1);
	}
	else
	{
		own_proccol = proccols-1;
	}
	lind = ind_withinrow - (own_proccol * n_perproc);

	return own_proccol;
}



// The full distribution is actually a two-level distribution that matches the matrix distribution
// In this scheme, each processor row (except the last) is responsible for t = floor(n/sqrt(p)) elements. 
// The last processor row gets the remaining (n-floor(n/sqrt(p))*(sqrt(p)-1)) elements
// Within the processor row, each processor (except the last) is responsible for loc = floor(t/sqrt(p)) elements. 
// Example: n=103 and p=16
// All processors P_ij for i=0,1,2 and j=0,1,2 get floor(floor(102/4)/4) = 6 elements
// All processors P_i3 for i=0,1,2 get 25-6*3 = 7 elements
// All processors P_3j for j=0,1,2 get (102-25*3)/4 = 6 elements
// Processor P_33 gets 27-6*3 = 9 elements  
template <class IT, class NT>
IT FullyDist<IT,NT>::LengthUntil() const
{
	int procrows = commGrid->GetGridRows();
	int my_procrow = commGrid->GetRankInProcCol();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	IT n_thisrow;	// length assigned to this processor row	
	if(my_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	

	int proccols = commGrid->GetGridCols();
	int my_proccol = commGrid->GetRankInProcRow();

	IT n_perproc = n_thisrow / proccols;	// length on a typical processor

	return ((n_perprocrow * my_procrow)+(n_perproc*my_proccol));
}

// Return the length until this processor, within this processor row only
template <class IT, class NT>
IT FullyDist<IT,NT>::RowLenUntil() const
{
	int procrows = commGrid->GetGridRows();
	int my_procrow = commGrid->GetRankInProcCol();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	IT n_thisrow;	// length assigned to this processor row	
	if(my_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	

	int proccols = commGrid->GetGridCols();
	int my_proccol = commGrid->GetRankInProcRow();

	IT n_perproc = n_thisrow / proccols;	// length on a typical processor

	return (n_perproc*my_proccol);
}


template <class IT, class NT>
IT FullyDist<IT,NT>::MyLocLength() const
{
	int procrows = commGrid->GetGridRows();
	int my_procrow = commGrid->GetRankInProcCol();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	IT n_thisrow;	// length assigned to this processor row	
	if(my_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	

	int proccols = commGrid->GetGridCols();
	int my_proccol = commGrid->GetRankInProcRow();
	IT n_perproc = n_thisrow / proccols;	// length on a typical processor
	if(my_proccol == proccols-1)
		return (n_thisrow - (n_perproc*(proccols-1)));
	else
		return n_perproc;	
}


template <class IT, class NT>
IT FullyDist<IT,NT>::MyRowLength() const
{
	int procrows = commGrid->GetGridRows();
	int my_procrow = commGrid->GetRankInProcCol();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	IT n_thisrow;	// length assigned to this processor row	
	if(my_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	

	return n_thisrow;
}

#endif
