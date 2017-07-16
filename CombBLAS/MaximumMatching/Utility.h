#include "../CombBLAS.h"

/*
 * Serial: Check the validity of the matching solution;
 we need a better solution using invert
 */
template <class IT, class NT>
bool isMatching(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col)
{
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    for(int i=0; i< mateRow2Col.glen ; i++)
    {
        int t = mateRow2Col[i];
        
        if(t!=-1 && mateCol2Row[t]!=i)
        {
            if(myrank == 0)
                cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    
    for(int i=0; i< mateCol2Row.glen ; i++)
    {
        int t = mateCol2Row[i];
        if(t!=-1 && mateRow2Col[t]!=i)
        {
            if(myrank == 0)
                cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    return true;
}



template <class IT>
void CheckMatching(FullyDistVec<IT,IT> & mateRow2Col, FullyDistVec<IT,IT> & mateCol2Row)
{
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int64_t nrow = mateRow2Col.TotalLength();
    int64_t ncol = mateCol2Row.TotalLength();
    FullyDistSpVec<IT,IT> mateRow2ColSparse (mateRow2Col, [](IT mate){return mate!=-1;});
    FullyDistSpVec<IT,IT> mateCol2RowSparse (mateCol2Row, [](IT mate){return mate!=-1;});
    FullyDistSpVec<IT,IT> mateRow2ColInverted = mateRow2ColSparse.Invert(ncol);
    FullyDistSpVec<IT,IT> mateCol2RowInverted = mateCol2RowSparse.Invert(nrow);

    
    
    bool isMatching = false;
    if((mateCol2RowSparse == mateRow2ColInverted) && (mateRow2ColSparse == mateCol2RowInverted))
        isMatching = true;
    
     bool isPerfectMatching = false;
    if((mateRow2ColSparse.getnnz()==nrow) && (mateCol2RowSparse.getnnz() == ncol))
        isPerfectMatching = true;
    
    
    if(myrank == 0)
    {
        cout << "-------------------------------" << endl;
        if(isMatching)
        {
            
            cout << "| This is a matching         |" << endl;
            if(isPerfectMatching)
            cout << "| This is a perfect matching |" << endl;
            
            
        }
        else
            cout << "| This is not a matching |" << endl;
        cout << "-------------------------------" << endl;
    }

}













