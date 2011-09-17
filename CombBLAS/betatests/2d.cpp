#include <iostream>
#include <cmath>

using namespace std;


int main()
{
	MPI_Comm squarerowcomm, squarecolcomm;
	MPI_Comm tallrowcomm, tallcolcomm;
	MPI_Comm widerowcomm, widecolcomm;
	int rank, nprocs;
	MPI_Init( 0, 0 );
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs); 
    	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
	grcols = (int)std::sqrt((float)nprocs);
    	int myproccol = rank % grcols;
	int myprocrow = rank / grrows;
    	MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &squarerowcomm );
    	MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &squarecolcomm );
	return 0;
}
