#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "mpi.h"

using namespace std;


void DoA2A(MPI_Comm & World, int N, int rank)
{
	int size;
	MPI_Comm_size(World, &size); 

	int * sendcnt = new int[size];
	int * recvcnt = new int[size];
	N = N + rank;	// add some noise
	for(int i=0; i<size-1; ++i)	
	{
		sendcnt[i] = N/size;	// sending N/size to everyone 
	}
	sendcnt[size-1] = N - (size-1)* (N/size); 
	double * data = new double[N];
	for (int i = 0; i< N; ++i)
	{
		data[i] = (double) i;
	}
	random_shuffle(data, data + N);

	
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);

	int * sdispls = new int[size];
	int * rdispls = new int[size];
	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<size-1; ++i)	
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	int totrecv = accumulate(recvcnt,recvcnt+size,0);
	double * recvbuf = new double[totrecv];

	double t1 = MPI_Wtime();
	MPI_Alltoallv(data, sendcnt, sdispls, MPI_DOUBLE, recvbuf, recvcnt, rdispls, MPI_DOUBLE, World);
	double t2 = MPI_Wtime();
	

	if(rank == 0)
	{
		cout << "Grid size: " << size << endl;
		cout << "Total data received: " << totrecv << " doubles" << endl;
		cout << "Time: " << t2-t1 << " seconds" << endl;
		cout << "Bandwidth: " << (static_cast<double>(totrecv)*sizeof(double))/(t2-t1) << " bytes/sec" << endl;
	}
	delete [] sendcnt;
	delete [] recvcnt;
	delete [] data;
	delete [] recvbuf;
	delete [] sdispls;
	delete [] rdispls;
}

int main()
{
	int SIZE = 1000000;	// data size: one million
	MPI_Comm squarerowcomm, squarecolcomm;
	MPI_Comm tallrowcomm, tallcolcomm;
	MPI_Comm widerowcomm, widecolcomm;
	int rank, nprocs;
	MPI_Init( 0, 0 );
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs); 
    	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
	// First do square grid
	int grcols = (int)std::sqrt((float)nprocs);
	int grrows = grcols;
	
    	int myproccol = rank % grcols;
	int myprocrow = rank / grrows;
    	MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &squarerowcomm );
    	MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &squarecolcomm );
	DoA2A(squarerowcomm, SIZE, rank);
		
	
	// Now do tall grid
	grcols = grcols * 2;
	grrows = grrows / 2; 
	
	
	return 0;
}
