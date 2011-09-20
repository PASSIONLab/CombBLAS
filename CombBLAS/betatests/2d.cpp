#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include "mpi.h"

using namespace std;

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
	istringstream iss(s);
	return !(iss >> f >> t).fail();
}

void DoAG(MPI_Comm & World, int N)
{
	int size, rank;
	MPI_Comm_size(World, &size); 
	MPI_Comm_rank(World, &rank);

	int * recvcnt = new int[size];
	N = N + rank;	// add some noise
	recvcnt[rank] = N;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	int * dpls = new int[size]();
	partial_sum(recvcnt, recvcnt+size-1, dpls+1); 
	int totrecv = accumulate(recvcnt, recvcnt+size, 0);
	double * recvbuf = new double[totrecv];

	double * data = new double[N];
	for (int i = 0; i< N; ++i)
	{
		data[i] = (double) i;
	}
	random_shuffle(data, data + N);
	
	double t1 = MPI_Wtime();
	MPI_Allgatherv(data, N, MPI_DOUBLE, recvbuf, recvcnt, dpls, MPI_DOUBLE, World);
	double t2 = MPI_Wtime();

	double time = t2-t1;
	double maxtime;
	double avetime;
	int grank, nprocs;
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs); 
	MPI_Comm_rank( MPI_COMM_WORLD, &grank);
	MPI_Reduce( &time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce( &time, &avetime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avetime = avetime / static_cast<double>(nprocs);
	
	if(grank == 0)
	{
		cout << "Allgatherv grid size: " << size << endl;
		cout << "Total data received (per proc): " << totrecv << " doubles" << endl;
		cout << "Average time: " << avetime << " seconds" << endl;
		cout << "Max time: " << maxtime << " seconds" << endl;
		cout << "Average bandwidth: " << (static_cast<double>(totrecv)*sizeof(double))/(avetime) << " bytes/sec" << endl;
	}

	t1 = MPI_Wtime();
	for(int i=0; i< 10; ++i)
		MPI_Allgatherv(data, N, MPI_DOUBLE, recvbuf, recvcnt, dpls, MPI_DOUBLE, World);
	t2 = MPI_Wtime();

	time = t2-t1;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce( &time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce( &time, &avetime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avetime = avetime / static_cast<double>(nprocs);

	if(grank == 0)
        {
		cout << "*** Subsequent 10 runs with the same data ***" << endl;
                cout << "Average time: " << avetime/10 << " seconds" << endl;
		cout << "Max time: " << maxtime/10 << " seconds" << endl;
                cout << "Bandwidth (average): " << (static_cast<double>(totrecv)*sizeof(double)*10.0)/(avetime) << " bytes/sec" << endl;
        }

	delete [] recvcnt;
	delete [] data;
	delete [] recvbuf;
	delete [] dpls;
}


void DoA2A(MPI_Comm & World, int N)
{
	int size, rank;
	MPI_Comm_size(World, &size); 
	MPI_Comm_rank(World, &rank);

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

	double time = t2-t1;
	double maxtime;
	double avetime;
	int grank, nprocs;
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs); 
	MPI_Comm_rank( MPI_COMM_WORLD, &grank);
	MPI_Reduce( &time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce( &time, &avetime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avetime = avetime / static_cast<double>(nprocs);
	
	if(grank == 0)
	{
		cout << "Alltoallv grid size: " << size << endl;
		cout << "Total data received: " << totrecv << " doubles" << endl;
		cout << "Average time: " << avetime << " seconds" << endl;
		cout << "Max time: " << maxtime << " seconds" << endl;
		cout << "Average bandwidth: " << (static_cast<double>(totrecv)*sizeof(double))/(avetime) << " bytes/sec" << endl;
	}

	t1 = MPI_Wtime();
	for(int i=0; i< 10; ++i)
        	MPI_Alltoallv(data, sendcnt, sdispls, MPI_DOUBLE, recvbuf, recvcnt, rdispls, MPI_DOUBLE, World);
        t2 = MPI_Wtime();

	time = t2-t1;
	MPI_Reduce( &time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce( &time, &avetime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	avetime = avetime / static_cast<double>(nprocs);

	if(grank == 0)
        {
		cout << "*** Subsequent 10 runs with the same data ***" << endl;
                cout << "Average time: " << avetime/10 << " seconds" << endl;
		cout << "Max time: " << maxtime/10 << " seconds" << endl;
                cout << "Bandwidth (average): " << (static_cast<double>(totrecv)*sizeof(double)*10.0)/(avetime) << " bytes/sec" << endl;
        }

	delete [] sendcnt;
	delete [] recvcnt;
	delete [] data;
	delete [] recvbuf;
	delete [] sdispls;
	delete [] rdispls;
}

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		cout << "Please specify the data size in millions (example: 4 means four millions doubles)";
		return 0;
	}
	int SIZE;
	from_string(SIZE,string(argv[1]),std::dec);
	SIZE *= 1000000;
	MPI_Comm squarerowcomm, squarecolcomm;
	MPI_Comm tallrowcomm, tallcolcomm;
	MPI_Comm widerowcomm, widecolcomm;
	int rank, nprocs;
	MPI_Init( 0, 0 );
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs); 
    	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	
	// First do square grid
	int grcols = (int)std::sqrt((float)nprocs);
	int grrows = grcols;
	
    	int myproccol = rank % grcols;
	int myprocrow = rank / grrows;
    	MPI_Comm_split( MPI_COMM_WORLD, myprocrow, rank, &squarerowcomm );
    	MPI_Comm_split( MPI_COMM_WORLD, myproccol, rank, &squarecolcomm );
	DoA2A(squarerowcomm, SIZE);
	DoAG(squarecolcomm, SIZE);
		
	// Now do tall grid
	grcols = grcols * 2;
	grrows = grrows / 2; 
	
	return 0;
}
