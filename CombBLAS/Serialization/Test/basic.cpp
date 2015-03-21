#include "../Serialize.h"
#include <sstream>
#include <mpi.h>
#include <cassert>
#include <chrono>
#include "boost/serialization/vector.hpp"
#include "boost/serialization/set.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

#define NUM_SEND 1000000

using namespace std;

template<class T>
void test_send(const T& t) {
  stringbuf ss;
  chrono::time_point<chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();
  encode(ss, t);
  chrono::time_point<chrono::high_resolution_clock> end = chrono::high_resolution_clock::now();
  chrono::duration<double> dur = end - start;
  double rate = ss.str().size() / dur.count() / 1024 / 1024;
  cout << ss.str().size() << " bytes at " << rate << " MB/s encode" << endl;

  MPI_Send(ss.str().data(), ss.str().size(), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
}

template<class T>
void test_send_boost(const T& t) {
  stringbuf ss;
  boost::archive::binary_oarchive oa(ss);
  chrono::time_point<chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();
  oa << t;
  chrono::time_point<chrono::high_resolution_clock> end = chrono::high_resolution_clock::now();
  chrono::duration<double> dur = end - start;
  double rate = ss.str().size() / dur.count() / 1024 / 1024;
  cout << ss.str().size() << " bytes at " << rate << " MB/s encode (boost)" << endl;

  MPI_Send(ss.str().data(), ss.str().size(), MPI_BYTE, 1, 1, MPI_COMM_WORLD);
}

template<class T>
void test_recv(const T& t) {
  // find out how much data we are recieving
  MPI_Status status;
  int count;
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_BYTE, &count);

  // recieve data
  string s(count, 0);
  MPI_Recv((void *)s.data(), count, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

  // decode data
  stringbuf iss(s);
  T tt;

  chrono::time_point<chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();
  decode(iss, tt);
  chrono::time_point<chrono::high_resolution_clock> end = chrono::high_resolution_clock::now();
  chrono::duration<double> dur = end - start;
  auto rate = count / dur.count() / 1024 / 1024;
  cout << rate << " MB/s decode" << endl;

  if (t != tt) throw;
}

template<class T>
void test_recv_boost(const T& t) {
  // find out how much data we are recieving
  MPI_Status status;
  int count;
  MPI_Probe(0, 1, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_BYTE, &count);

  // recieve data
  string s(count, 0);
  MPI_Recv((void *)s.data(), count, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);

  // decode data
  stringbuf iss(s);
  boost::archive::binary_iarchive oa(iss);
  T tt;

  chrono::time_point<chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();
  oa >> tt;
  chrono::time_point<chrono::high_resolution_clock> end = chrono::high_resolution_clock::now();
  chrono::duration<double> dur = end - start;
  auto rate = count / dur.count() / 1024 / 1024;
  cout << rate << " MB/s decode (boost)" << endl;

  if (t != tt) throw;
}

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  vector<set<int>> myVector;
  for(int i = 0; i < NUM_SEND; i++) {
    myVector.emplace_back(set<int>{i,i,i,i,i,i});
  }

  if (myrank == 0) {
    cout << "Testing " << "std::vector<std::set<int>>" << " size: " << myVector.size() << endl;
    test_send(myVector);
    test_send_boost(myVector);
  } else if (myrank == 1) {
    test_recv(myVector);
    test_recv_boost(myVector);
    cout << "Test passed" << endl;
  }
  MPI_Finalize();
}
