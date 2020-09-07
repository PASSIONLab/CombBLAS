#ifndef NODE_H 
#define NODE_H

#include <iostream>
#include <math.h>
#include <set>
#include <limits>
#include "../../trunk/CombBLAS/CombBLAS.h"

using namespace std;

MPI_Op staticMPIop;

struct Node {
  int id;
  double dist;
  int parent;

  Node(): id(-1), dist(numeric_limits<double>::infinity()), parent(-1) { }

  Node(int _id): id(_id), dist(numeric_limits<double>::infinity()), parent(-1) { }

  Node(double _dist): id(-1), dist(_dist), parent(-1) { }

  Node(double _dist, int _parent): id(-1), dist(_dist), parent(_parent) { }

  Node(int _id, double _dist, int _parent): id(_id), dist(_dist), parent(_parent) { }

  Node(const Node & _node): id(_node.id), dist(_node.dist), parent(_node.parent) { }

  operator double() const {
    return dist; 
  }

  Node& operator= (const Node& _node) {
    id = _node.id; 
    dist = _node.dist; 
    parent = _node.parent; 
    return *this; 
  }

  Node& operator-= (const Node& _node) { 
    dist -= _node.dist; 
    parent-=_node.parent; 
    return *this; 
  }

  Node operator- (const Node& _node) { 
    Node ret = *this; 
    ret -= _node; 
    return ret; 
  }

  Node& operator= (const int & _id) { 
    id = _id; 
    return *this; 
  }
};

struct SPSRing {
  static MPI_Op mpi_op() { 
    return staticMPIop; 
  }

  static bool returnedSAID() {
    return false;
  }

  // select the shorter distance
  static Node add(const Node & arg1, const Node & arg2) {
    // TODO: add self loop check?
    if(arg1.dist <= arg2.dist)
      return arg1;
    else
      return arg2;
  }

  // add the length of the current edge to the parent's distance.
  static Node multiply(const double & arg1, const Node & arg2) {
    return Node(arg2.dist + arg1, arg2.id);
  }
};

template <typename c, typename t>
inline std::basic_ostream<c,t>& operator<<
(std::basic_ostream<c,t>& lhs, const Node& rhs) { 
  return lhs << "(node: id = " << rhs.id+1 << ", dist = " << rhs.dist << 
    ", parent = " << rhs.parent+1 << ")"; 
}

MPI::Datatype Node_MPI_datatype;
template<> MPI::Datatype MPIType< Node > ( void ) {
  return Node_MPI_datatype;
}

template <> struct promote_trait<Node, Node> {                                          
  typedef Node T_promote;                   
};

template <> struct promote_trait<double, Node> {                                       
  typedef Node T_promote;                   
};

template <> struct promote_trait<Node, double> {                                       
  typedef Node T_promote;                   
};

// define SRing ops...
void apply(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
  Node* in = (Node*)invec;
  Node* inout = (Node*)inoutvec;

  for (int i = 0; i < *len; i++) {
    inout[i] = SPSRing::add(in[i], inout[i]);
  }
}

#endif
