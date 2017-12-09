#ifndef _MPI_OP_H
#define _MPI_OP_H

#include <iostream>
#include <typeinfo>
#include <map>
#include <functional>
#include <mpi.h>
#include <stdint.h>
#include "Operations.h"
#include "MPIType.h"    // type_info_compare definition

namespace combblas {

class MPIOpCache
{
private:
    typedef std::map<std::type_info const*, MPI_Op, type_info_compare> stored_map_type;
    stored_map_type map;
    
public:
    void clear()
    {
        int is_finalized=0;
        MPI_Finalized(&is_finalized);
        if (! is_finalized ) 	// do not free after call to MPI_FInalize
        {
            // ignore errors in the destructor
            for (stored_map_type::iterator it=map.begin(); it != map.end(); ++it)
            {
                MPI_Op_free(&(it->second));
            }
        }
    }
    ~MPIOpCache()
    {
        clear();
    }
    MPI_Op get(const std::type_info* t)
    {
        stored_map_type::iterator pos = map.find(t);
        if (pos != map.end())
            return pos->second;
        else
            return MPI_OP_NULL;
    }
    
    void set(const std::type_info* t, MPI_Op datatype)
    {
#ifdef NOTGNU
        if (map.find(t) != map.end()) map.erase(t);
        map.insert(std::make_pair(t, datatype));
#else
        map[t] = datatype;
#endif
    }
};

extern MPIOpCache mpioc;	// global variable


// MPIOp: A class that has a static op() function that takes no arguments and returns the corresponding MPI_Op
// if and only if the given Op has a mapping to a valid MPI_Op
// No concepts checking for the applicability of Op on the datatype T at the moment
// In the future, this can be implemented via metafunction forwarding using mpl::or_ and mpl::bool_

template <typename Op, typename T, typename Enable = void>
struct MPIOp
{
    static void funcmpi(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
    {
        Op myop;    // you need to create the object instance
        T * pinvec = static_cast<T*>(invec);
        T * pinoutvec = static_cast<T*>(inoutvec);
        for (int i = 0; i < *len; i++)
        {
            pinoutvec[i] = myop(pinvec[i], pinoutvec[i]);
        }
    }
    static MPI_Op op()
    {
        std::type_info const* t = &typeid(Op);
        MPI_Op foundop = mpioc.get(t);
        
        if (foundop == MPI_OP_NULL)
        {
            MPI_Op_create(funcmpi, false, &foundop);
          
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            if(myrank == 0)
                std::cout << "Creating a new MPI Op for " << t->name() << std::endl;
            
            mpioc.set(t, foundop);
        }
        return foundop;
    }
};

template<typename T> struct MPIOp< maximum<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_MAX; } };
template<typename T> struct MPIOp< minimum<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_MIN; } };
template<typename T> struct MPIOp< std::plus<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_SUM; } };
template<typename T> struct MPIOp< std::multiplies<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_PROD; } };
template<typename T> struct MPIOp< std::logical_and<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_LAND; } };
template<typename T> struct MPIOp< std::logical_or<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_LOR; } };
template<typename T> struct MPIOp< logical_xor<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_LXOR; } };
template<typename T> struct MPIOp< bitwise_and<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > { static MPI_Op op() { return MPI_BAND; } };
template<typename T> struct MPIOp< bitwise_or<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > {  static MPI_Op op() { return MPI_BOR; } };
template<typename T> struct MPIOp< bitwise_xor<T>,T,typename std::enable_if<std::is_pod<T>::value, void>::type > { static MPI_Op op() { return MPI_BXOR; } };

}

#endif
