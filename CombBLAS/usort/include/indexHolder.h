
/**
  @file indexHolder.h
  @brief A small helper class used while sorting a pair of value and its index in an array/vector. 
  @author Hari Sundar


  This is used for making the sort stable, when we have a large number of duplicates.
  */

#ifndef __INDEX_HOLDER_H_
#define __INDEX_HOLDER_H_

#include <iostream>

/**
  @brief A small helper class used while sorting a pair of value and its index in an array/vector. 
  */
template <typename T> 
class IndexHolder {
  public:
    unsigned long		index;
    T			          value;

    /** @name Constructors and Destructor */
    //@{
    IndexHolder() {
      value = T();
      index = 0;
    };
    IndexHolder(T  v, unsigned long i) { value = v; index = i;};
    IndexHolder(const IndexHolder<T>& rhs) {
      value = rhs.value;
      index = rhs.index;
    };

    ~IndexHolder() {};
    
    //@}

    

    /**
      @return a < b
      */
    static bool lessThan (const IndexHolder<T>& a, const IndexHolder<T>& b) {
      return a < b;
    }

    /** @name Overload Operators */
    //@{
    IndexHolder<T>& operator= (const IndexHolder<T>& other) {
      if ( this != &other ) {
        value = other.value;
        index = other.index;
      }

      return *this;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const IndexHolder& dt) {
      os << "idx-hldr: " << dt.value << ":" << dt.index << std::endl;
      return os;
    }

    
    bool  operator < ( IndexHolder<T> const  &other) const {
      if ( value == other.value ) 
        return (index < other.index );
      else
        return ( value < other.value );
    }

    bool  operator > ( IndexHolder<T> const  &other) const {
      if ( value == other.value ) 
        return (index > other.index );
      else
        return ( value > other.value );
    }

    bool  operator <= ( IndexHolder<T> const  &other) const {
      if ( value == other.value ) 
        return (index <= other.index );
      else
        return ( value <=  other.value );
    }

    bool  operator >= ( IndexHolder<T> const  &other) const {
      if ( value == other.value ) 
        return (index >= other.index );
      else
        return ( value >= other.value );
    }

    bool  operator == ( IndexHolder<T> const  &other) const {
      return ( ( value == other.value ) && (index == other.index) );
    }

    bool  operator != ( IndexHolder<T> const  &other) const {
      return (  ( value != other.value) || (index != other.index) );
    }
    //@}

};


namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

  template <typename T>
    class Mpi_datatype<IndexHolder<T> > {
      public:
        static MPI_Datatype value()
        {
          static bool         first = true;
          static MPI_Datatype datatype;

          if (first) { 
            /*
            IndexHolder<T> test;

            first = false;
            int block[2];  
            MPI_Aint disp[2];
            MPI_Datatype type[2];

            block[0] = 1;
            block[0] = 1;
            type[0] = MPI_LONG;
            type[1] = Mpi_datatype<T>::value(); 
            disp[0] = 0; 
            disp[1] = sizeof(long);  // sizeof(unsigned long); 
*/
            MPI_Type_contiguous(sizeof(IndexHolder<T>), MPI_BYTE, &datatype);
            // MPI_Type_create_struct(2, block, disp, type, &datatype);
            MPI_Type_commit(&datatype);
          }

          return datatype;
        }
    };

}//end namespace par


#endif

