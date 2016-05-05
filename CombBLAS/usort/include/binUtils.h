
/**
  @file binUtils.h
  @brief A set of efficient functions that use binary operations to perform some small computations.
  @author Hari Sundar, hsundar@gmail.com 
  */

#ifndef __BIN_UTILS_H_
#define __BIN_UTILS_H_
#include <vector>

/**
  @namespace binOp
  @brief A set of functions for fast binary operations.
  @author Hari Sundar
  */
namespace binOp{

  /**
    @return true if n is a power of 2.
    */
  bool isPowerOfTwo(unsigned int n);

  /**
    @return the minimum number of digits required to represent num in binary
    */
  unsigned int binLength(unsigned int num) ;

  /**
    return log to base 2 of num
    */
  unsigned int fastLog2(unsigned int num) ;

  /**
    @brief Converts a decimal number to binary
    @param dec the decimal number
    @param binLen the number of digits required in the binary representation
    @param result the binary representation
    @return error flag
    */
  int toBin(unsigned int dec, unsigned int binLen,  std::vector<bool>& result);

  /**
    @param numBin binary representation of the number
    @param binLen length of numBin
    @return the decimal representation of the binary number
    */
  unsigned int binToDec(unsigned int* numBin, unsigned int binLen) ;

  /**
    @return compute the next highest power of 2 of 32-bit v
    */
  int getNextHighestPowerOfTwo(unsigned int n);

  /**
    @return  compute the prev highest power of 2 of 32-bit v
    */
  int getPrevHighestPowerOfTwo(unsigned int n);

  /**
   * psuedo random generator ... kind of ...
   */
  unsigned int reversibleHash(unsigned int x);


}//end namespace

#endif
