
/**
  @file binUtils.C
  @brief A set of functions for fast binary operations
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  */

#include <vector>
#include <cassert>
#include "binUtils.h"

namespace binOp {

  unsigned int fastLog2(unsigned int num) {
    if(num) {
      return (binLength(num) - 1);
    } else {
      assert(false);
      return -1;
    }
  }//end function

  unsigned int binLength(unsigned int num) {
    unsigned int len = 1;
    while(num > 1) {
      num = (num >> 1);
      len++;
    }
    return len;
  }//end function

  int toBin(unsigned int num, unsigned int binLen,  std::vector<bool>& numBin) {
    numBin = std::vector<bool>(binLen);
    for(unsigned int i = 0; i < binLen; i++) {
      numBin[i]=0;
    }//end for
    unsigned int pos = binLen -1;
    while(num > 0) {
      numBin[pos] = (num%2);
      num = num/2;  
      pos--;
    }  //end while  
    return 1;
  }//end function

  unsigned int binToDec(unsigned int* numBin, unsigned int binLen) {
    unsigned int res = 0;
    for(unsigned int i = 0; i< binLen; i++) {
      res = (2*res) + numBin[i];
    }
    return res;
  }//end function


  bool isPowerOfTwo(unsigned int n) {
    return (n && (!(n & (n - 1))));
  }

  // compute the next highest power of 2 of 32-bit v
  int getNextHighestPowerOfTwo(unsigned int n) {
    unsigned int v = n;
    assert(v > 0);
    v--;
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    v++;
    return v;
  }

  // compute the prev highest power of 2 of 32-bit v
  int getPrevHighestPowerOfTwo(unsigned int n) {
    unsigned int v = n;
    assert(v > 0);
    v--;
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    v++;
    return  (v >> 1);
  }

  unsigned int reversibleHash(unsigned int x) {
    x*=0xDEADBEEF;
    x=x^(x>>17);
    x*=0x01234567;
    x+=0x88776655;
    x=x^(x>>4);
    x=x^(x>>9);
    x*=0x91827363;
    x=x^(x>>7);
    x=x^(x>>11);
    x=x^(x>>20);
    x*=0x77773333;
    return x;
  }

}//end namespace

