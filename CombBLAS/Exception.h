#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

#include <iostream>
#include <exception>
using namespace std;

class outofrangeexception: public exception
{
  virtual const char* what() const throw()
  {
    return "Index out of range exception";
  }
} oorex;

#endif
