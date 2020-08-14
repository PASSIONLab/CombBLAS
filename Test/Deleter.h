#ifndef _DELETER_H_
#define _DELETER_H_

#include <iostream>

template<typename A>
void DeleteAll(A arr1)
{
        delete [] arr1;
}

template<typename A, typename B>
void DeleteAll(A arr1, B arr2)
{
        delete [] arr2;
        DeleteAll(arr1);
}

template<typename A, typename B, typename C>
void DeleteAll(A arr1, B arr2, C arr3)
{
        delete [] arr3;
        DeleteAll(arr1, arr2);
}

template<typename A, typename B, typename C, typename D>
void DeleteAll(A arr1, B arr2, C arr3, D arr4)
{
        delete [] arr4;
        DeleteAll(arr1, arr2, arr3);
}


template<typename A, typename B, typename C, typename D, typename E>
void DeleteAll(A arr1, B arr2, C arr3, D arr4, E arr5)
{
        delete [] arr5;
        DeleteAll(arr1, arr2, arr3, arr4);
}

#endif
