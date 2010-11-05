#include "pyCombBLAS.h"

int64_t invert64(int64_t v)
{
	return ~v;
}

int64_t abs64(int64_t v)
{
	if (v < 0) return -v;
	return v;
}


