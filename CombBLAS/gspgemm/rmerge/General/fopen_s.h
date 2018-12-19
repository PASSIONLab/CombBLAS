#pragma once
#include <stdio.h>
#include <errno.h>

//This is a wrapper to create fopen_s for gcc
//fopen_s is an function provided by Visual Studio
#if defined(__GNUC__)
static int fopen_s(FILE **f, const char *name, const char *mode){
    int ret = 0;
    //assert(f);
    *f = fopen(name, mode);
    /* Can't be sure about 1-to-1 mapping of errno and MS' errno_t */
    if (!*f)
        ret = errno;
    return ret;
}
#endif
