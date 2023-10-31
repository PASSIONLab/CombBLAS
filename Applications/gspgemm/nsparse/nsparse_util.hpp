#ifndef _NSPARSE_UTIL_H_
#define _NSPARSE_UTIL_H_

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"

#define failed(...)														\
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess)&& (localError != hipErrorPeerAccessAlreadyEnabled)&&        \
                     (localError != hipErrorPeerAccessNotEnabled )) {                              \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }


#endif
