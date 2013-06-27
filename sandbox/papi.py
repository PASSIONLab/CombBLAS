from ctypes import *

# load the library
cdll.LoadLibrary("libpapi.so.4")
libpapi = CDLL("libpapi.so.4")

print libpapi['PAPI_VER_CURRENT']

def PAPI_event_name_to_code(EventName):
    code = c_int()
    error = libpapi.PAPI_event_name_to_code(c_char_p(str(EventName)), byref(code));
    print "error:",error
    return code.value


