from ctypes import *

# load the library
cdll.LoadLibrary("libpapi.so.4")
libpapi = CDLL("libpapi.so.4")

def get_papi_event_list(str_events):
	pass

