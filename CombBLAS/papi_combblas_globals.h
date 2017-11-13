#ifndef PAPI_COMBBLAS_GLOBALS_
#define PAPI_COMBBLAS_GLOBALS_

#include <papi.h>
#include <string>
#include <vector>


char spmv_errorstring[PAPI_MAX_STR_LEN+1];
std::string bfs_papi_labels = {"SpMV", "fringe_updt", "parents_updt"};
int num_bfs_papi_labels = 3;


enum bfs_papi_enum { SpMV, fringe_updt, parents_updt };


std::string spmv_papi_labels = {"Fan-Out", "LocalSpMV", "Fan-In", "Merge"}; 

std::string combblas_event_names [] = {"PAPI_TOT_INS", "PAPI_L1_TCM", "PAPI_L2_TCM", "PAPI_L3_TCM"};
int combblas_papi_events [] = {PAPI_TOT_INS, PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM};
int combblas_papi_num_events = 4;

// outer index: SpMV iteration, middle index: papi_labels, inner index: papi events
// dimensions: <#iterations> <num_bfs_papi_labels> <combblas_papi_num_events+1>
std::vector< std::vector< std::vector<long long> > > bfs_counters;
std::vector< std::vector< std::vector<long long> > > spmv_counters;	



#endif
