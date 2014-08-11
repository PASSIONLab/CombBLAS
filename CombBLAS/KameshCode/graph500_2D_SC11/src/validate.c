#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "graph.h"

/*
static int cmp_fun(const void *a, const void *b) {
      return (*(const int64_t *)a - *(const int64_t *)b);
}
*/

int validate_bfs_result(dist_graph_t *g, uint64_t root, uint64_t* pred, uint64_t
        pred_array_size) {


    uint64_t i, j;
    uint64_t bitmask63;
    uint32_t *adj;
    const uint8_t odd_mask = 0x0F;
    const uint8_t even_mask = 0xF0;
    int rank_rep = (rank % (g->nproc_rows * g->nproc_cols));
 
    adj = g->adj;
#if __x86_64__
    bitmask63 = (1UL<<63)-1;
#else
    bitmask63 = (1ULL<<63)-1;
#endif

#if !STORE_PRED
    return 1;
#endif

    /* Only run validation for small problem sizes */
    if (g->n > (1<<25))
        return 1;

    /* Sort the predecessor array by start vertex */
    // qsort(pred, pred_array_size, 16, cmp_fun); 

    /* Check if there's an edge corresponding to [u, pred(u)] */
    /* For local vertices, also check BFS level */ 
    for (i = 0; i < 2*pred_array_size; i+=2) {
        uint64_t u = pred[i];
        uint64_t pred_u = pred[i+1];
        //uint64_t u_local = u % g->n_local_row;
        if ((u == 0) && (pred_u == 0)) {
            continue;
        }

        if (pred_u == bitmask63) {
            if ((root % g->n_local_col) == u) {  
                continue;
            } else {
                fprintf(stderr, "root vertex not found %lu %lu\n", u, root);
                return 0;
            }
        }
        
        uint64_t u_global = u + (g->comm_data).jcol * g->n_local_col;

#if REPLICATE_D
        uint32_t u_local_val = u;
        uint32_t du = g->d[u_local_val/2];
        if ((u_local_val & 1U) == 0) { 
            du = (du & even_mask)>>4;
        } else {
            du = (du & odd_mask);
        }
        if (du == 0) {
            fprintf(stderr, "rank %d, u %lu, ucolproc %lu\n", 
                    rank, u_global, u/g->n_local);
            assert(du > 0);
        } 
#else
        uint32_t u_local_val = u_global % g->n_local;

        uint32_t du = g->d[u_local_val/2];
        if ((u_local_val & 1U) == 0) { 
            du = (du & even_mask)>>4;
        } else {
            du = (du & odd_mask);
        }
        assert(du > 0); 
#endif
                  
        int found_u = 0;
        /* Check the adjacency list of pred_u for u */
        //uint64_t pred_u_local = pred_u % g->n_local_row;
        //assert((pred_u/g->n_local_row) == (g->comm_data).irow);

        if ((pred_u/g->n_local_row) == (g->comm_data).irow) {
       
            uint64_t u_local_row = (u + (g->comm_data).jcol *  g->n_local_col) %
                g->n_local_row;
            assert(u_local_row < g->n_local_row);
            uint64_t pred_u_local = pred_u % g->n_local_row;

            for (j=g->num_edges[pred_u_local]; j<g->num_edges[pred_u_local+1]; j++) {
                uint64_t v = adj[j];
                if (v == u) {
                    found_u = 1;
                    break;
                }
            }

            if (g->n_local_col == g->n_local) {
                if (found_u == 0) {
                    fprintf(stderr, "rank %d, Missing edge: i %lu pred_u %lu u %lu\n", 
                        rank, i, pred_u_local, u);
                    return 0;
                }
            }
        }
        
        if ((pred_u/g->n_local) == rank_rep) {
        
            if (g->n_local % 2 != 0) 
                continue;    
            
            uint32_t pred_u_local_val = pred_u % g->n_local;
            
            uint32_t dv = g->d_trans[pred_u_local_val/2];
            
            if ((pred_u_local_val & 1U) == 0) { 
                dv = (dv & even_mask)>>4;
            } else {
                dv = (dv & odd_mask);
            }
            
            if (dv != (du - 1)) {
                fprintf(stderr, "rank %d, Error: levels do not match for "
                        "%u (%u) %lu (%u)\n", 
                        rank, u_local_val, du, pred_u, dv);
                return 0;

            }
        }
        
    }

    return 1;
}


int validate_bfs_result_threaded(dist_graph_t *g, uint64_t root, uint64_t* pred, uint64_t
        pred_array_size) {


    uint64_t i, j;
    uint64_t bitmask63;
    uint32_t *adj;
    int rank_rep = (rank % (g->nproc_rows * g->nproc_cols));
 
    adj = g->adj;
#if __x86_64__
    bitmask63 = (1UL<<63)-1;
#else
    bitmask63 = (1ULL<<63)-1;
#endif

#if !STORE_PRED
    return 1;
#endif

    /* Only run validation for small problem sizes */
    if (g->n > (1<<25))
        return 1;

    /* Sort the predecessor array by start vertex */
    // qsort(pred, pred_array_size, 16, cmp_fun); 

    /* Check if there's an edge corresponding to [u, pred(u)] */
    /* For local vertices, also check BFS level */ 
    for (i = 0; i < 2*pred_array_size; i+=2) {
        uint64_t u = pred[i];
        uint64_t pred_u = pred[i+1];
        //uint64_t u_local = u % g->n_local_row;
        if ((u == 0) && (pred_u == 0)) {
            continue;
        }

        if (pred_u == bitmask63) {
            if ((root % g->n_local_col) == u) {  
                continue;
            } else {
                fprintf(stderr, "root vertex not found %lu %lu\n", u, root);
                return 0;
            }
        }
        
        uint64_t u_global = u + (g->comm_data).jcol * g->n_local_col;

#if REPLICATE_D
        uint32_t u_local_val = u;
        uint32_t du = g->d[u_local_val];
        if (du == 0) {
            fprintf(stderr, "rank %d, u %lu, ucolproc %lu\n", 
                    rank, u_global, u/g->n_local);
            assert(du > 0);
        } 
#else
        uint32_t u_local_val = u_global % g->n_local;

        uint32_t du = g->d[u_local_val];
        assert(du > 0); 
#endif
                  
        int found_u = 0;
        /* Check the adjacency list of pred_u for u */
        //uint64_t pred_u_local = pred_u % g->n_local_row;
        //assert((pred_u/g->n_local_row) == (g->comm_data).irow);

        if ((pred_u/g->n_local_row) == (g->comm_data).irow) {
       
            uint64_t u_local_row = (u + (g->comm_data).jcol *  g->n_local_col) %
                g->n_local_row;
            assert(u_local_row < g->n_local_row);
            uint64_t pred_u_local = pred_u % g->n_local_row;

            for (j=g->num_edges[pred_u_local]; j<g->num_edges[pred_u_local+1]; j++) {
                uint64_t v = adj[j];
                if (v == u) {
                    found_u = 1;
                    break;
                }
            }

            if (g->n_local_col == g->n_local) {
                if (found_u == 0) {
                    fprintf(stderr, "rank %d, Missing edge: i %lu pred_u %lu u %lu\n", 
                        rank, i, pred_u_local, u);
                    return 0;
                }
            }
        }
        
        if ((pred_u/g->n_local) == rank_rep) {
        
            uint32_t pred_u_local_val = pred_u % g->n_local;
            
            uint32_t dv = g->d_trans[pred_u_local_val];
            
            if (dv != (du - 1)) {
                fprintf(stderr, "rank %d, edge %lu, Error: levels do not match for "
                        "%u (%u) %lu (%u)\n", 
                        rank, i, 
                        u_local_val, du, pred_u, dv);
                return 0;

            }
        }
        
    }

    return 1;

}
