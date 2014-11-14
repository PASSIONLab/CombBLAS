#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <stdint.h>
#include <limits.h>

#if defined(__x86_64__)
#include <mm_malloc.h>
#else 
#define _mm_malloc(a,b) malloc((a))
#define _mm_free(a) free((a))
#endif

extern int rank;
extern int nprocs;

typedef uint32_t vid_t;
typedef uint32_t eid_t;

#define MAX_NUMPROCS 1024

#define ROOT_VERT_RNG_SEED 2323232

#define IDEAL_ALIGNMENT 16

#define TIME_BFS_SUBROUTINES 1

#define LOCAL_QUEUE_SIZE 512
#define LOCAL_SENDBUF_SIZE 64

#define PROC_ASSERT_CHECK 1

#define PRED_CACHE_BYPASS 1
#define STORE_PRED 0

#define SAMEPROC_NOREAD 1

#define REPLICATE_D 1

typedef struct {

    /* for all-to-all communication */
    int32_t *sendbuf_counts;
    int32_t *recvbuf_counts;
    
    /* displs precomputed when graph is created */
    int32_t *sendbuf_displs;
    int32_t *recvbuf_displs;
 
    uint32_t *adj_sendbuf;
    uint32_t *adj_recvbuf;
    int32_t *adj_sendbuf_counts;
    int32_t *adj_recvbuf_counts;
    int32_t *adj_sendbuf_displs;
    int32_t *adj_recvbuf_displs;

#if USE_MPI
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm replicas_comm;
    MPI_Comm replicas_comm2;
#endif
    int irow;
    int jcol;
    int krep;

} dist_graph_comm_data_t;


typedef struct {

    uint64_t n;
    uint64_t m;

    uint64_t n_local;
    uint64_t m_local;

    uint64_t n_local_row;
    uint64_t n_local_col;

    uint64_t m_inter_part_edges;

    vid_t *adj;
    
    eid_t *num_edges;

    vid_t *original_vertex_ids;

    uint8_t *d;
    uint8_t *d_trans;
    uint8_t *d_trans_full;
    uint32_t *queue_current;
    uint32_t *queue_next;
    dist_graph_comm_data_t comm_data;
    
    int nproc_rows;
    int nproc_cols;
    int nreplicas;
    int nprocs_per_replica;



} dist_graph_t;


typedef struct {
    
    uint64_t n;
    uint64_t n_local;
    uint64_t n_start;
    uint64_t m;
    uint64_t m_local;
    uint64_t m_local_allocsize;
    int SCALE;
    uint32_t *gen_edges;   

} graph_gen_data_t;

typedef struct {

    /* used for local binning */
    /* Each array is of size nthreads*nprocs */
    int *pedge_bin_counts;
    int *pedge_bin_displs;

    uint64_t *phisto_counts;
    uint64_t *phisto_displs;
    uint64_t *phisto_counts_global;

    /* for all-to-all communication */
    /* arrays of size nprocs */ 
    int *sendbuf_counts;
    int *sendbuf_displs;

    int *recvbuf_counts;
    int *recvbuf_displs;

    uint32_t *sendbuf_edges;
    uint32_t *recvbuf_edges;

    uint64_t *nrange;

    uint64_t *perm_sendbuf;
    uint64_t *perm_recvbuf;
    uint64_t  perm_recvbuf_size;
    uint64_t  perm_n_offset;

} graph_gen_aux_data_t;


int gen_graph_edges(graph_gen_data_t *ggi, graph_gen_aux_data_t *ggaux);

int run_bfs(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited);
int run_bfs_threaded(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited);
int run_bfs_2Dgraph(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited);
int run_bfs_2Dgraph_threaded(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited);


int create_dist_graph(const uint64_t num_edges, uint32_t *edges, dist_graph_t *g);
int create_2Ddist_graph(const uint64_t num_edges, uint32_t *edges, dist_graph_t *g);

int free_graph(dist_graph_t *g);
int validate_bfs_result(dist_graph_t *g, uint64_t root, 
        uint64_t* pred, uint64_t pred_array_size);
int validate_bfs_result_threaded(dist_graph_t *g, uint64_t root, 
        uint64_t* pred, uint64_t pred_array_size);


int find_bfs_start_vertices(int num_bfs_roots, dist_graph_t* g, uint64_t* bfs_roots);

double get_seconds();

int local_parallel_quicksort (void *data, int64_t n, size_t elem_size, 
        int (*cmpfn)(const void *x, const void *y));

void get_statistics(const double *x, int n, double *r);


#endif
