#include <mpi.h>
extern "C" {
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
}
#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <algorithm>

#define USE_GNU_PARALLELMODE 0

#if USE_GNU_PARALLELMODE
#include <parallel/algorithm>
#endif

#define TIME_GRAPHCREATE_STEPS 0

typedef struct {
    uint32_t u;
    uint32_t v;
    uint32_t uv_off;
} packed_edge;


static int vid_cmp(const void *a, const void *b) {
    const uint16_t *ap;
    const uint16_t *bp;
    uint64_t av, bv, av32, av16, bv32, bv16;
    ap = (const uint16_t *) a;
    bp = (const uint16_t *) b;

    av32 = ap[0]; av16 = ap[1];
    bv32 = bp[0]; bv16 = bp[1];
    
    av = (av32<<32) + (av16<<16) + ap[2];
    bv = (bv32<<32) + (bv16<<16) + bp[2];
    if (av < bv)
        return -1;
    if (av > bv)
        return 1;
    return 0;
}


inline static bool eid_ucmp_cpp(const packed_edge& a, const packed_edge& b) {

    const uint64_t bitmask16 = ((1UL<<16) - 1);
    uint64_t auv_off = a.uv_off;
    uint64_t buv_off = b.uv_off;
   
    uint64_t au = a.u + ((auv_off & bitmask16)<<32);
    uint64_t bu = b.u + ((buv_off & bitmask16)<<32);

    return (au < bu);
}

inline static bool eid_vcmp_cpp(const packed_edge &a, const packed_edge &b) {

    uint64_t auv_off = a.uv_off;
    uint64_t buv_off = b.uv_off;
   
    uint64_t av = a.v + ((auv_off >> 16)<<32);
    uint64_t bv = b.v + ((buv_off >> 16)<<32);

    return (av < bv);
}

static int Alltoall_inplace_exchange_and_compact(
        void *val_ptr, 
        int elem_size, 
        int32_t *sendbuf_counts, int32_t *sendbuf_displs, 
        int32_t *recvbuf_displs, 
        int32_t *sendrecvbuf_counts, int32_t *sendrecvbuf_displs) {

    uint32_t *graph_gen_edges = (uint32_t *) val_ptr;

    /* 
    if (rank == 1) {
        for (int i=0; i<nprocs; i++) {
            fprintf(stderr, "init sendbuf to rank %d, sendsize %d, arrsize %d\n", 
                    i, sendbuf_counts[i],
                sendrecvbuf_counts[i]);
            for (uint64_t j = sendbuf_displs[i]; j <
                    sendbuf_displs[i]+sendbuf_counts[i]; j++) {
                fprintf(stderr, "[%u %u]", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            }
            fprintf(stderr, "\n");
        }
    }
    */

    /* */
    assert(elem_size == 3*sizeof(uint32_t));

    for (int i=nprocs-1; i>=0; i--) {
        
        int64_t offset = sendrecvbuf_displs[i] + sendrecvbuf_counts[i] - 1 
                           - (sendbuf_displs[i] + sendbuf_counts[i] - 1);

        assert(offset >= 0);
        for (int64_t j=sendbuf_displs[i]+sendbuf_counts[i]-1; j>=sendbuf_displs[i]; j--) {
            graph_gen_edges[3*(offset+j)] = graph_gen_edges[3*j];
            graph_gen_edges[3*(offset+j)+1] = graph_gen_edges[3*j+1];
            graph_gen_edges[3*(offset+j)+2] = graph_gen_edges[3*j+2];
        }
    }

    /* Move holes to the right */
    for (int i=0; i<nprocs; i++) {
       
        if (sendbuf_counts[i] == sendrecvbuf_counts[i]) {
            continue;
        }

        int64_t offset = sendrecvbuf_counts[i] - sendbuf_counts[i];
        assert(offset > 0); 

        for (int64_t j=sendrecvbuf_displs[i]; j<sendrecvbuf_displs[i]+sendbuf_counts[i]; j++) {
            graph_gen_edges[3*j] = graph_gen_edges[3*(offset+j)];
            graph_gen_edges[3*j+1] = graph_gen_edges[3*(offset+j)+1];
            graph_gen_edges[3*j+2] = graph_gen_edges[3*(offset+j)+2];
        }
        for (int64_t j=sendrecvbuf_displs[i]+sendbuf_counts[i]; j<sendrecvbuf_displs[i+1]; j++) {
            graph_gen_edges[3*j] = 0;
            graph_gen_edges[3*j+1] = 0;
            graph_gen_edges[3*j+2] = 0;
        }
    }
#if 0
    if (rank == 0) {
        for (int i=0; i<nprocs; i++) {
            fprintf(stderr, "sendbuf to rank %d, sendsize %d, arrsize %d\n", i, sendbuf_counts[i],
                sendrecvbuf_counts[i]);
            for (uint64_t j = sendrecvbuf_displs[i]; j <
                    sendrecvbuf_displs[i]+sendbuf_counts[i]; j++) {
                fprintf(stderr, "[%u %u]", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            }
            fprintf(stderr, "\n");
        }
    }

    if (rank == 1) {
        for (int i=0; i<nprocs; i++) {
            fprintf(stderr, "sendbuf to rank %d, sendsize %d, arrsize %d\n", i, sendbuf_counts[i],
                sendrecvbuf_counts[i]);
            for (uint64_t j = sendrecvbuf_displs[i]; j <
                    sendrecvbuf_displs[i]+sendbuf_counts[i]; j++) {
                fprintf(stderr, "[%u %u]", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    /* Move graph_gen_edges */
#if USE_MPI
    for (int i=0; i<nprocs; i++) {
        
        for (int j=i; j<nprocs; j++) {
            
            if (rank == i) {
                
                /* Send to and recv from j */
                MPI_Status sr_status;
                int sr_size = sendrecvbuf_counts[j];
                uint32_t* buf_ptr = graph_gen_edges + 3*sendrecvbuf_displs[j];
                MPI_Sendrecv_replace(buf_ptr, 3*sr_size, MPI_UNSIGNED, j, (j+1)+(i+1), 
                        j, (j+1)+(i+1), MPI_COMM_WORLD, &sr_status);

            } else if (rank == j) {

                /* Send to and recv from i */
                MPI_Status sr_status;
                int sr_size  = sendrecvbuf_counts[i];
                uint32_t* buf_ptr = graph_gen_edges + 3*sendrecvbuf_displs[i];
                MPI_Sendrecv_replace(buf_ptr, 3*sr_size, MPI_UNSIGNED, i, (i+1)+(j+1), 
                        i, (i+1)+(j+1), MPI_COMM_WORLD, &sr_status);

            }

        }
    }
#endif
#if 0
    if (rank == 0) {
        for (int i=0; i<nprocs; i++) {
            fprintf(stderr, "recvbuf from rank %d, recvsize %d, arrsize %d\n", i, recvbuf_counts[i],
                sendrecvbuf_counts[i]);
            for (uint64_t j = sendrecvbuf_displs[i]; j <
                    sendrecvbuf_displs[i]+recvbuf_counts[i]; j++) {
                fprintf(stderr, "[%u %u]", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            }
            fprintf(stderr, "\n");
        }
    }

    if (rank == 1) {
        for (int i=0; i<nprocs; i++) {
            fprintf(stderr, "recvbuf from rank %d, recvsize %d, arrsize %d\n", i, recvbuf_counts[i],
                sendrecvbuf_counts[i]);
            for (uint64_t j = sendrecvbuf_displs[i]; j <
                    sendrecvbuf_displs[i]+recvbuf_counts[i]; j++) {
                fprintf(stderr, "[%u %u]", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    /* Compact array to remove holes */
    for (int i=0; i<nprocs; i++) {
        int64_t k = sendrecvbuf_displs[i];
        for (int64_t j=recvbuf_displs[i]; j<recvbuf_displs[i+1]; j++) {
            graph_gen_edges[3*j] = graph_gen_edges[3*k];
            graph_gen_edges[3*j+1] = graph_gen_edges[3*k+1];
            graph_gen_edges[3*j+2] = graph_gen_edges[3*k+2];
            //if (rank == 0)
            //    fprintf(stderr, "[%u %u] ", graph_gen_edges[3*j], graph_gen_edges[3*j+1]);
            k++;
        }
        //fprintf(stderr, "\n");
    }

    return 0;
}

int create_2Ddist_graph(const uint64_t nedges_local, uint32_t* edges_ptr, dist_graph_t* g) {

    uint32_t *graphgen_edges;

    int32_t *sendbuf_counts, *sendbuf_displs;
    int32_t *recvbuf_counts, *recvbuf_displs;

    uint64_t sendbuf_size, recvbuf_size;

    uint32_t *vertex_degrees, *vertex_displs;

    uint64_t *n_max;
    uint64_t n_max_l, n_global;
    uint64_t n_local, n, m_local;
    uint64_t n_local_row, n_local_col;
    int64_t nproc_rows, nproc_cols;
    int irow, jcol, krep;
    
    nproc_rows = g->nproc_rows;
    nproc_cols = g->nproc_cols;
    krep = rank/(nproc_rows*nproc_cols);
    irow = (rank-krep*nproc_cols*nproc_rows)/nproc_cols;
    jcol =
        (rank-krep*nproc_cols*nproc_rows-irow*nproc_cols);
    
    const uint64_t bitmask16 = (1UL<<16)-1;
#if defined(__x86_64__)
    const uint64_t bitmask32 = (1UL<<32)-1;
#else
    const uint64_t bitmask32 = (1ULL<<32)-1;
#endif
#if TIME_GRAPHCREATE_STEPS
    double elt;
#endif
    //int32_t degree0_count, degree1_count, degree2_count, hdegree_count;
    //int32_t local_queue_size;
  
    if (6UL*nedges_local >= bitmask32) {
        fprintf(stderr, "Error! Local edge count is greater than 32-bit indexing limit!\n");
        assert(6UL*nedges_local < bitmask32);
    }

    graphgen_edges = edges_ptr;

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds();
#endif

    /* Scan edges once to get global vertex count */
#pragma omp parallel
{
    
    int tid, nthreads;
    uint64_t n_max_local;

#ifdef _OPENMP
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else
    tid = 0;
    nthreads = 1;
#endif

    if (tid == 0) {
        n_max = (uint64_t *) _mm_malloc (nthreads * 8 * sizeof(uint64_t), 4*IDEAL_ALIGNMENT);
        assert(n_max != NULL);
    }

 
#pragma omp barrier

    n_max_local = 0;
#pragma omp for
    for (int64_t i=0; i<((int64_t)nedges_local); i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        uint64_t uv_off = graphgen_edges[3*i+2];
        
        u += ((uv_off & bitmask16)<<32);
        v += ((uv_off>>16)<<32);

        if (u > n_max_local)
            n_max_local = u;
        if (v > n_max_local)
            n_max_local = v;
    }

    n_max[tid*8] = n_max_local;

#pragma omp barrier
    if (tid == 0) {
        n_max_l = n_max[0];
        for (int i=1; i<nthreads; i++) {
            if (n_max[i*8] > n_max_l)
                n_max_l = n_max[i*8];
        }
    }

    if (tid == 0) {
        _mm_free(n_max);
    }
#pragma omp barrier

}

#if USE_MPI
    MPI_Allreduce(&n_max_l, &n_global, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    n_global = n_max_l;
#endif

    n_global++;
    /* Round to nearest power of 2 */
    /*
    uint64_t ng = n_global;
    uint64_t log2_n = 0;
    while (ng != 0) {
        log2_n++;
        ng = (ng>>1);
    }

    n_global = (1UL<<log2_n);
    //n_global++;
    */

    
    if (n_global % nprocs != 0)
        n_global = (n_global/nprocs + 1) * nprocs;
  
    n = n_global;
    n_local = n/(nproc_cols*nproc_rows);

    n_local_row = n_local * nproc_cols;
    n_local_col = n_local * nproc_rows;
 
    /*
    if (rank == 0)
        fprintf(stderr, "n %lu, n_local %lu, n_local_row %lu init m_local %lu\n", 
            n, n_local, n_local_row, nedges_local);
    */

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Find global vertex count:     %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    sendbuf_counts = (int32_t *) malloc(nprocs * sizeof(int32_t));
    recvbuf_counts = (int32_t *) malloc(nprocs * sizeof(int32_t));
    sendbuf_displs = (int32_t *) malloc((nprocs + 1) * sizeof(int32_t));
    recvbuf_displs = (int32_t *) malloc((nprocs + 1) * sizeof(int32_t));

    for (int i=0; i<nprocs; i++) {
        sendbuf_counts[i] = 0;
        recvbuf_counts[i] = 0;
    }

    /* ToDo: parallelize */
    /* Bin edges based on owner process */
    for (int64_t i = 0; i < (int64_t) nedges_local; i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        //uint64_t uv_off = graphgen_edges[3*i+2];
        //u += ((uv_off & bitmask16) << 32);
        //v += ((uv_off >> 16) << 32);

        int64_t uv_eproc = (u/n_local_row) * nproc_cols +
            (v/n_local_col);

        int64_t vu_eproc = (v/n_local_row) * nproc_cols +
            (u/n_local_col);
      
        assert(vu_eproc < nproc_cols * nproc_rows);

        sendbuf_counts[uv_eproc] += 2;
        sendbuf_counts[vu_eproc] += 2;
   
#if 0 
        uint64_t v_proc = v / n_local;
        if (u != v)
            sendbuf_counts[v_proc] += 1;
#endif
    }
    

    sendbuf_displs[0] = 0;
    for (int i=1; i<=nprocs; i++) {
        sendbuf_displs[i] = sendbuf_displs[i-1] + sendbuf_counts[i-1];
    }
    sendbuf_size = sendbuf_displs[nprocs];

    assert(sendbuf_size == 4*nedges_local);

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);    
 
    MPI_Alltoall(sendbuf_counts, 1, MPI_INT, 
            recvbuf_counts, 1, MPI_INT, MPI_COMM_WORLD);
#else
    recvbuf_counts[0] = sendbuf_counts[0];
#endif

    recvbuf_displs[0] = 0;
    for (int i=1; i<=nprocs; i++) {
        recvbuf_displs[i] = recvbuf_displs[i-1] + recvbuf_counts[i-1];
    }
    recvbuf_size = recvbuf_displs[nprocs];
  
    uint64_t recvbuf_size_max = 0; 
#if USE_MPI
    MPI_Reduce(&recvbuf_size, &recvbuf_size_max, 1, 
            MPI_UNSIGNED_LONG, MPI_MAX, 
            0, MPI_COMM_WORLD);
#else
    recvbuf_size_max = recvbuf_size;
#endif
    /*
    if (rank == 0) {
        fprintf(stderr, "Peak load imbalance: %3.4lf\n",
                ((double)recvbuf_size_max/(4*nedges_local)));
    }
    */

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Update send/recv counts:      %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    uint32_t *sendbuf_edges = (uint32_t *) malloc(sendbuf_size *
            sizeof(uint32_t));
    assert(sendbuf_edges != NULL);
  
    for (int64_t i=0; i<nprocs; i++) {
        sendbuf_counts[i]= 0;
    }

    /* Pack up sendbuf edges */
    for (int64_t i = 0; i < (int64_t) nedges_local; i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];

        uint64_t uv_eproc = (u/n_local_row) * nproc_cols +
            (v/n_local_col);

        uint64_t vu_eproc = (v/n_local_row) * nproc_cols +
            (u/n_local_col);
        
        uint64_t uv_insertpos = sendbuf_displs[uv_eproc] + 
                                sendbuf_counts[uv_eproc];
        sendbuf_counts[uv_eproc] += 2;
       
        sendbuf_edges[uv_insertpos] = u;
        sendbuf_edges[uv_insertpos+1] = v;
        
        uint64_t vu_insertpos = sendbuf_displs[vu_eproc] + 
                                sendbuf_counts[vu_eproc];
        sendbuf_counts[vu_eproc] += 2;
  
        sendbuf_edges[vu_insertpos] = v;
        sendbuf_edges[vu_insertpos+1] = u; 
#if 0 
        uint64_t v_proc = v / n_local;
        if (u != v)
            sendbuf_counts[v_proc] += 1;
#endif
    }

    free(graphgen_edges);
    
    uint32_t *recvbuf_edges = (uint32_t *) malloc(recvbuf_size *
            sizeof(uint32_t));
    assert(recvbuf_edges != NULL);
 
#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Packing up sendbuf:            %9.6lf s.\n", elt);
    elt = get_seconds();
#endif
 
   assert(recvbuf_displs[nprocs] == (int)recvbuf_size);
   assert(sendbuf_displs[nprocs] == (int)sendbuf_size);

#if USE_MPI 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoallv(sendbuf_edges, 
            sendbuf_counts, sendbuf_displs, MPI_UNSIGNED, 
            recvbuf_edges, 
            recvbuf_counts, recvbuf_displs, MPI_UNSIGNED, 
            MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    memcpy(recvbuf_edges, sendbuf_edges,
            sizeof(uint32_t)*sendbuf_counts[0]);
#endif
    free(sendbuf_edges);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Alltoall exchange:            %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    vertex_degrees = (uint32_t *) malloc(n_local_row 
            * sizeof(uint32_t));
    assert(vertex_degrees != NULL);

    vertex_displs = (uint32_t *) malloc((n_local_row +1)
            * sizeof(uint32_t));
    assert(vertex_displs != NULL);

#pragma omp parallel for 
    for (uint64_t i=0; i<n_local_row; i++) {
        vertex_degrees[i] = 0;
        vertex_displs[i] = 0;
    }

#pragma omp parallel for
    for (uint64_t i=0; i<recvbuf_size; i+=2) {
        //assert((recvbuf_edges[i]/n_local_row) == irow);
        //assert((recvbuf_edges[i+1]/n_local_col) == jcol);
        uint64_t u = recvbuf_edges[i] % n_local_row;
        __sync_fetch_and_add(vertex_degrees+u, 1);
        //vertex_degrees[u]++;
    }

    m_local = recvbuf_size/2;

    g->m_local = m_local;
#if USE_MPI
    MPI_Comm_split(MPI_COMM_WORLD, krep, irow*nproc_cols+jcol, 
            &((g->comm_data).replicas_comm));
    MPI_Comm_split(MPI_COMM_WORLD, irow*nproc_cols+jcol, krep,
            &((g->comm_data).replicas_comm2));
    MPI_Comm_split((g->comm_data).replicas_comm, irow, jcol,
            &((g->comm_data).row_comm));
    MPI_Comm_split((g->comm_data).replicas_comm, jcol, irow,
            &((g->comm_data).col_comm));
#endif
    (g->comm_data).irow = irow;
    (g->comm_data).jcol = jcol;
    (g->comm_data).krep = krep;

#if USE_MPI
    MPI_Bcast(&(g->m_local), 1, MPI_UNSIGNED_LONG, 0, 
            (g->comm_data).replicas_comm2);
#endif
    m_local = g->m_local;

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Global vertex degree update:  %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

#if USE_MPI
    MPI_Barrier((g->comm_data).replicas_comm);
#endif

    vertex_displs[0] = 0;
    /* update vertex degrees */
    for (uint64_t i=0; i<n_local_row; i++) {
        vertex_displs[i+1] = vertex_degrees[i] + vertex_displs[i];
    }

    uint32_t *adj = (uint32_t *) malloc(m_local * sizeof(uint32_t));
    assert(adj != NULL);

#pragma omp parallel for
    for (uint64_t i=0; i<m_local; i++) {
        adj[i] = 0;
    }

    for (uint64_t i=0; i<n_local_row; i++) {
        vertex_degrees[i] = 0;
    }

#pragma omp parallel for
    for (uint64_t i=0; i<recvbuf_size; i+=2) {
        uint64_t u = recvbuf_edges[i] % n_local_row;
        uint64_t epos = vertex_displs[u] + __sync_fetch_and_add(vertex_degrees+u, 1);
        adj[epos] = recvbuf_edges[i+1] % n_local_col; 
    }

    free(vertex_degrees);
    free(recvbuf_edges);

    g->adj = adj;
    g->num_edges = vertex_displs;
    g->n_local = n_local;
    g->n_local_row = n_local_row;
    g->n_local_col = n_local_col;
    g->n = n;
 
#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Creating adj:  %9.6lf s.\n", elt);
    elt = get_seconds();
#endif


    (g->comm_data).sendbuf_counts = sendbuf_counts;
    (g->comm_data).sendbuf_displs = sendbuf_displs;
    (g->comm_data).recvbuf_counts = recvbuf_counts;
    (g->comm_data).recvbuf_displs = recvbuf_displs;

    /* Broadcast g from replica 0 to all other replicas */
#if USE_MPI
    MPI_Bcast(g->adj, g->m_local, MPI_UNSIGNED, 0, 
            (g->comm_data).replicas_comm2);
    MPI_Bcast(g->num_edges, n_local_row+1, MPI_UNSIGNED, 0,
            (g->comm_data).replicas_comm2);
#endif
   
    //g->m_inter_part_edges = inter_part_count;

#if REPLICATE_D
    g->d = (uint8_t *) malloc(n_local_col * sizeof(uint8_t));
    assert(g->d != NULL);
#else
    g->d = (uint8_t *) malloc(n_local * sizeof(uint8_t));
    assert(g->d != NULL);
#endif

    g->queue_current = (uint32_t *) _mm_malloc(n_local_row * sizeof(uint32_t), 16);
    g->queue_next    = (uint32_t *) _mm_malloc(n_local_row * sizeof(uint32_t), 16);
    
    assert(g->queue_current != NULL);
    assert(g->queue_next != NULL);

    int32_t *a2a_sendbuf_counts = (int32_t *) malloc(nproc_rows
            * sizeof(int32_t));
    int32_t *a2a_sendbuf_displs = (int32_t *) malloc(nproc_rows
            * sizeof(int32_t));
    int32_t *a2a_recvbuf_counts = (int32_t *) malloc(nproc_rows
            * sizeof(int32_t));
    int32_t *a2a_recvbuf_displs = (int32_t *) malloc(nproc_rows
            * sizeof(int32_t));
    assert(a2a_sendbuf_counts != NULL);
    assert(a2a_sendbuf_displs != NULL);
    assert(a2a_recvbuf_counts != NULL);
    assert(a2a_recvbuf_displs != NULL);


    for (int64_t i=0; i<nproc_rows; i++) {
        a2a_sendbuf_counts[i] = 0;
        a2a_recvbuf_counts[i] = 0;
        a2a_sendbuf_displs[i] = 0;
        a2a_recvbuf_displs[i] = 0;
    }

    for (uint64_t i=0; i<n_local_row; i++) {
        for (uint64_t j=g->num_edges[i]; j<g->num_edges[i+1]; j++) {
            int64_t v = g->adj[j];
            int64_t v_local_row_rank = v/n_local;
            assert(v_local_row_rank < nproc_rows);
            a2a_sendbuf_counts[v_local_row_rank] += 2;
        }
    }

    for (int64_t i=1; i<nproc_rows; i++) {
        a2a_sendbuf_displs[i] = a2a_sendbuf_displs[i-1] +
            a2a_sendbuf_counts[i-1];
    }

#if USE_MPI
    MPI_Barrier((g->comm_data).col_comm);    
 
    MPI_Alltoall(a2a_sendbuf_counts, 1, MPI_INT, 
            a2a_recvbuf_counts, 1, MPI_INT,
            (g->comm_data).col_comm);
#else
    a2a_recvbuf_counts[0] = a2a_sendbuf_counts[0];
#endif

    a2a_recvbuf_displs[0] = 0;
    for (int i=1; i<nproc_rows; i++) {
        a2a_recvbuf_displs[i] = a2a_recvbuf_displs[i-1] + 
            a2a_recvbuf_counts[i-1];
    }
    uint64_t a2a_recvbuf_size = a2a_recvbuf_displs[nproc_rows-1] +
                        a2a_recvbuf_counts[nproc_rows-1];
  
    uint32_t *a2a_recvbuf = (uint32_t *) malloc(a2a_recvbuf_size *
            sizeof(uint32_t));
 
    uint32_t *a2a_sendbuf = (uint32_t *) malloc(2 * m_local *
            sizeof(uint32_t));
    assert(a2a_sendbuf != NULL);
    assert(a2a_recvbuf != NULL);

#if TIME_GRAPHCREATE_STEPS
    long max_ecount = 0;
    long total_ecount = 0;
    long total_edgecut = 0;
    long max_edgecut = 0;

    long local_ecount = m_local;
    long local_edgecut = (a2a_recvbuf_size - a2a_recvbuf_counts[irow])/2;

    MPI_Allreduce(&local_ecount, &max_ecount, 1, MPI_LONG, 
            MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_ecount, &total_ecount, 1, MPI_LONG, 
            MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&local_edgecut, &max_edgecut, 1, MPI_LONG, 
            MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_edgecut, &total_edgecut, 1, MPI_LONG, 
            MPI_SUM, MPI_COMM_WORLD);


    if (rank == 0) {
        fprintf(stderr, "Total edge cut: %ld\n", total_edgecut/2);
        fprintf(stderr, "Max edge cut: %ld\n", max_edgecut);
        fprintf(stderr, "Max edge count: %ld\n", max_ecount);
        fprintf(stderr, "Avg edge count: %ld\n",
                total_ecount/(2*nprocs));
    }
    //fprintf(stderr, "rank %d ecount %ld\n", rank, local_ecount);
#endif

#pragma omp parallel for
    for (uint64_t i=0; i<2*m_local; i++) {
        a2a_sendbuf[i] = 0;
    }

#pragma omp parallel for
    for (uint64_t i=0; i<a2a_recvbuf_size; i++) {
        a2a_recvbuf[i] = 0;
    }


    /*
    fprintf(stderr, "recvbuf size %ld, sendbuf size %ld\n", 
            a2a_recvbuf_size, 2*m_local);
    */
 
#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Creating adj_send/recv buf:  %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    (g->comm_data).adj_sendbuf = a2a_sendbuf;
    (g->comm_data).adj_recvbuf = a2a_recvbuf;
    (g->comm_data).adj_sendbuf_counts = a2a_sendbuf_counts;
    (g->comm_data).adj_recvbuf_counts = a2a_recvbuf_counts;
    (g->comm_data).adj_sendbuf_displs = a2a_sendbuf_displs;
    (g->comm_data).adj_recvbuf_displs = a2a_recvbuf_displs;

    return 0;
}


int create_dist_graph(const uint64_t nedges_local, uint32_t* edges_ptr, dist_graph_t* g) {

    uint32_t *graphgen_edges;

    int32_t *sendbuf_counts, *sendbuf_displs;
    int32_t *recvbuf_counts, *recvbuf_displs;
    int32_t *sendrecvbuf_counts, *sendrecvbuf_displs;

    uint64_t sendbuf_size, recvbuf_size, sendrecvbuf_size;

    uint32_t *vertex_degrees, *vertex_displs;
    uint16_t *adj;

    uint64_t *n_max;
    uint64_t n_max_l, n_global;
    uint64_t n_local, n_start, n, m_local, m, m_local_allocsize;

    const uint64_t bitmask16 = (1UL<<16)-1;
#if defined(__x86_64__)
    const uint64_t bitmask32 = (1UL<<32)-1;
#else
    const uint64_t bitmask32 = (1ULL<<32)-1;
#endif
#if TIME_GRAPHCREATE_STEPS
    double elt;
#endif
    int32_t degree0_count, degree1_count, degree2_count, hdegree_count;
    int32_t local_queue_size;
  
    if (6UL*nedges_local >= bitmask32) {
        fprintf(stderr, "Error! Local edge count is greater than 32-bit indexing limit!\n");
        assert(6UL*nedges_local < bitmask32);
    }

    graphgen_edges = edges_ptr;

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds();
#endif

    /* Scan edges once to get global vertex count */
#pragma omp parallel
{
    
    int tid, nthreads;
    uint64_t n_max_local;

#ifdef _OPENMP
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else
    tid = 0;
    nthreads = 1;
#endif

    if (tid == 0) {
        n_max = (uint64_t *) _mm_malloc (nthreads * 8 * sizeof(uint64_t), 4*IDEAL_ALIGNMENT);
        assert(n_max != NULL);
    }

 
#pragma omp barrier

    n_max_local = 0;
#pragma omp for
    for (int64_t i=0; i<((int64_t)nedges_local); i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        uint64_t uv_off = graphgen_edges[3*i+2];
        
        u += ((uv_off & bitmask16)<<32);
        v += ((uv_off>>16)<<32);

        if (u > n_max_local)
            n_max_local = u;
        if (v > n_max_local)
            n_max_local = v;
    }

    n_max[tid*8] = n_max_local;

#pragma omp barrier
    if (tid == 0) {
        n_max_l = n_max[0];
        for (int i=1; i<nthreads; i++) {
            if (n_max[i*8] > n_max_l)
                n_max_l = n_max[i*8];
        }
    }

    if (tid == 0) {
        _mm_free(n_max);
    }
#pragma omp barrier

}

#if USE_MPI
    MPI_Allreduce(&n_max_l, &n_global, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    n_global = n_max_l;
#endif

    /* Round to nearest power of 2 */
    uint64_t ng = n_global;
    uint64_t log2_n = 0;
    while (ng != 0) {
        log2_n++;
        ng = (ng>>1);
    }

    n_global = (1UL<<log2_n);
    //n_global++;
    
    if (n_global % nprocs != 0)
        n_global = (n_global/nprocs + 1) * nprocs;
    
    n = n_global;
    n_local = n/nprocs;

    if (rank == 0)
        fprintf(stderr, "n %lu, n_local %lu, init m_local %lu\n", 
            n, n_local, nedges_local);

    m_local_allocsize = (37UL*nedges_local)/20;
    if (rank == 0)
        fprintf(stderr, "Expected memory utilization: "
            "%9.6lf GB (per task), %9.6lf GB (cumulative), size %9.6lf %9.6lf GB\n", 
            (m_local_allocsize*3*4+n_local*10*4)/1073741824.0, 
            nprocs*(m_local_allocsize*3*4+n_local*10*4)/1073741824.0, 
            nprocs*(nedges_local*3*4)/1073741824.0,
            nprocs*(m_local_allocsize*3*4)/1073741824.0);
 
#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Find global vertex count:     %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    /* Sort edges by start vertex */
    packed_edge *gedges = (packed_edge *) graphgen_edges;
    
#if USE_GNU_PARALLELMODE
    __gnu_parallel::sort(gedges, gedges+nedges_local, eid_ucmp_cpp,
            __gnu_parallel::quicksort_tag());
#else
    std::sort(gedges, gedges+nedges_local, eid_ucmp_cpp);
#endif

    //local_parallel_quicksort(graphgen_edges, nedges_local, 3*sizeof(uint32_t), eid_ucmp);
    //qsort(graphgen_edges, nedges_local, 3*sizeof(uint32_t), eid_ucmp);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Sort edges by start vertex:   %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    sendbuf_counts = (int32_t *) malloc(nprocs * sizeof(int32_t));
    recvbuf_counts = (int32_t *) malloc(nprocs * sizeof(int32_t));
    sendbuf_displs = (int32_t *) malloc((nprocs + 1) * sizeof(int32_t));
    recvbuf_displs = (int32_t *) malloc((nprocs + 1) * sizeof(int32_t));
    sendrecvbuf_counts = (int32_t *) malloc(nprocs * sizeof(int32_t));
    sendrecvbuf_displs = (int32_t *) malloc((nprocs + 1) * sizeof(int32_t));

    for (int i=0; i<nprocs; i++) {
        sendbuf_counts[i] = 0;
        recvbuf_counts[i] = 0;
    }

    /* ToDo: parallelize */
    /* Bin edges based on owner process */
    for (int64_t i = 0; i < (int64_t) nedges_local; i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        uint64_t uv_off = graphgen_edges[3*i+2];
        u += ((uv_off & bitmask16) << 32);
        v += ((uv_off >> 16) << 32);

        int64_t u_proc = u / n_local;
        assert(u_proc < nprocs);
        sendbuf_counts[u_proc] += 1;
   
#if 0 
        uint64_t v_proc = v / n_local;
        if (u != v)
            sendbuf_counts[v_proc] += 1;
#endif
    }
    

    sendbuf_displs[0] = 0;
    for (int i=1; i<=nprocs; i++) {
        sendbuf_displs[i] = sendbuf_displs[i-1] + sendbuf_counts[i-1];
    }
    sendbuf_size = 3*sendbuf_displs[nprocs];

    assert(sendbuf_size == 3*nedges_local);

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);    
 
    MPI_Alltoall(sendbuf_counts, 1, MPI_INT, 
            recvbuf_counts, 1, MPI_INT, MPI_COMM_WORLD);
#else
    recvbuf_counts[0] = sendbuf_counts[0];
#endif

    recvbuf_displs[0] = 0;
    for (int i=1; i<=nprocs; i++) {
        recvbuf_displs[i] = recvbuf_displs[i-1] + recvbuf_counts[i-1];
    }
    recvbuf_size = 3*recvbuf_displs[nprocs];

    for (int i=0; i<nprocs; i++) {
        if (sendbuf_counts[i] > recvbuf_counts[i])
            sendrecvbuf_counts[i] = sendbuf_counts[i];
        else
            sendrecvbuf_counts[i] = recvbuf_counts[i];
    }
    sendrecvbuf_displs[0] = 0;
    for (int i=1; i<=nprocs; i++) {
        sendrecvbuf_displs[i] = sendrecvbuf_displs[i-1] 
            + sendrecvbuf_counts[i-1];
    }
    sendrecvbuf_size = 3*sendrecvbuf_displs[nprocs];

    /*    
    if (rank == 0) 
        fprintf(stderr, "rank %d, sendbuf size %lu, recvbuf size %lu, "
            "sendrecvbuf size %lu, array size %lu\n", 
            rank, sendbuf_size, recvbuf_size, sendrecvbuf_size, 3*nedges_local);
    */
  
    if (sendrecvbuf_size/3UL > m_local_allocsize) {
        fprintf(stderr, "rank %d, sendrecvbuf size: %lu, m_local_allocsize %lu\n",
                rank, sendrecvbuf_size/3UL, m_local_allocsize);
        assert(sendrecvbuf_size/3UL <= m_local_allocsize);
    }

    uint64_t sendrecvbuf_size_in = sendrecvbuf_size/3;
    uint64_t sendrecvbuf_size_max = 0;
#if USE_MPI 
    MPI_Reduce(&sendrecvbuf_size_in, &sendrecvbuf_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 
            0, MPI_COMM_WORLD);
#else
    sendrecvbuf_size_max = sendrecvbuf_size_in;
#endif
    if (rank == 0) {
        fprintf(stderr, "Load imbalance, first redistribution: %3.4lf\n",
                ((double)sendrecvbuf_size_max/nedges_local));
    }

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Update send/recv counts:      %9.6lf s.\n", elt);
    elt = get_seconds();
#endif


    Alltoall_inplace_exchange_and_compact(graphgen_edges, 
            3*sizeof(uint32_t), 
            sendbuf_counts, sendbuf_displs, 
            recvbuf_displs, 
            sendrecvbuf_counts, sendrecvbuf_displs);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Alltoall exchange:            %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    vertex_degrees = (uint32_t *) malloc(n_local * sizeof(uint32_t));
    assert(vertex_degrees != NULL);

    uint32_t *vertex_degrees_local = (uint32_t *) malloc(n_local * sizeof(uint32_t));
    assert(vertex_degrees_local != NULL);

#pragma omp parallel for 
    for (int64_t i=0; i<(int64_t)n_local; i++) {
        vertex_degrees[i] = 0;
        vertex_degrees_local[i] = 0;
    }

    /* ToDo: parallelize */
    n_start = n_local * rank;
    for (uint64_t i=0; i<recvbuf_size/3; i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        uint64_t uv_off = graphgen_edges[3*i+2];
        u += ((uv_off & bitmask16)<<32);
        v += ((uv_off>>16)<<32);
        uint64_t u_local = u - n_start;
        assert(u_local < n_local);
        vertex_degrees_local[u_local]++;
    }
 

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Local vertex degree update:   %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    /* Sort recvbuf by end vertex */
    //sort(graphgen_edges, graphgen_edges+recvbuf_size/3
    gedges = (packed_edge *) graphgen_edges;
#if USE_GNU_PARALLELMODE
    __gnu_parallel::sort(gedges, gedges+recvbuf_size/3, eid_vcmp_cpp, 
            __gnu_parallel::quicksort_tag());
#else
    std::sort(gedges, gedges+recvbuf_size/3, eid_vcmp_cpp);
#endif
    //local_parallel_quicksort(graphgen_edges, recvbuf_size/3, 3*sizeof(uint32_t), eid_vcmp);
    //qsort(graphgen_edges, recvbuf_size/3, 3*sizeof(uint32_t), eid_vcmp);

#if TIME_GRAPHCREATE_STEPS           
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Sort edges by end vertex:     %9.6lf s.\n", elt);
    elt = get_seconds();
#endif


    /* Perform nprocs Reduce ops to update global 
       vertex degrees */
    /* Create <vertex, degree> arrays as required */

    uint32_t *vertex_degrees_sendbuffer = (uint32_t *) malloc(2 * n_local * sizeof(uint32_t));
    assert(vertex_degrees_sendbuffer != NULL);

    uint32_t *vertex_degrees_recvbuffer = (uint32_t *) malloc(2 * n_local * sizeof(uint32_t));
    assert(vertex_degrees_recvbuffer != NULL);


    uint64_t graphgen_edges_startpos = 0;
    uint64_t vpart_size = n_local/nprocs;
    assert(vpart_size > 0);
    if (n_local % nprocs != 0) {
        vpart_size++;
    }

    uint32_t *vertex_degrees_cumulative = (uint32_t *) malloc(vpart_size * sizeof(uint32_t));
    assert(vertex_degrees_cumulative != NULL);
    int64_t u_prev = (1UL<<62);

    for (int i=0; i<nprocs; i++) {

#pragma omp parallel for 
        for (int64_t j=0; j<nprocs; j++) {
            sendbuf_counts[j] = 0;
            recvbuf_counts[j] = 0;
        }

        n_start = n_local * i;
        int64_t vsendbuf_size = 0;
        int64_t vrecvbuf_size = 0;

        for (uint64_t j=graphgen_edges_startpos; j<recvbuf_size/3; j++) {
            uint64_t u = graphgen_edges[3*j];
            uint64_t v = graphgen_edges[3*j+1];
            uint64_t uv_off = graphgen_edges[3*j+2];
            u += ((uv_off & bitmask16)<<32);
            v += ((uv_off>>16)<<32);
            int64_t v_local = v - n_start;
            if (v_local >= (int64_t) n_local) {
                graphgen_edges_startpos = j;
                break;
            }

            if (v_local == u_prev) {
                vertex_degrees_sendbuffer[vsendbuf_size-1]++;
            } else {
                u_prev = v_local;
                vertex_degrees_sendbuffer[vsendbuf_size] = v_local;
                vertex_degrees_sendbuffer[vsendbuf_size+1] = 1;
                vsendbuf_size += 2;
                assert(((int) v_local/vpart_size) < nprocs);
                sendbuf_counts[v_local/vpart_size] += 2;
            }
        }

#if USE_MPI
        MPI_Alltoall(sendbuf_counts, 1, MPI_INT, 
            recvbuf_counts, 1, MPI_INT, MPI_COMM_WORLD);
#else
        recvbuf_counts[0] = sendbuf_counts[0];
#endif

        sendbuf_displs[0] = 0;
        for (int j=1; j<=nprocs; j++) {
            sendbuf_displs[j] = sendbuf_displs[j-1] + sendbuf_counts[j-1];
        }
        if (vsendbuf_size != sendbuf_displs[nprocs]) {
            fprintf(stderr, "rank %d, vsendbuf size %ld, displs %d\n", 
                    rank, vsendbuf_size, sendbuf_displs[nprocs]);
            assert(vsendbuf_size == sendbuf_displs[nprocs]);
        }

        recvbuf_displs[0] = 0;
        for (int j=1; j<=nprocs; j++) {
            recvbuf_displs[j] = recvbuf_displs[j-1] + recvbuf_counts[j-1];
        }
        vrecvbuf_size = recvbuf_displs[nprocs];
        assert(vrecvbuf_size < (int64_t) n_local * 2);

#if USE_MPI
        MPI_Alltoallv(vertex_degrees_sendbuffer, sendbuf_counts, sendbuf_displs, 
            MPI_UNSIGNED, 
            vertex_degrees_recvbuffer, recvbuf_counts, recvbuf_displs, 
            MPI_UNSIGNED, 
            MPI_COMM_WORLD);
#else
        memcpy(vertex_degrees_recvbuffer, vertex_degrees_sendbuffer, 
                sendbuf_counts[0]*sizeof(uint32_t));

#endif

#pragma omp parallel for
        for (int64_t j=0; j<(int64_t)vpart_size; j++) {
            vertex_degrees_cumulative[j] = 0;
        }

        int64_t v_local_start = rank * vpart_size;
//#pragma omp parallel for
        for (int j=0; j<vrecvbuf_size; j+=2) {
            int64_t v_local_p = vertex_degrees_recvbuffer[j] - v_local_start;
            //assert(v_local_p < vpart_size);
            //assert(v_local_p >= 0);
            //uint32_t incr_value = vertex_degrees_recvbuffer[j+1];
            //__sync_fetch_and_add((uint32_t *)&vertex_degrees_cumulative[v_local_p], incr_value);
            vertex_degrees_cumulative[v_local_p] += 
               vertex_degrees_recvbuffer[j+1]; 
        }
#if USE_MPI
        MPI_Gather(vertex_degrees_cumulative, vpart_size, MPI_UNSIGNED, 
                vertex_degrees_recvbuffer, vpart_size, MPI_UNSIGNED, i,
                MPI_COMM_WORLD);
#else
        memcpy(vertex_degrees_recvbuffer, vertex_degrees_cumulative,
                vpart_size*sizeof(uint32_t));
#endif
        assert(vpart_size * nprocs >= n_local);

        if (rank == i) {
#pragma omp parallel for
            for (uint64_t j=0; j<n_local; j++) {
                vertex_degrees[j] = vertex_degrees_recvbuffer[j];
            }
            /*
            for (int j=n_local; j<vpart_size*nprocs; j++) {
                assert(vertex_degrees_recvbuffer[j] == 0);
            }
            */
        }

    }

    free(vertex_degrees_sendbuffer);
    free(vertex_degrees_recvbuffer);
    free(vertex_degrees_cumulative);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Global vertex degree update:  %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* update vertex degres */
    for (uint64_t i=0; i<n_local; i++) {
        vertex_degrees[i] += vertex_degrees_local[i];
    }

    degree0_count = degree1_count = degree2_count = 0;
    hdegree_count = 0;
    
    vertex_displs = (uint32_t *) _mm_malloc((n_local + 1) * sizeof(uint32_t), 16);
    assert(vertex_displs != NULL);
 
    uint32_t *vertex_displs_local = (uint32_t *) 
                                    _mm_malloc((n_local + 1) * sizeof(uint32_t), 16);
    assert(vertex_displs_local != NULL);
    
    vertex_displs[0] = 0;
    vertex_displs_local[0] = 0;
    for (uint64_t i=1; i<=n_local; i++) {
        
        vertex_displs_local[i] = vertex_displs_local[i-1] + 3 * vertex_degrees_local[i-1];
        vertex_displs[i]       = vertex_displs[i-1] + 3 * vertex_degrees[i-1];

        if (vertex_degrees[i-1] == 0)
            degree0_count++;
        if (vertex_degrees[i-1] == 1)
            degree1_count++;
        if (vertex_degrees[i-1] == 2)
            degree2_count++;
        if (vertex_degrees[i-1] > 256)
            hdegree_count++;
    }

    m_local = vertex_displs[n_local]/3;
    
    /* 
    if (rank == 0)
        fprintf(stderr, "rank %d, local edges %lu, degree0 %d, degree1 %d, "
            "degree2 %d, hdegree %d\n", 
            rank, m_local, degree0_count, 
            degree1_count, degree2_count, hdegree_count);
    */

    
    if (m_local/2 >= m_local_allocsize) {
        fprintf(stderr, "rank %d, m_local/2 %lu, m_local_alloc_size %lu\n", 
                rank, m_local/2, m_local_allocsize);
        assert(m_local/2 < m_local_allocsize);
    }

    sendrecvbuf_size_in = m_local/2;
    sendrecvbuf_size_max = 0;
#if USE_MPI
    MPI_Reduce(&sendrecvbuf_size_in, &sendrecvbuf_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 
            0, MPI_COMM_WORLD);
#else
    sendrecvbuf_size_max = sendrecvbuf_size_in;
#endif
    if (rank == 0) {
        fprintf(stderr, "Load imbalance, second redistribution: %3.4lf\n",
                ((double)sendrecvbuf_size_max/nedges_local));
    }


#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Updating vertex displs:       %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    /* Move adj's into correct location */
    //local_parallel_quicksort(graphgen_edges, recvbuf_size/3, 3*sizeof(uint32_t), eid_ucmp);
    gedges = (packed_edge *) graphgen_edges;
#if USE_GNU_PARALLELMODE
    __gnu_parallel::sort(gedges, gedges+recvbuf_size/3, eid_ucmp_cpp, 
            __gnu_parallel::quicksort_tag());
#else
    std::sort(gedges, gedges+recvbuf_size/3, eid_ucmp_cpp);
#endif
    //qsort(graphgen_edges, recvbuf_size/3, 3*sizeof(uint32_t), eid_ucmp);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Sort edges by start vertex:   %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    uint64_t adj_size = 3*m_local;
    if ((adj_size & 1UL) == 1)
        adj_size += 1;
    uint64_t adj_size_max = 0;
    uint64_t m_local_allocsize_globalmin = 0;
#if USE_MPI
    MPI_Allreduce(&adj_size, &adj_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&m_local_allocsize, &m_local_allocsize_globalmin, 1, MPI_UNSIGNED_LONG,
            MPI_MIN, MPI_COMM_WORLD);
#else
    adj_size_max = adj_size;
    m_local_allocsize_globalmin = m_local_allocsize;
#endif
    adj = (uint16_t *) graphgen_edges;

    uint64_t epos = 0;
    n_start = n_local * rank;
    for (uint64_t i=0; i<recvbuf_size/3; i++) {
        uint64_t u = graphgen_edges[3*i];
        uint64_t v = graphgen_edges[3*i+1];
        uint64_t uv_off = graphgen_edges[3*i+2];
        u += ((uv_off & bitmask16)<<32);
        v += ((uv_off>>16)<<32);
        uint64_t v_proc = v/n_local;
        assert((int) v_proc < nprocs);
        uint32_t u_local = u - n_start;
        assert(u_local < n_local);
        adj[epos]   = (v>>32) & bitmask16;
        adj[epos+1] = (v>>16) & bitmask16;
        adj[epos+2] =  v & bitmask16;
        epos += 3;
    }
    assert(epos == vertex_displs_local[n_local]);
    assert(3*vertex_degrees[n_local-1] + vertex_displs[n_local-1] == vertex_displs[n_local]);
    assert(3*vertex_degrees_local[n_local-1] + vertex_displs_local[n_local-1] 
            == vertex_displs_local[n_local]);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Compress adj array:           %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    for (int64_t i=n_local-1; i>=0; i--) {
      
        int64_t offset = vertex_displs[i] + 3*vertex_degrees[i] - 3
                           - (vertex_displs_local[i] + 3*vertex_degrees_local[i] - 3);

        int64_t j_start = ((int64_t) vertex_displs_local[i]) 
                            + ((int64_t) 3L*vertex_degrees_local[i]) - 3L;
        int64_t j_end   = vertex_displs_local[i];

        for (int64_t j=j_start; j>=j_end; j-=3) {

            if (offset+j >= (int64_t) adj_size) {
                fprintf(stderr, "Error! i %ld, offset %ld, j %ld, adj_size %lu, degrees %u"
                        " %u\n", 
                        i, offset, j, adj_size, vertex_degrees_local[i], vertex_degrees[i]);
                assert(offset+j < (int64_t) adj_size);
            }
            adj[offset+j]   = adj[j];
            adj[offset+j+1] = adj[j+1];
            adj[offset+j+2] = adj[j+2];
        }
    }

    /* Move holes to the right */
    for (int64_t i=0; i<(int64_t) n_local; i++) {
       
        uint64_t offset = 3*(vertex_degrees[i] - vertex_degrees_local[i]);
       
        for (uint64_t j=vertex_displs[i]; j<vertex_displs[i]+3*vertex_degrees_local[i]; j+=3) {
            adj[j]   = adj[offset+j];
            adj[j+1] = adj[offset+j+1];
            adj[j+2] = adj[offset+j+2];
        }
        for (uint64_t j=vertex_displs[i]+3*vertex_degrees_local[i]; 
                j<vertex_displs[i+1]; j+=3) {
            adj[j] = bitmask16;
            adj[j+1] = bitmask16;
            adj[j+2] = bitmask16;
        }
    }

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Move local adjs:              %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    for (int64_t i=0; i<(int64_t) n_local; i++) {
        uint64_t u = n_start + i;
        for (uint64_t j=vertex_displs[i]; j<vertex_displs[i]+3*vertex_degrees_local[i]; j+=3) {
            uint64_t v = (((uint64_t)adj[j])<<32) + (((uint64_t)adj[j+1])<<16) +
                         (uint64_t)adj[j+2];   
            uint64_t v_proc = v/n_local;
            if ((int) v_proc >= nprocs) {
                fprintf(stderr, "[%lu %lu] ", u, v);
                assert((int) v_proc < nprocs);
            }
        }
    }

    for (int64_t i=0; i<(int64_t) n_local; i++) {
        vertex_degrees[i] = vertex_degrees_local[i];
    }
    
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int64_t a2a_sendbuf_size = (3*m_local_allocsize_globalmin - adj_size_max/2 - 8)/2;

    a2a_sendbuf_size = (a2a_sendbuf_size/(6*nprocs))*6*nprocs;
    assert(a2a_sendbuf_size > 0);

    uint64_t a2a_procbuf_size = (a2a_sendbuf_size/nprocs);
    assert(a2a_procbuf_size % 3 == 0);

    uint64_t a2a_recvbuf_size = a2a_sendbuf_size;

    uint32_t *a2a_sendbuf = graphgen_edges + adj_size_max/2;
    uint32_t *a2a_recvbuf = graphgen_edges + adj_size_max/2 + a2a_sendbuf_size;

    assert(adj_size_max/2 + a2a_sendbuf_size + a2a_recvbuf_size 
            < 3*m_local_allocsize_globalmin);

    
    //fprintf(stderr, "rank %d, a2a_sendbuf size %ld, a2a_procbuf_size %lu %lu m_local %lu\n", 
    //    rank, a2a_sendbuf_size, a2a_procbuf_size, 
    //    adj_size_max/2, m_local_allocsize_globalmin);
     

    /* Buffered AlltoAll, reuse the a2a bufs multiple times,
       keep updating send and recv counts */
    uint64_t i_startpos = 0;
    uint64_t adj_startpos = vertex_displs[0];

    for (int i=0; i<nprocs; i++) {
        sendbuf_displs[i] = i * a2a_procbuf_size;
        recvbuf_displs[i] = i * a2a_procbuf_size;
    }

    n_start = rank*n_local;
    int a2a_iter_num = 0;
    while (1) {

        int64_t i;
        int64_t j=0; 

        for (i=0; i<nprocs; i++) {
            sendbuf_counts[i] = 0;
            recvbuf_counts[i] = 0;
        }

        int a2a_bufload_done = 0;
        
        for (i=i_startpos; i<(int64_t) n_local; i++) {
            uint64_t u = n_start + i;
            uint64_t j_start;
            if (i == (int64_t) i_startpos) {
                j_start = adj_startpos;
            } else {
                j_start = vertex_displs[i];
            }
            for (j=j_start; 
                    j<(int64_t) vertex_displs[i]+3*vertex_degrees_local[i]; j+=3) {
                uint64_t v = (((uint64_t)adj[j])<<32) + (((uint64_t)adj[j+1])<<16) +
                         (uint64_t)adj[j+2];   
                uint64_t v_proc = v / n_local;
                if ((int) v_proc >= nprocs) {
                    fprintf(stderr, "i %ld, j %ld, v_proc %lu\n", i, j, v_proc);
                    assert((int) v_proc < nprocs);
                }
                /* Send packed <v, i> to v_proc */
                if (sendbuf_counts[v_proc] == (int) a2a_procbuf_size) {
                    i_startpos = i;
                    adj_startpos = j;
                    a2a_bufload_done = 1;
                    break;
                }
                epos = v_proc * a2a_procbuf_size + sendbuf_counts[v_proc];
                a2a_sendbuf[epos] = v & bitmask32;
                a2a_sendbuf[epos+1] = u & bitmask32;
                a2a_sendbuf[epos+2] = (v>>32) + ((u>>32)<<16);
                sendbuf_counts[v_proc] += 3;
            }
            if (a2a_bufload_done)
                break;
        }
 

        int current_proc_done = 0;
        int all_proc_done = 0;

        if ((i == (int64_t) n_local+1) || ((i == (int64_t) n_local) && (j == vertex_displs[n_local-1] +
                    3*vertex_degrees_local[n_local-1]))) {
            current_proc_done = 1;
            i_startpos = n_local+1;
        }
#if USE_MPI
        MPI_Alltoall(sendbuf_counts, 1, MPI_INT, 
            recvbuf_counts, 1, MPI_INT, MPI_COMM_WORLD);

        /*
        fprintf(stderr, "sendbuf count %d displs %d, recv %d %d\n", 
                sendbuf_counts[0], sendbuf_displs[0], recvbuf_counts[0], recvbuf_displs[0]);
        */

        MPI_Alltoallv(a2a_sendbuf, sendbuf_counts, sendbuf_displs, 
            MPI_UNSIGNED, 
            a2a_recvbuf, recvbuf_counts, recvbuf_displs, 
            MPI_UNSIGNED, 
            MPI_COMM_WORLD);
#else
        recvbuf_counts[0] = sendbuf_counts[0];
        //fprintf(stderr, "count %d %d\n", recvbuf_displs[0], recvbuf_counts[0]);
        for (i=recvbuf_displs[0]; i<recvbuf_displs[0]+recvbuf_counts[0]; i++) {
            a2a_recvbuf[i] = a2a_sendbuf[i]; 
            //fprintf(stderr, "%d ", i);
        }
#endif
        for (i=0; i<nprocs; i++) {
            for (j=i*((int64_t)a2a_procbuf_size); 
                    j<i*((int64_t)a2a_procbuf_size) + recvbuf_counts[i]; j+=3) {
                uint64_t u = a2a_recvbuf[j];
                uint64_t v = a2a_recvbuf[j+1];
                uint64_t uv_off = a2a_recvbuf[j+2];
                //assert(uv_off == 0);
                u += ((uv_off & bitmask16) << 32);
                v += ((uv_off >> 16) << 32);

                uint64_t u_local = u - n_start;
                assert(u_local < n_local);
                /* Add v to adjacency array of u_local */
                epos = vertex_displs[u_local] +
                    3*vertex_degrees[u_local];
                if (epos + 2 > vertex_displs[u_local+1]) {
                    fprintf(stderr, "Error! rank %d, u_local %lu, v %lu, epos %lu, degrees %u %u %u\n", 
                            rank, u_local, v, epos, vertex_degrees_local[u_local], 
                            vertex_degrees[u_local],
                            (vertex_displs[u_local+1]-vertex_displs[u_local])/3);
                    assert(epos + 2 <= vertex_displs[u_local+1]);
                }
                assert(adj[epos] == bitmask16);
                assert(adj[epos+1] == bitmask16);
                assert(adj[epos+2] == bitmask16);
                 
                adj[epos]   = (v>>32) & bitmask16;
                adj[epos+1] = (v>>16) & bitmask16;
                adj[epos+2] =  v & bitmask16;
                vertex_degrees[u_local]++;
            }
        }
#if USE_MPI
        MPI_Allreduce(&current_proc_done, &all_proc_done, 1, 
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
        all_proc_done = current_proc_done;
#endif
        //if (rank == 0)
        //    fprintf(stderr, "a2a iter %d, done status %d\n", a2a_iter_num, all_proc_done);
        a2a_iter_num++;
        if (all_proc_done == nprocs)
            break;
    }

#if TIME_GRAPHCREATE_STEPS
    elt = get_seconds() - elt;
    if (rank == 0) {
        fprintf(stderr, 
        "Creating adj array:           %9.6lf s.\n", elt);
        fprintf(stderr, 
        "Number of A2A iterations:     %d\n", a2a_iter_num);
    }
    elt = get_seconds();
#endif


    for (int64_t i=0; i<((int64_t)n_local); i++) {
        assert((vertex_displs[i] + 3*vertex_degrees[i]) == vertex_displs[i+1]);
    }

#pragma omp parallel for
    for (int64_t i=0; i<(int64_t)n_local; i++) {
        qsort(adj+vertex_displs[i], 
                (vertex_displs[i+1] - vertex_displs[i])/3, 
                6, vid_cmp);
    }

#if TIME_GRAPHCREATE_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Local adj sort:               %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

#if 0
    fprintf(stderr, "before duplicate removal\n");
    for (uint64_t i=0; i<n_local; i++) {
        for (uint64_t j=vertex_displs[i]; j<vertex_displs[i+1]; j+=3) {
            uint64_t v = (((uint64_t)adj[j])<<32) + 
                         (((uint64_t)adj[j+1])<<16) +
                         (uint64_t)adj[j+2];   
            fprintf(stderr, "[%llu %llu] ", i+n_start, v);
        }
    }
#endif

    /* remove duplicate edges in two passes */
#pragma omp parallel for
    for (int64_t i=0; i<(int64_t)n_local; i++) {
        for (uint64_t j=vertex_displs[i]+3; j<vertex_displs[i+1]; j+=3) {
            uint64_t v = (((uint64_t)adj[j])<<32) + 
                         (((uint64_t)adj[j+1])<<16) +
                         (uint64_t)adj[j+2];   
            uint64_t v_prev = (((uint64_t)adj[j-3])<<32) + 
                         (((uint64_t)adj[j-2])<<16) +
                         (uint64_t)adj[j-1]; 
            if (v_prev == i+n_start) {
                 /* Clear out v_prev */
                adj[j-3] = bitmask16;
                adj[j-2] = bitmask16;
                adj[j-1] = bitmask16;
                vertex_degrees[i]--;
            } else if (v == v_prev) {
                /* Clear out v_prev */
                adj[j-3] = bitmask16;
                adj[j-2] = bitmask16;
                adj[j-1] = bitmask16;
                vertex_degrees[i]--;
            }

        }
    }

    /* update vertex_displs */
    vertex_displs[0] = 0;
    for (uint64_t i=1; i<=n_local; i++) {
        vertex_displs[i] = vertex_displs[i-1] + 3 * vertex_degrees[i-1];
    } 
    
    uint64_t curr_epos = 0;
    uint64_t epos_filled = 0;

    while (curr_epos < 3*m_local) {
        if ((adj[curr_epos] == bitmask16) &&
            (adj[curr_epos+1] == bitmask16) &&
            (adj[curr_epos+2] == bitmask16)) {

        } else {
            /* copy curr_epos to epos_filled */
            adj[epos_filled]   = adj[curr_epos];
            adj[epos_filled+1] = adj[curr_epos+1];
            adj[epos_filled+2] = adj[curr_epos+2];
            epos_filled += 3;
        }
        curr_epos += 3;
    }
    assert(vertex_displs[n_local] == epos_filled);
    m_local = epos_filled/3;
    // fprintf(stderr, "rank %d, m_local %lu\n", rank, m_local);

#if TIME_GRAPHCREATE_STEPS           
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "Removing duplicates:          %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

#if USE_MPI
    MPI_Allreduce(&m_local, &m, 1, MPI_UNSIGNED_LONG, 
            MPI_SUM, MPI_COMM_WORLD);
#else
    m = m_local;
#endif

    free(vertex_degrees);
    free(vertex_degrees_local);
    free(vertex_displs_local);

    g->adj = (uint32_t *) adj;
    g->num_edges = vertex_displs;
    g->n_local = n_local;
    g->n = n;
    g->m_local = m_local;
    g->m = m;
    //g->m_inter_part_edges = inter_part_count;

    local_queue_size = n_local;
    g->d = (uint8_t *) _mm_malloc(local_queue_size * sizeof(uint8_t), 16);
    g->queue_current = (uint32_t *) _mm_malloc(n_local * sizeof(uint32_t), 16);
    g->queue_next    = (uint32_t *) _mm_malloc(n_local * sizeof(uint32_t), 16);
    
    assert(g->d != NULL);
    assert(g->queue_current != NULL);
    assert(g->queue_next != NULL);


    (g->comm_data).sendbuf_counts = sendbuf_counts;
    (g->comm_data).sendbuf_displs = sendbuf_displs;
    (g->comm_data).recvbuf_counts = recvbuf_counts;
    (g->comm_data).recvbuf_displs = recvbuf_displs;
    (g->comm_data).adj_sendbuf = a2a_sendbuf;
    (g->comm_data).adj_recvbuf = a2a_recvbuf;

    free(sendrecvbuf_counts);
    free(sendrecvbuf_displs);

    return 0;
}

int free_graph(dist_graph_t* g) {

    _mm_free(g->adj); 
    _mm_free(g->num_edges); 
    
    free(g->d);
    _mm_free(g->queue_current); 
    _mm_free(g->queue_next);

    free((g->comm_data).adj_sendbuf_counts);
    free((g->comm_data).adj_sendbuf_displs);
    free((g->comm_data).adj_recvbuf_counts);
    free((g->comm_data).adj_recvbuf_displs);

    _mm_free((g->comm_data).adj_sendbuf);
    _mm_free((g->comm_data).adj_recvbuf);

#if USE_MPI
    MPI_Comm_free(&(g->comm_data).col_comm);
    MPI_Comm_free(&(g->comm_data).row_comm);
    MPI_Comm_free(&(g->comm_data).replicas_comm);
    MPI_Comm_free(&(g->comm_data).replicas_comm2);


#endif

    free((g->comm_data).sendbuf_counts);
    free((g->comm_data).sendbuf_displs);
    free((g->comm_data).recvbuf_counts);
    free((g->comm_data).recvbuf_displs);

    return 0;
}

