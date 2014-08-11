#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include "omp.h"
#endif
#if USE_MPI
#include "mpi.h"
#endif

#include "graph.h"
#include "RngStream.h"

#define NPROCBINS 1
#define TIME_GRAPHGEN_STEPS 1
#define DEBUG_GRAPHGEN_VERBOSE 0
#define GRAPHGEN_RNG_SEED 2323
#define PERMUTE_VERTICES 1

static int init_graph_gen_aux_data(graph_gen_aux_data_t *ggaux,
        int nprocs, int tid, int nthreads) {
    
    long i;

    if (tid == 0) {

        ggaux->phisto_counts = (uint64_t *) malloc(nprocs * NPROCBINS * nthreads *
                sizeof(uint64_t));
        ggaux->phisto_displs = (uint64_t *) malloc(nprocs * NPROCBINS * nthreads *
                sizeof(uint64_t));
 
        ggaux->phisto_counts_global = (uint64_t *) malloc(nprocs * NPROCBINS * sizeof(uint64_t));


        ggaux->pedge_bin_counts = (int *) malloc(nprocs * nthreads * sizeof(int));
        ggaux->pedge_bin_displs = (int *) malloc(nprocs * nthreads * sizeof(int));


        ggaux->sendbuf_counts = (int *) malloc(nprocs * sizeof(int));
        ggaux->sendbuf_displs = (int *) malloc((nprocs + 1) * sizeof(int));

        ggaux->recvbuf_counts = (int *) malloc(nprocs * sizeof(int));
        ggaux->recvbuf_displs = (int *) malloc((nprocs + 1) * sizeof(int));
 
        ggaux->sendbuf_edges = NULL;
        ggaux->recvbuf_edges = NULL;
        
    }
 
#pragma omp barrier

#pragma omp for
    for (i=0; i<nprocs*nthreads*NPROCBINS; i++) {
        ggaux->phisto_counts[i] = 0;
        ggaux->phisto_displs[i] = 0;
    }

#pragma omp for
    for (i=0; i<nprocs*nthreads; i++) {
        ggaux->pedge_bin_counts[i] = 0;
        ggaux->pedge_bin_displs[i] = 0;
    }

#pragma omp for
    for (i=0; i<nprocs; i++) {
        ggaux->sendbuf_counts[i] = 0;
        ggaux->recvbuf_counts[i] = 0;
        ggaux->sendbuf_displs[i] = 0;
        ggaux->recvbuf_displs[i] = 0;
            
    }
   
    return 0;
}

static int free_graph_gen_data(graph_gen_aux_data_t* ggaux) {

#if 0
    if (ggi->gen_edges) {
        free(ggi->gen_edges);
    }
#endif

    if (ggaux->pedge_bin_counts) {
        free(ggaux->pedge_bin_counts);
    }

    if (ggaux->pedge_bin_displs) {
        free(ggaux->pedge_bin_displs);
    }

    if (ggaux->sendbuf_counts) {
        free(ggaux->sendbuf_counts);
    }

    if (ggaux->sendbuf_displs) {
        free(ggaux->sendbuf_displs);
    }


    if (ggaux->recvbuf_counts) {
        free(ggaux->recvbuf_counts);
    }

    if (ggaux->recvbuf_displs) {
        free(ggaux->recvbuf_displs);
    }


    if (ggaux->sendbuf_edges) {
        free(ggaux->sendbuf_edges);
    }

    
    if (ggaux->recvbuf_edges) {
        free(ggaux->recvbuf_edges);
    }

    if (ggaux->phisto_counts) {
        free(ggaux->phisto_counts);
    }

    if (ggaux->phisto_counts_global) {
        free(ggaux->phisto_counts_global);
    }
 
    if (ggaux->phisto_displs) {
        free(ggaux->phisto_displs);
    }  

    return 0;    
}

static int rng_num_cmp_pos1(const void *a, const void *b) {
    const uint64_t *ai = (uint64_t *) a;
    const uint64_t *bi = (uint64_t *) b;  
    return (ai[1] - bi[1]);
}

static int rng_num_cmp_pos0(const void *a, const void *b) {
    const uint64_t *ai = (uint64_t *) a;
    const uint64_t *bi = (uint64_t *) b;  
    return (ai[0] - bi[0]);
}

static int startv_cmp(const void *a, const void *b) {
    uint64_t ai_startv, ai_startv_off, bi_startv, bi_startv_off;
    
    const uint32_t *ai = (uint32_t *) a;
    const uint32_t *bi = (uint32_t *) b; 

    ai_startv = ai[0];
    ai_startv_off = ai[2];
    ai_startv += ((ai_startv_off>>16)<<32);

    bi_startv = bi[0];
    bi_startv_off = bi[2];
    bi_startv += ((bi_startv_off>>16)<<32);

    return (ai_startv - bi_startv);
}


static int endv_cmp(const void *a, const void *b) {
    uint64_t ai_endv, ai_endv_off, bi_endv, bi_endv_off;
    uint64_t bitmask16;

    const uint32_t *ai = (uint32_t *) a;
    const uint32_t *bi = (uint32_t *) b; 
    bitmask16 = (1UL<<16)-1;

    ai_endv = ai[1];
    ai_endv_off = ai[2];
    ai_endv += ((ai_endv_off & bitmask16)<<32);

    bi_endv = bi[1];
    bi_endv_off = bi[2];
    bi_endv += ((bi_endv_off & bitmask16)<<32);

    return (ai_endv - bi_endv);
}

/* ToDo: replace with binary search */
static int binsearch_uget_start_pos(const uint32_t *A, const uint64_t array_size, 
        const uint64_t val, uint64_t *start_pos_ptr) {

    uint64_t pos;
    uint64_t u, u_off;
    uint64_t init_val;

    init_val = *start_pos_ptr;
    pos = 0;
    for (uint64_t i=init_val; i<array_size; i++) {

        u = A[3*i];
        u_off = A[3*i+2];
        u += ((u_off>>16)<<32);

        if (u >= val) {
            pos = i;
            break;
        }
    }


    *start_pos_ptr = pos;
    return 0;
}

/* ToDo: replace with binary search */
static int binsearch_uget_end_pos(const uint32_t *A, const uint64_t array_size, 
        const uint64_t val, const uint64_t start_pos, uint64_t *end_pos_ptr) {

    uint64_t pos;
    uint64_t u, u_off;
    uint64_t i;

    pos = start_pos;
    for (i=start_pos; i<array_size; i++) {

        u = A[3*i];
        u_off = A[3*i+2];
        u += ((u_off>>16)<<32);

        if (u > val) {
            pos = i;
            break;
        }
    }

    if (i == array_size) {
        pos = array_size;
    }

    *end_pos_ptr = pos;
    return 0;
}

/* ToDo: replace with binary search */
static int binsearch_vget_start_pos(const uint32_t *A, const uint64_t array_size, 
        const uint64_t val, uint64_t *start_pos_ptr) {

    uint64_t pos;
    uint64_t v, v_off;
    uint64_t bitmask16;
    uint64_t init_val;

    bitmask16 = (1UL<<16)-1;
    init_val = *start_pos_ptr;
    pos = 0;
    for (uint64_t i=init_val; i<array_size; i++) {

        v = A[3*i+1];
        v_off = A[3*i+2];
        v += ((v_off & bitmask16)<<32);

        if (v >= val) {
            pos = i;
            break;
        }
    }


    *start_pos_ptr = pos;
    return 0;
}

/* ToDo: replace with binary search */
static int binsearch_vget_end_pos(const uint32_t *A, const uint64_t array_size, 
        const uint64_t val, const uint64_t start_pos, uint64_t *end_pos_ptr) {

    uint64_t pos;
    uint64_t v, v_off;
    uint64_t bitmask16;
    uint64_t i;
    bitmask16 = (1UL<<16)-1;

    pos = start_pos;
    for (i=start_pos; i<array_size; i++) {

        v = A[3*i+1];
        v_off = A[3*i+2];
        v += ((v_off & bitmask16)<<32);

        if (v > val) {
            pos = i;
            break;
        }
    }

    if (i == array_size) {
        pos = array_size;
    }

    *end_pos_ptr = pos;
    return 0;
}

#if 0
static int print_edges(uint32_t *gen_edges, uint64_t m) {

    uint64_t i, u, v, uv_off, bitmask16;
    bitmask16 = (1UL<<16)-1;

    fprintf(stderr, "\n\n");
    for (i=0; i<m; i++) {
        u = gen_edges[3*i];
        v = gen_edges[3*i+1];
        uv_off = gen_edges[3*i+2];
        u += ((uv_off>>16)<<32);
        v += ((uv_off & bitmask16)<<32);
#ifdef __x86_64__
        fprintf(stderr, "(%lu %lu) ", u, v);
#else
        fprintf(stderr, "(%llu %llu) ", u, v);
#endif
    }
    fprintf(stderr, "\n\n");
    return 0;
}
#endif

static int parallel_local_sort(uint64_t** perm_ptr, int sortvpos, uint64_t perm_recvbuf_size) {

    if (sortvpos == 0) 
        local_parallel_quicksort(*perm_ptr, perm_recvbuf_size, 
        //qsort(*perm_ptr, perm_recvbuf_size, 
                2*sizeof(uint64_t), rng_num_cmp_pos0);
    else    
        local_parallel_quicksort(*perm_ptr, perm_recvbuf_size, 
        //qsort(*perm_ptr, perm_recvbuf_size, 
                2*sizeof(uint64_t), rng_num_cmp_pos1);
    return 0;
}

static int parallel_intpair_sort(graph_gen_aux_data_t* ggaux, 
        uint64_t init_array_size, uint64_t maxval, int sortvpos,
        uint64_t **perm_ptr, uint64_t* perm_size_ptr) {

    uint64_t proc_bucket_size;
    uint64_t *perm;
    perm = *perm_ptr;

    proc_bucket_size = maxval/nprocs;
    if (maxval % nprocs != 0) {
        proc_bucket_size++;
    }

    assert(proc_bucket_size > 0);

#if TIME_GRAPHGEN_STEPS
    double elt;
    elt = get_seconds();
#endif

#pragma omp parallel
{    

    uint64_t *phisto_counts;
    int tid, nthreads;

#ifdef _OPENMP
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else
    tid = 0;
    nthreads = 1;
#endif

    /*
    if (tid == 0) {
        fprintf(stderr, "rank %d, init array size %d, nprocs %d\n", rank,
                init_array_size, nprocs);
    }
    */

    phisto_counts = ggaux->phisto_counts + nprocs*NPROCBINS*tid;

    for (int i=0; i<nprocs; i++) {
        phisto_counts[i] = 0;
    }

#pragma omp barrier

#pragma omp for 
    for (uint64_t i=0; i<init_array_size; i++) {
        uint64_t proc_bin_id = perm[2*i+sortvpos]/proc_bucket_size;
        phisto_counts[proc_bin_id]++;
    }

#pragma omp for 
    for (int i=0; i<nprocs; i++) {
        uint64_t cumulative_proc_bin_count = 0;
        for (int j=0; j<nthreads; j++) {
            cumulative_proc_bin_count += ggaux->phisto_counts[j*nprocs*NPROCBINS + i];
        }
        ggaux->phisto_counts_global[i] = cumulative_proc_bin_count;
    }

} /* end of OpenMP parallel region */

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "binning time:                 %9.6lf s.\n", elt);
    
    elt = get_seconds();
#endif

    /*
    for (int i=0; i<nprocs; i++) {
        fprintf(stderr, "[%d %lu %lu] ", rank, i, ggaux->phisto_counts_global[i]);
    }
    */

#if USE_MPI 
#if TIME_GRAPHGEN_STEPS
    double elt2 = get_seconds();
#endif

    /* update sendbuf_counts, recvbuf_counts, ... */
    for (int i=0; i<nprocs; i++) {
        ggaux->sendbuf_counts[i] = 2 * ggaux->phisto_counts_global[i];
    }

    ggaux->sendbuf_displs[0] = 0;
    for (int i=1; i<nprocs; i++) {
        ggaux->sendbuf_displs[i] = ggaux->sendbuf_counts[i-1] 
                                 + ggaux->sendbuf_displs[i-1];
    }

    if  (ggaux->sendbuf_displs[nprocs-1] + ggaux->sendbuf_counts[nprocs-1] !=
            2UL*init_array_size) {
        fprintf(stderr, "displs %d, counts %d, array size %lu\n",
                ggaux->sendbuf_displs[nprocs-1], ggaux->sendbuf_counts[nprocs-1],
                2UL*init_array_size);

        assert(ggaux->sendbuf_displs[nprocs-1] + ggaux->sendbuf_counts[nprocs-1] ==
            2UL*init_array_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if TIME_GRAPHGEN_STEPS
    elt2 = get_seconds() - elt2;
    if (rank == 0)
        fprintf(stderr, 
        "barrier time:                 %9.6lf s.\n", elt2);
                
#endif

    /* Pack up perm_sendbuf */
    ggaux->perm_sendbuf = (uint64_t *) malloc(2*init_array_size*sizeof(uint64_t));
    assert(ggaux->perm_sendbuf != NULL);

#pragma omp parallel for
    for (uint64_t i=0; i<2*init_array_size; i++) {
        ggaux->perm_sendbuf[i] = 0;
    }

    for (int i=0; i<nprocs; i++) {
        ggaux->sendbuf_counts[i] = 0;
    }

    for (int i=0; i<init_array_size; i++) {
        uint64_t proc_bin_id = perm[2*i+sortvpos]/proc_bucket_size;
        int pos = ggaux->sendbuf_displs[proc_bin_id] + ggaux->sendbuf_counts[proc_bin_id];
        ggaux->sendbuf_counts[proc_bin_id] += 2;
        ggaux->perm_sendbuf[pos]   = perm[2*i];
        ggaux->perm_sendbuf[pos+1] = perm[2*i+1];
    }

    free(perm);

    MPI_Barrier(MPI_COMM_WORLD);

#if TIME_GRAPHGEN_STEPS
    elt2 = get_seconds();
#endif

    /* Get recvbuf counts */ 
    MPI_Alltoall(ggaux->sendbuf_counts, 1, MPI_INT, 
        ggaux->recvbuf_counts, 1, MPI_INT,
        MPI_COMM_WORLD);

#if TIME_GRAPHGEN_STEPS
    elt2 = get_seconds() - elt2;
    if (rank == 0)
        fprintf(stderr, 
        "alltoall time:                %9.6lf s.\n", elt2);
#endif

    ggaux->recvbuf_displs[0] = 0;

    uint64_t perm_recvbuf_size = ggaux->recvbuf_counts[0];
    for (int i=1; i<nprocs; i++) {
        perm_recvbuf_size += ggaux->recvbuf_counts[i];
        ggaux->recvbuf_displs[i] = ggaux->recvbuf_displs[i-1] + ggaux->recvbuf_counts[i-1];
    }

    assert(perm_recvbuf_size < ((1UL<<31)-1));

    ggaux->perm_recvbuf = (uint64_t *) malloc(perm_recvbuf_size * sizeof(uint64_t));
    assert(ggaux->perm_recvbuf != NULL);

#pragma omp parallel for
    for (uint64_t i=0; i<perm_recvbuf_size; i++) {
        ggaux->perm_recvbuf[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if TIME_GRAPHGEN_STEPS
    elt2 = get_seconds();
#endif

    MPI_Alltoallv(ggaux->perm_sendbuf, ggaux->sendbuf_counts, 
        ggaux->sendbuf_displs, 
        MPI_UNSIGNED_LONG, ggaux->perm_recvbuf, 
        ggaux->recvbuf_counts, ggaux->recvbuf_displs,
        MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

#if TIME_GRAPHGEN_STEPS
    elt2 = get_seconds() - elt2;
    if (rank == 0)
        fprintf(stderr, 
        "alltoallv time:               %9.6lf s.\n", elt2);
#endif

    free(ggaux->perm_sendbuf);
    perm = ggaux->perm_recvbuf;

    ggaux->perm_sendbuf = NULL;
    ggaux->perm_recvbuf = NULL;

    /* Find offset */
    uint64_t perm_n_offset = 0;
    ggaux->sendbuf_counts[rank] = perm_recvbuf_size;
    MPI_Allgather(ggaux->sendbuf_counts+rank, 1, MPI_INT, 
            ggaux->recvbuf_counts, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i=0; i<rank; i++) {
        perm_n_offset += ggaux->recvbuf_counts[i];
    }

    perm_n_offset = perm_n_offset/2;
   
    /*
    if (sortvpos==1) 
        fprintf(stderr, "rank %d, count %lu, perm_offset %lu\n", rank, 
            perm_recvbuf_size, perm_n_offset);
    */

#else
    uint64_t perm_recvbuf_size = 2*init_array_size;
    uint64_t perm_n_offset = 0;
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "parallel exchange time:       %9.6lf s.\n", elt);
    elt = get_seconds();
#endif


    /* Sort perm_recvbuf */
    parallel_local_sort(&perm, sortvpos, perm_recvbuf_size/2); 
  
#if TIME_GRAPHGEN_STEPS 
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "local sort time:              %9.6lf s.\n", elt);
    elt = get_seconds();
#endif

    if (sortvpos == 1) {
#pragma omp parallel for
        for (uint64_t i=0; i<perm_recvbuf_size/2; i++) {
            /*
            if (i != 0)
                assert(perm[2*i+sortvpos] >= perm[2*(i-1)+sortvpos]);
            */
            perm[2*i+1] = perm_n_offset + i;
        }

#if TIME_GRAPHGEN_STEPS
        elt = get_seconds() - elt;
        if (rank == 0)
            fprintf(stderr, 
        "perm reload time:             %9.6lf s.\n", elt);
#endif
    }

    *perm_size_ptr = perm_recvbuf_size;
    *perm_ptr = perm;

    return 0;
}

static int apply_relabeling_startv(uint32_t *gen_edges, uint64_t *perm, uint64_t
        perm_array_size, uint64_t *recv_perm_buffer, graph_gen_data_t *ggi) {

    uint64_t m_local;
    uint64_t bitmask16, bitmask32;
#if TIME_GRAPHGEN_STEPS
    double elt;
#endif

    m_local = ggi->m_local;
    bitmask16 = (1UL<<16)-1;
#ifdef __x86_64__
    bitmask32 = (1UL<<32)-1;
#else
    bitmask32 = (1ULL<<32)-1;
#endif

    /* Sort edges by start ID */
#if TIME_GRAPHGEN_STEPS
    elt = get_seconds();
#endif

    //qsort(gen_edges, m_local, 3*sizeof(uint32_t), startv_cmp);
    local_parallel_quicksort(gen_edges, m_local, 3*sizeof(uint32_t), startv_cmp);

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0) { 
        fprintf(stderr, "edges after startv sort\n");
        print_edges(gen_edges, m_local);
    }
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;

    if (rank == 0)
        fprintf(stderr, 
        "local sort1 time:             %9.6lf s.\n", elt);

    elt = get_seconds();
#endif

    uint64_t end_pos_genedges_prev = 0;

    for (int i=0; i<nprocs; i++) {
       
        uint64_t *perm_buffer;
        uint64_t perm_buffer_size;
        uint64_t start_pos_genedges, end_pos_genedges;
        uint64_t curr_perm_buffer_pos;
        
        start_pos_genedges = end_pos_genedges_prev;

        if (rank == i) {
            perm_buffer = perm;
            perm_buffer_size = perm_array_size;
        } else {
            perm_buffer = recv_perm_buffer;
        }

#if USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&perm_buffer_size, 1, MPI_UNSIGNED_LONG, i, MPI_COMM_WORLD);
        MPI_Bcast(perm_buffer, perm_buffer_size, MPI_UNSIGNED_LONG, i, MPI_COMM_WORLD);
#endif

        /* Apply permutation */
        binsearch_uget_start_pos(gen_edges, m_local, perm_buffer[2*0], 
                &start_pos_genedges);
        binsearch_uget_end_pos(gen_edges, m_local, 
                perm_buffer[2*(perm_buffer_size/2-1)], 
                start_pos_genedges, &end_pos_genedges);
     
        /* 
        fprintf(stderr, "i %d, rank %d, vals %lu %lu, start %lu, end %lu\n", i, rank,
                perm_buffer[0], perm_buffer[2*(perm_buffer_size/2-1)], 
                start_pos_genedges, end_pos_genedges);
        */
        end_pos_genedges_prev = end_pos_genedges;

        curr_perm_buffer_pos = 0;
#pragma omp parallel for firstprivate(curr_perm_buffer_pos) schedule(static)
        for (uint64_t j=start_pos_genedges; j<end_pos_genedges; j++) {
        
            uint64_t k, u, v, uv_off, u_new;
            uint64_t val_not_found;

            val_not_found = 0;

            u = gen_edges[3*j];
            v = gen_edges[3*j+1];
            uv_off = gen_edges[3*j+2];

            u += ((uv_off>>16)<<32);
            v += ((uv_off & bitmask16)<<32);

            if (u != perm_buffer[2*curr_perm_buffer_pos]) {
                for (k=curr_perm_buffer_pos; k<perm_buffer_size/2; k++) {
                    if (perm_buffer[2*k] == u) {
                        curr_perm_buffer_pos = k;
                        break;
                    }
                    if (perm_buffer[2*k] > u) {
                        val_not_found = 1;
                        break;
                    }
                }
                if ((val_not_found == 1) || (k == perm_buffer_size/2)) {
                    fprintf(stderr, "error in startv relabeling\n");
                    continue;
                }
            }

            assert(u == perm_buffer[2*curr_perm_buffer_pos]);
            
            u_new = perm_buffer[2*curr_perm_buffer_pos+1];

            gen_edges[3*j]   = u_new & bitmask32;
            gen_edges[3*j+2] = ((u_new>>32)<<16)+(v>>32);

        } 

    }

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0) { 
        fprintf(stderr, "edges after permute\n");
        print_edges(gen_edges, m_local);
    }
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "startv relabel time:          %9.6lf s.\n", elt);
#endif
    return 0;
}

static int apply_relabeling_endv(uint32_t *gen_edges, uint64_t *perm, uint64_t
        perm_array_size, uint64_t *recv_perm_buffer, graph_gen_data_t *ggi) {

    uint64_t m_local;
    uint64_t bitmask16, bitmask32;

    m_local = ggi->m_local;
    bitmask16 = (1UL<<16)-1;
#ifdef __x86_64__
    bitmask32 = (1UL<<32)-1;
#else
    bitmask32 = (1ULL<<32)-1;
#endif

    /* Sort edges by end ID */
#if TIME_GRAPHGEN_STEPS
    double elt;
    elt = get_seconds();
#endif

    //qsort(gen_edges, m_local, 3*sizeof(uint32_t), endv_cmp);
    local_parallel_quicksort(gen_edges, m_local, 3*sizeof(uint32_t), endv_cmp);

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0) { 
        fprintf(stderr, "edges after endv sort\n");
        print_edges(gen_edges, m_local);
    }
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "local sort2 time:             %9.6lf s.\n", elt);

    elt = get_seconds();
#endif
    uint64_t end_pos_genedges_prev = 0;

    for (int i=0; i<nprocs; i++) {
       
        uint64_t *perm_buffer;
        uint64_t perm_buffer_size;
        uint64_t start_pos_genedges, end_pos_genedges;
        uint64_t curr_perm_buffer_pos;
        
        start_pos_genedges = end_pos_genedges_prev;

        if (rank == i) {
            perm_buffer = perm;
            perm_buffer_size = perm_array_size;
        } else {
            perm_buffer = recv_perm_buffer;
        }

#if USE_MPI
        MPI_Bcast(&perm_buffer_size, 1, MPI_UNSIGNED_LONG, i, MPI_COMM_WORLD);
        MPI_Bcast(perm_buffer, perm_buffer_size, MPI_UNSIGNED_LONG, i, MPI_COMM_WORLD);
#endif

        /* Apply permutation */
        binsearch_vget_start_pos(gen_edges, m_local, perm_buffer[2*0], &start_pos_genedges);
        binsearch_vget_end_pos(gen_edges, m_local, perm_buffer[2*(perm_buffer_size/2-1)], 
                                    start_pos_genedges, &end_pos_genedges);
      
        end_pos_genedges_prev = end_pos_genedges;

        curr_perm_buffer_pos = 0;
#pragma omp parallel for firstprivate(curr_perm_buffer_pos) schedule(static)
        for (uint64_t j=start_pos_genedges; j<end_pos_genedges; j++) {
        
            uint64_t k, u, v, uv_off, v_new;
            uint64_t val_not_found;

            val_not_found = 0;

            u = gen_edges[3*j];
            v = gen_edges[3*j+1];
            uv_off = gen_edges[3*j+2];

            u += ((uv_off>>16)<<32);
            v += ((uv_off & bitmask16)<<32);

            if (v != perm_buffer[2*curr_perm_buffer_pos]) {
                for (k=curr_perm_buffer_pos; k<perm_buffer_size/2; k++) {
                    if (perm_buffer[2*k] == v) {
                        curr_perm_buffer_pos = k;
                        break;
                    }
                    if (perm_buffer[2*k] > v) {
                        val_not_found = 1;
                        break;
                    }
                }
                if ((val_not_found == 1) || (k == perm_buffer_size/2)) {
                    fprintf(stderr, "error in endv relabeling\n");
                    continue;
                }
            }

            assert(v == perm_buffer[2*curr_perm_buffer_pos]);
            
            v_new = perm_buffer[2*curr_perm_buffer_pos+1];

            gen_edges[3*j+1] = v_new & bitmask32;
            gen_edges[3*j+2] = ((u>>32)<<16)+(v_new>>32);

        } 

    }

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "endv relabel time:            %9.6lf s.\n", elt);
#endif

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0) { 
        fprintf(stderr, "edges after permute\n");
        print_edges(gen_edges, m_local);
    }
#endif

    return 0;
}

static int graph_gen_rmat(graph_gen_data_t* ggi, 
        graph_gen_aux_data_t* ggaux) {

    uint64_t *perm, *recv_perm_buffer;
    uint64_t perm_array_size, perm_array_size1;
             
#if USE_MPI
    uint64_t max_perm_array_size;
#endif

#ifdef _OPENMP
#pragma omp parallel
{
#endif

    uint64_t n, m, m_local, n_local, n_start;
    uint64_t i, u, v, step;
    int SCALE;

    int tid, nthreads;
    int nstreams_per_thread;
    RngStream rs1, rs2;
    double *rng_seed;
    double p, S, S_inv, a, b, c, d;
    double av1, bv1, cv1, dv1, var, var2;

    uint32_t *gen_edges;
    uint64_t bitmask32;


#ifdef _OPENMP
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else    
    tid = 0;
    nthreads = 1;
#endif
  
    bitmask32 = (1ULL<<32) - 1;

    nstreams_per_thread = 3; /* two for use in this routine */
    rng_seed = (double *) malloc(6 * sizeof(double));
    assert(rng_seed != NULL);
    rng_seed[0] = 12345; rng_seed[1] = 12345; rng_seed[2] = 12345;
    rng_seed[3] = 12345; rng_seed[4] = 12345; rng_seed[5] = 12345;

    /* RMAT gen parameters */
    //a = 0.45; b = 0.15; c = 0.15; d = 0.25;
    //a = 0.25; b = 0.25; c = 0.25; d = 0.25;
    a = 0.57; b = 0.19; c = 0.19; d = 0.05;
    var = 0.1;
    var2 = 1.0 - var/2.0;

    RngStream_ParInit(rank, nprocs, tid, nthreads, nstreams_per_thread, rng_seed);
    rs1 = RngStream_CreateStream("", rng_seed);
    rs2 = RngStream_CreateStream("", rng_seed);

    init_graph_gen_aux_data(ggaux, nprocs, tid, nthreads);

    n = ggi->n;
    m = ggi->m;
    m_local = m/nprocs;
    ggi->m_local = m_local;
    /* 15% additional space buffer */
    if (nprocs > 1) 
        ggi->m_local_allocsize = (m_local * 115UL)/100UL;
    else
        ggi->m_local_allocsize = m_local;

    ggi->m = m = m_local * nprocs;

    SCALE = ggi->SCALE;

    if (tid == 0) {

        if (rank == 0) {
            fprintf(stderr, "%d MPI tasks, %d OpenMP threads\n", nprocs,
                    nthreads);
#ifdef __x86_64__
            fprintf(stderr, "SCALE %d, n: %lu, m: %lu, m_local: %lu, graph gen memory: "
                    "%3.6lf GB\n", 
                    SCALE, n, m, m_local,
                    (12.0*ggi->m_local_allocsize+(32.0*n/nprocs))/1073741824.0);
#else
            fprintf(stderr, "SCALE %d, n: %llu, m: %llu, m_local: %llu, graph"
                    " gen memory: "
                    "%3.6lf GB\n", 
                    SCALE, n, m,
                    m_local,(12.0*ggi->m_local_allocsize+(32.0*n/nprocs))/1073741824.0);
#endif
        }

        ggi->gen_edges = (uint32_t *) malloc(3 * ggi->m_local_allocsize * sizeof(uint32_t));
        assert(ggi->gen_edges != NULL);

    }
   
#pragma omp barrier

    gen_edges = ggi->gen_edges;

#pragma omp for schedule(static)
    for (i=0; i<m_local; i++) {
        gen_edges[3*i] = 0;
        gen_edges[3*i+1] = 0;
        gen_edges[3*i+2] = 0;
    }

#if TIME_GRAPHGEN_STEPS
    double elt;
    elt = get_seconds();
#endif

#pragma omp for
    for (i=0; i<m_local; i++) {
        u = 0; v = 0;
 
        step = n/2;

        av1 = a;
        bv1 = b;
        cv1 = c;
        dv1 = d;

        p = RngStream_RandU01(rs1);
           
        if (p < av1) {
            /* Do nothing */
        } else if ((p >= av1) && (p < (av1+bv1))) {
            v += step;
        } else if ((p >= (av1+bv1)) && (p < (av1+bv1+cv1))) {
            u += step;
        } else {
            u += step;
            v += step;
        }
        
        for (int j=1; j<SCALE; j++) {
            step = step/2;

            /* Vary a,b,c,d by up to 10% */
                
            av1 *= (var2 + var * RngStream_RandU01(rs2));
            bv1 *= (var2 + var * RngStream_RandU01(rs2));
            cv1 *= (var2 + var * RngStream_RandU01(rs2));
            dv1 *= (var2 + var * RngStream_RandU01(rs2));

            S = av1 + bv1 + cv1 + dv1;
            S_inv = 1.0/S;
            av1 = av1*S_inv;
            bv1 = bv1*S_inv;
            cv1 = cv1*S_inv;
            dv1 = dv1*S_inv;
                

            /* Choose partition */
            p = RngStream_RandU01(rs1);


            if (p < av1) {
                /* Do nothing */
            } else if ((p >= av1) && (p < (av1+bv1))) {
                v += step;
            } else if ((p >= (av1+bv1)) && (p < (av1+bv1+cv1))) {
                u += step;
            } else {
                u += step;
                v += step;
            }
        }
       

        gen_edges[3*i]   = u & bitmask32;
        gen_edges[3*i+1] = v & bitmask32;
        gen_edges[3*i+2] = ((u>>32)<<16)+(v>>32);
    }
   
#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    
    if (tid == 0)
        fprintf(stderr, 
        "rank %d, edge gen time:       %9.6lf s.\n", rank, elt);
#endif

#pragma omp barrier
   
#if TIME_GRAPHGEN_STEPS 
    elt = get_seconds();
#endif

#if PERMUTE_VERTICES
    /* Next, generate a permutation of vertex IDs */
    n_local = n/nprocs;
    n_start = n_local * rank;
    if (rank == nprocs - 1)
        n_local = n - n_start;

    if (tid == 0) {
        ggi->n_local = n_local;
        ggi->n_start = n_start;
        perm = (uint64_t *) malloc(2 * n_local * sizeof(uint64_t));
        assert(perm != NULL);
    }

#pragma omp barrier

#if DEBUG_GRAPHGEN_VERBOSE
    fprintf(stderr, "\npermutation values:\n");
#endif


    //srandom(22323*rank+1);
#pragma omp for
    for (i=0; i<n_local; i++) {
        perm[2*i] = n_start + i;
        //perm[2*i+1] = (n_start + i);
        //perm[2*i+1] = (n_start + n_local - 1 - i);
        //perm[2*i+1] = 8UL * (random() % n);
        perm[2*i+1] = (ggi->n) * RngStream_RandU01(rs1);
#if DEBUG_GRAPHGEN_VERBOSE
#if __x86_64__
        fprintf(stderr, "%lu %lu\n", perm[2*i], perm[2*i+1]);
#else
        fprintf(stderr, "%llu %llu\n", perm[2*i], perm[2*i+1]);
#endif
#endif
    }

#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if ((rank == 0) && (tid == 0))
        fprintf(stderr, 
        "perm init time:               %9.6lf s.\n", elt);
#endif

#pragma omp barrier

    RngStream_DeleteStream(rs1);
    RngStream_DeleteStream(rs2);
    free(rng_seed);
#pragma omp barrier

#ifdef _OPENMP
}
#endif

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0)
        print_edges(ggi->gen_edges, ggi->m_local);
#endif

#if TIME_GRAPHGEN_STEPS
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double elt;
    elt = get_seconds();
#endif

#if PERMUTE_VERTICES

    /* Sort perm by the randomly generated value, in parallel */
    parallel_intpair_sort(ggaux, ggi->n_local, ggi->n, 1, &perm,
            &perm_array_size1);

#if DEBUG_GRAPHGEN_VERBOSE
    fprintf(stderr, "\npermutation values after first sort:\n");


    for (int i=0; i<perm_array_size1/2; i++) {
#if __x86_64__
        fprintf(stderr, "%lu %lu\n", perm[2*i], perm[2*i+1]);
#else
        fprintf(stderr, "%llu %llu\n", perm[2*i], perm[2*i+1]);
#endif
    }
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "perm sort1 time:              %9.6lf s.\n", elt);

    elt = get_seconds();
#endif

    /* 
    fprintf(stderr, "rank %d, perm_array_size1 %lu\n", rank, perm_array_size1); 
    fprintf(stderr, "rank %d, perm_array_size  %lu\n", rank, perm_array_size);
    
    for (uint64_t i=0; i<perm_array_size1/2; i++) {
        assert(perm[2*i] < ggi->n);
    }
    */

    /* Sort again by vertex ID */
    parallel_intpair_sort(ggaux, perm_array_size1/2, ggi->n, 0, &perm,
            &perm_array_size);

    /*
    for (uint64_t i=0; i<perm_array_size/2; i++) {
        perm[2*i+1] = perm[2*i];
    }
    */

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == nprocs/2) {
        fprintf(stderr, "\npermutation values after second  sort:\n");
        for (int i=0; i<perm_array_size/2; i++) {
#if __x86_64__
            fprintf(stderr, "%lu %lu\n", perm[2*i], perm[2*i+1]);
#else
            fprintf(stderr, "%llu %llu\n", perm[2*i], perm[2*i+1]);
#endif
        }
    }
#endif

#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "perm sort2 time:              %9.6lf s.\n", elt);
#endif

    /* get max local permutation array size */
#if USE_MPI  
    MPI_Allreduce(&perm_array_size, &max_perm_array_size, 1, MPI_UNSIGNED_LONG, 
            MPI_MAX, MPI_COMM_WORLD);

    recv_perm_buffer = (uint64_t *) malloc(max_perm_array_size * sizeof(uint64_t));
    assert(recv_perm_buffer != NULL);
#else
    //max_perm_array_size = perm_array_size;
    recv_perm_buffer = NULL;
#endif

    /* Permute start vertices */
    apply_relabeling_startv(ggi->gen_edges, perm, perm_array_size, recv_perm_buffer, 
            ggi);

    /* Permute end vertices */
    apply_relabeling_endv(ggi->gen_edges, perm, perm_array_size, recv_perm_buffer, 
            ggi);

    free(perm);   
#if USE_MPI
    free(recv_perm_buffer);
#endif

#endif /* end of permute vertices */

    /* Shuffle edges */
    /* Global shuffle not necessary, as the R-MAT generator uses i
       global vertex IDs to begin with, 
       and edges haven't been reordered globally.
     */
#if TIME_GRAPHGEN_STEPS
    elt = get_seconds();
#endif
#if TIME_GRAPHGEN_STEPS
    elt = get_seconds() - elt;
    if (rank == 0)
        fprintf(stderr, 
        "shuffle edges time:           %9.6lf s.\n", elt);
#endif

#if DEBUG_GRAPHGEN_VERBOSE
    if (rank == 0) 
        print_edges(ggi->gen_edges, ggi->m_local);
#endif
#if 0
    int sz = ggi->m_local;
    float *Data = (float *) malloc(sz * sizeof(float));
    assert(Data != NULL);
    for (int i=0; i<sz; i++) {
        Data[i] = 1.1 * rand() * 5000 / RAND_MAX;
    }
    
    elt = get_seconds();
    local_parallel_quicksort(Data, sz, sizeof(float), floatcompare);
    elt = get_seconds() - elt;
    fprintf(stderr, "parallel quicksort time: %9.6lf\n", elt);

    for (int i=1; i<sz; i++) {
        assert(Data[i] >= Data[i-1]);
    }
 
    for (int i=0; i<sz; i++) {
        Data[i] = 1.1 * rand() * 5000 / RAND_MAX;
    }
 
    elt = get_seconds();
    qsort(Data, sz, sizeof(float), floatcompare);
    elt = get_seconds() - elt;
    fprintf(stderr, "qsort time: %9.6lf\n", elt);

    for (int i=1; i<sz; i++) {
        assert(Data[i] >= Data[i-1]);
    }
 
    free(Data);

    int npart = 4;
    uint64_t part_size = (ggi->n)/npart;
    if (ggi->n % npart != 0) 
        part_size++;
    fprintf(stderr, "part size %d\n", part_size);
    int *part_hist = (int *) calloc(npart, sizeof(int));
    uint64_t bitmask16 = (1UL<<16)-1;
    for (uint64_t i=0; i<ggi->m_local; i++) {
        uint64_t u = ggi->gen_edges[3*i];
        uint64_t v = ggi->gen_edges[3*i+1];
        uint64_t uv_off = ggi->gen_edges[3*i+2];

        u += ((uv_off>>16)<<32);
        v += ((uv_off & bitmask16)<<32);
        part_hist[u/part_size]++;
        part_hist[v/part_size]++;
    }

    for (int i=0; i<npart; i++) {
        fprintf(stderr, "rank %d part %d, size %d\n", rank, i, part_hist[i]);
    }
#endif

 
    return 0;
}

int gen_graph_edges(graph_gen_data_t *ggi, graph_gen_aux_data_t *ggaux) {

    graph_gen_rmat(ggi, ggaux);
    free_graph_gen_data(ggaux);
    return 0;
}

