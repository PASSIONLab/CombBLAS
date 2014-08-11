#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#if USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "graph.h"

#if PRED_CACHE_BYPASS
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

int run_bfs_2Dgraph_threaded(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited) {
 
    uint64_t next_vert_count;
    uint64_t current_vert_count; 
    uint64_t global_queue_vert_count;
    uint32_t *queue_current, *queue_next, *tmp_queue_ptr;
    uint64_t vis_level;
    uint64_t pred_count;

    pred_count = 0;

#pragma omp parallel
{

    uint64_t pred_vert_pair_count;
    const uint32_t* restrict adj;
    const uint32_t* restrict num_edges;
    uint64_t n_local, n_local_col, n_local_row;
    uint8_t* d;

    uint64_t pred_local[LOCAL_QUEUE_SIZE]
        __attribute__ ((aligned (16)));
        
#if USE_MPI
    int32_t* restrict recvbuf_displs;
    uint32_t *adj_sendbuf, *adj_recvbuf;
#endif
    int32_t *sendbuf_counts, *recvbuf_counts;
    
    uint32_t queue_next_local[LOCAL_QUEUE_SIZE]
        __attribute__ ((aligned (16)));
   
#if USE_MPI 
    uint32_t adj_sendbuf_local[LOCAL_SENDBUF_SIZE*MAX_NUMPROCS]
        __attribute__ ((aligned (16)));
    uint32_t adj_sendbuf_counts_local[MAX_NUMPROCS]
        __attribute__ ((aligned (16)));
    assert(g->nproc_rows <= MAX_NUMPROCS);
#endif

    int32_t *adj_sendbuf_counts, *adj_sendbuf_displs;
    int32_t *adj_recvbuf_counts, *adj_recvbuf_displs;
    
    int irow, jcol, rank_rep;
    int nproc_rows, nproc_cols;

    uint64_t bitmask63;

    int tid;

#if TIME_BFS_SUBROUTINES
    double elt, full_elt, init_elt, mpi_elt, 
           sift_elt, sift_elt_part, load_elt, trans_elt, trans_elt_part, load_elt_part, mpi_elt_part;
    elt = init_elt = mpi_elt = load_elt = sift_elt = 
        full_elt = load_elt_part = sift_elt_part = trans_elt = trans_elt_part = mpi_elt_part = 0.0;
#endif

#if __x86_64__
    bitmask63 = (1UL<<63)-1; 
#else
    bitmask63 = (1ULL<<63)-1;
#endif

#ifdef _OPENMP 
    tid = omp_get_thread_num();
#else
    tid = 0;
#endif

#pragma omp barrier

#if TIME_BFS_SUBROUTINES
    if ((rank == 0) && (tid == 0))
        full_elt = elt = get_seconds();
#endif

    pred_vert_pair_count = 0;


    n_local = g->n_local;
    n_local_col = g->n_local_col;
    n_local_row = g->n_local_row;
    nproc_rows = g->nproc_rows;
    nproc_cols = g->nproc_cols;
    d = g->d;

    sendbuf_counts = (g->comm_data).sendbuf_counts;
    recvbuf_counts = (g->comm_data).recvbuf_counts;
    //sendbuf_displs = (g->comm_data).sendbuf_displs;
    recvbuf_displs = (g->comm_data).recvbuf_displs;
    adj_sendbuf    = (g->comm_data).adj_sendbuf;
    adj_recvbuf    = (g->comm_data).adj_recvbuf;
    adj_sendbuf_counts = (g->comm_data).adj_sendbuf_counts;
    adj_sendbuf_displs = (g->comm_data).adj_sendbuf_displs;
    adj_recvbuf_counts = (g->comm_data).adj_recvbuf_counts;
    adj_recvbuf_displs = (g->comm_data).adj_recvbuf_displs;
    
    adj = g->adj;
    num_edges = g->num_edges;
    
    irow = (g->comm_data).irow; jcol = (g->comm_data).jcol; 
    rank_rep = (rank % (nproc_rows * nproc_cols));
    int num_a2a_iterations = 0;

    if (tid == 0) {
        queue_current = g->queue_current;
        queue_next = g->queue_next;
        *nvisited = 0;

        for (int i=0; i<nprocs; i++) {
            sendbuf_counts[i] = 0;
            recvbuf_counts[i] = 0;
    
        }
    }
   
#if USE_MPI 
    for (int i=0; i<nproc_rows; i++) {
        adj_sendbuf_counts_local[i] = 0;
    }
#endif

    int next_vert_count_local = 0;

#pragma omp barrier


#if REPLICATE_D
#pragma omp for
    for (int64_t i=0; i<n_local_col; i++) {
        d[i] = 0;
    }
#else
#pragma omp for
    for (int64_t i=0; i<n_local; i++) {
        d[i] = 0;
    }
#endif

    uint32_t src_owner_proc_row = root/n_local_row;
    
    if (tid == 0) {
        if (src_owner_proc_row == irow) {

            queue_current[0] = root;

            current_vert_count = 1;

        } else {

            current_vert_count = 0;
    
        }
    }

    uint32_t src_owner_proc = root/n_local;

    if (tid == 0) {
        if (src_owner_proc == rank_rep) {
#if STORE_PRED
            pred_local[pred_vert_pair_count]   = root % n_local_col;
            pred_local[pred_vert_pair_count+1] = bitmask63;
            pred_vert_pair_count = 2;
#endif
        }
    }

#if REPLICATE_D
    uint32_t src_owner_proc_col = root/n_local_col;
    if (tid == 0) {
        if (src_owner_proc_col == jcol) {
            uint64_t local_src_val = root % n_local_col;
            d[local_src_val] = 1;
        }
    }
#else
    if (tid == 0) {
        if (src_owner_proc == rank_rep) {
            uint64_t local_src_val = root % n_local;
            d[local_src_val] = 1;
            fprintf(stderr, "local src val %lu\n", local_src_val);
        }
    }
#endif

    if (tid == 0) {
        next_vert_count = 0;
        global_queue_vert_count = 1;
        vis_level = 2;
    }

#pragma omp barrier

#if TIME_BFS_SUBROUTINES
    if ((rank == 0) && (tid == 0)) {
        elt = get_seconds() - elt;
        init_elt = elt;
    }
#endif

    while (global_queue_vert_count > 0) {


        if (tid == 0) {
            for (int i=0; i<nproc_rows; i++) {
                adj_sendbuf_counts[i] = 0;
                adj_recvbuf_counts[i] = 0;
            }
        }

#if TIME_BFS_SUBROUTINES
        if ((rank == 0) && (tid == 0)) {
            load_elt_part = get_seconds();
        }
#endif

#pragma omp barrier

#pragma omp for schedule(guided) nowait
        for (uint64_t i=0; i<current_vert_count; i++) {

            uint64_t u = (queue_current[i] % n_local_row);
            uint64_t num_edges_u_start = num_edges[u];
            uint64_t num_edges_u_end   = num_edges[u+1];
                
            for (uint64_t j=num_edges_u_start; j<num_edges_u_end; j++) {

                uint64_t v = adj[j];

                uint64_t v_owner_proc_row = v/n_local;

#if SAMEPROC_NOREAD
#if USE_MPI
                if (v_owner_proc_row == irow) {
#endif
                    uint64_t u_global = n_local_row*irow + u;
#if REPLICATE_D
                    uint32_t d_val = d[v];

                    if (d_val == 0) {
                        d[v] = vis_level;

                        queue_next_local[next_vert_count_local++] = v +
                            jcol*n_local_col;
                        if (next_vert_count_local == LOCAL_QUEUE_SIZE) {
                            uint64_t queue_next_offset = 
                                __sync_fetch_and_add(&next_vert_count,
                                LOCAL_QUEUE_SIZE);
                            next_vert_count_local = 0;
                        
                            memcpy(queue_next+queue_next_offset, 
                                queue_next_local, 
                                LOCAL_QUEUE_SIZE * 4);
                        }
#if STORE_PRED
#if PRED_CACHE_BYPASS
                        __m128i pv = 
                            _mm_set_epi64(*(__m64 *) &u_global, 
                                *(__m64 *) &v);
                        _mm_stream_si128((__m128i *) 
                            &pred_local[pred_vert_pair_count], pv);
#else
                        pred_local[pred_vert_pair_count] = v;
                        pred_local[pred_vert_pair_count+1] = u_global;
#endif

                        pred_vert_pair_count += 2;
                        if (pred_vert_pair_count == LOCAL_QUEUE_SIZE) {
                            uint64_t pred_next_offset = 
                                __sync_fetch_and_add(&pred_count,
                                LOCAL_QUEUE_SIZE);
                            pred_vert_pair_count = 0;
                        
                            memcpy(pred + pred_next_offset, 
                                pred_local, LOCAL_QUEUE_SIZE * 8);
                        }

#endif
                    }

#else

                    uint64_t v_local = v % n_local;

                    uint32_t d_val = d[v_local];

                    if (d_val == 0) {
                        d[v_local] = vis_level;

                        queue_next_local[next_vert_count_local++] = v +
                            jcol*n_local_col;
                        if (next_vert_count_local == LOCAL_QUEUE_SIZE) {
                            uint64_t queue_next_offset = 
                                __sync_fetch_and_add(&next_vert_count,
                                LOCAL_QUEUE_SIZE);
                            next_vert_count_local = 0;
                        
                            memcpy(queue_next+queue_next_offset, 
                                queue_next_local, 
                                LOCAL_QUEUE_SIZE * 4);
                        }
#if STORE_PRED
#if PRED_CACHE_BYPASS
                        __m128i pv = 
                            _mm_set_epi64(*(__m64 *) &u_global, 
                                *(__m64 *) &v);
                        _mm_stream_si128((__m128i *) 
                            &pred_local[pred_vert_pair_count], pv);
#else
                        pred_local[pred_vert_pair_count] = v;
                        pred_local[pred_vert_pair_count+1] = u_global;
#endif
 
                        pred_vert_pair_count += 2;
                        if (pred_vert_pair_count == LOCAL_QUEUE_SIZE) {
                            uint64_t pred_next_offset = 
                                __sync_fetch_and_add(&pred_count,
                                LOCAL_QUEUE_SIZE);
                            pred_vert_pair_count = 0;
                        
                            memcpy(pred + pred_next_offset, 
                                pred_local, LOCAL_QUEUE_SIZE * 8);
                        }


#endif
                    }
#endif

#endif 

#if USE_MPI
#if SAMEPROC_NOREAD
                } else {
#endif

#if REPLICATE_D
                    uint32_t d_val = d[v];

                    if (d_val == 0) {
                        d[v] = vis_level;

                        uint32_t local_pos = v_owner_proc_row*LOCAL_SENDBUF_SIZE +
                                       adj_sendbuf_counts_local[v_owner_proc_row];
                        adj_sendbuf_local[local_pos] = v;
                        adj_sendbuf_local[local_pos+1] = u;
                        adj_sendbuf_counts_local[v_owner_proc_row] += 2;
 
                        if (adj_sendbuf_counts_local[v_owner_proc_row] 
                                    == LOCAL_SENDBUF_SIZE) {
                            
                            int64_t sendbuf_counts_offset = __sync_fetch_and_add(
                                adj_sendbuf_counts+v_owner_proc_row, 
                                LOCAL_SENDBUF_SIZE);
                            
                            uint32_t adj_sendbuf_offset = 
                                adj_sendbuf_displs[v_owner_proc_row] + 
                                sendbuf_counts_offset;

                            memcpy(adj_sendbuf+adj_sendbuf_offset, 
                                adj_sendbuf_local+v_owner_proc_row*LOCAL_SENDBUF_SIZE, 
                                LOCAL_SENDBUF_SIZE * 4);
                            adj_sendbuf_counts_local[v_owner_proc_row] = 0;
                        }

                        // int32_t pos = adj_sendbuf_displs[v_owner_proc_row] + 
                        //    adj_sendbuf_counts[v_owner_proc_row];

                        //adj_sendbuf_counts[v_owner_proc_row] += 2;
                        //adj_sendbuf[pos] = v;
                        //adj_sendbuf[pos+1] = u;
                    }
#else

                    uint32_t local_pos = v_owner_proc_row*LOCAL_SENDBUF_SIZE +
                                       adj_sendbuf_counts_local[v_owner_proc_row];
                    adj_sendbuf_local[local_pos] = v;
                    adj_sendbuf_local[local_pos+1] = u;
                    adj_sendbuf_counts_local[v_owner_proc_row] += 2;
 
                    if (adj_sendbuf_counts_local[v_owner_proc_row] 
                                    == LOCAL_SENDBUF_SIZE) {
                            
                        int64_t sendbuf_counts_offset = __sync_fetch_and_add(
                            adj_sendbuf_counts+v_owner_proc_row, 
                            LOCAL_SENDBUF_SIZE);
                            
                        uint32_t adj_sendbuf_offset = 
                            adj_sendbuf_displs[v_owner_proc_row] + 
                            sendbuf_counts_offset;

                        memcpy(adj_sendbuf+adj_sendbuf_offset, 
                            adj_sendbuf_local+v_owner_proc_row*LOCAL_SENDBUF_SIZE, 
                            LOCAL_SENDBUF_SIZE * 4);
                        adj_sendbuf_counts_local[v_owner_proc_row] = 0;
                    }
#endif           
                }
#endif
#if SAMEPROC_NOREAD
            }
#endif

        }
      
#pragma omp barrier 
 
#pragma omp critical
{
#if USE_MPI
        for (int i=0; i<nproc_rows; i++) {
            if (adj_sendbuf_counts_local[i] > 0) {
                uint32_t adj_sendbuf_offset = 
                    adj_sendbuf_displs[i] + adj_sendbuf_counts[i];
                //fprintf(stderr, "rank %d, count %d, init %d, offset %d\n", i, 
                //    adj_sendbuf_counts_local[i], sendbuf_counts[i], adj_sendbuf_offset);
                memcpy(adj_sendbuf+adj_sendbuf_offset, 
                    adj_sendbuf_local+i*LOCAL_SENDBUF_SIZE, 
                    adj_sendbuf_counts_local[i] * 4);
                adj_sendbuf_counts[i] += adj_sendbuf_counts_local[i];
                adj_sendbuf_counts_local[i] = 0;
            }
        }
#endif
}



#pragma omp barrier

#if TIME_BFS_SUBROUTINES
        if (tid == 0) {
            
            if (rank == 0) {
                load_elt += get_seconds() - load_elt_part;
                mpi_elt_part = get_seconds();
            }
        }
#endif

#if USE_MPI
        if (tid == 0) {
            MPI_Alltoall(adj_sendbuf_counts, 1, MPI_UNSIGNED, 
                adj_recvbuf_counts, 1, 
                MPI_UNSIGNED, (g->comm_data).col_comm);
            MPI_Alltoallv(adj_sendbuf, adj_sendbuf_counts, 
                adj_sendbuf_displs, MPI_UNSIGNED, 
                adj_recvbuf, adj_recvbuf_counts, adj_recvbuf_displs, 
                MPI_UNSIGNED, (g->comm_data).col_comm);
#if 0
            long adj_cut_count = 0;
            for (int i=0; i<nproc_rows; i++) {
                adj_cut_count += adj_sendbuf_counts[i];
            }
            MPI_Barrier(MPI_COMM_WORLD);

            long adj_cut_count_max = 0;
            long adj_cut_count_sum = 0;

            MPI_Allreduce(&adj_cut_count, &adj_cut_count_max, 1, MPI_LONG, MPI_MAX,
                    MPI_COMM_WORLD);
            MPI_Allreduce(&adj_cut_count, &adj_cut_count_sum, 1, MPI_LONG, MPI_SUM,
                    MPI_COMM_WORLD);

            adj_cut_count_total += adj_cut_count;

            if ((rank == 0) && (adj_cut_count_max > g->m_local/10))
                fprintf(stderr, "alltoallv, level %d, max %ld, total %ld, imbalance %3.4lf\n", 
                        vis_level, 
                    adj_cut_count_max, 
                    adj_cut_count_sum, ((double) adj_cut_count_max)*nprocs/adj_cut_count_sum);
            // MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
#endif

#if TIME_BFS_SUBROUTINES
#if 0
        uint64_t datavol_sendsize = 0;
        uint64_t datavol_recvsize = 0;
        for (i=0; i<nproc_rows; i++) {
            datavol_sendsize += adj_sendbuf_counts[i];
            datavol_recvsize += adj_recvbuf_counts[i];
        }
        MPI_Barrier((g->comm_data).col_comm);
#endif
        if (tid == 0) 
            MPI_Barrier((g->comm_data).col_comm);
        
        if ((rank == 0) && (tid == 0)) {
            mpi_elt_part = get_seconds() - mpi_elt_part;
            /*
            if (mpi_elt_part > 0.1) {
                fprintf(stderr, "level %lu, datavol %lu %lu, time %9.6lf s\n", 
                        vis_level, datavol_sendsize, datavol_recvsize, mpi_elt_part);        
            }
            */
            mpi_elt += mpi_elt_part;
            sift_elt_part = get_seconds();
        }
#endif

#pragma omp barrier

#if USE_MPI

#pragma omp for
        for (uint64_t i=0; i<nproc_rows; i++) {
        
            uint64_t recvbuf_start = adj_recvbuf_displs[i];
            uint64_t recvbuf_end = recvbuf_start + adj_recvbuf_counts[i];
            
            for (uint64_t j=recvbuf_start; j<recvbuf_end; j+= 2) {

                uint64_t v        = adj_recvbuf[j];
                uint64_t u_pred   = adj_recvbuf[j+1]; 
                uint64_t u_global = i*n_local_row+u_pred; 

#if REPLICATE_D
                uint32_t d_val = d[v];

                if (d_val == 0) {
                    d[v] = vis_level;

                    // queue_next[next_vert_count++] = v + jcol*n_local_col;
 
                    queue_next_local[next_vert_count_local++] = v +
                        jcol*n_local_col;
                    if (next_vert_count_local == LOCAL_QUEUE_SIZE) {
                        uint64_t queue_next_offset = 
                            __sync_fetch_and_add(&next_vert_count,
                            LOCAL_QUEUE_SIZE);
                        next_vert_count_local = 0;
                        
                        memcpy(queue_next+queue_next_offset, 
                            queue_next_local, 
                            LOCAL_QUEUE_SIZE * 4);
                    }

#if STORE_PRED
#if PRED_CACHE_BYPASS
                    __m128i pv = 
                        _mm_set_epi64(*(__m64 *) &u_global, 
                            *(__m64 *) &v);
                    _mm_stream_si128((__m128i *) 
                        &pred_local[pred_vert_pair_count], pv);
#else
                    pred_local[pred_vert_pair_count] = v;
                    pred_local[pred_vert_pair_count+1] = u_global;
#endif
 
                    pred_vert_pair_count += 2;
                    if (pred_vert_pair_count == LOCAL_QUEUE_SIZE) {
                        uint64_t pred_next_offset = 
                            __sync_fetch_and_add(&pred_count,
                            LOCAL_QUEUE_SIZE);
                        pred_vert_pair_count = 0;
                        
                        memcpy(pred + pred_next_offset, 
                            pred_local, LOCAL_QUEUE_SIZE * 8);
                    }
#endif
                }
#else

                uint64_t v_local = v % n_local;
                uint32_t d_val = d[v_local];

                if (d_val == 0) {
                    d[v_local] = vis_level;

                    queue_next_local[next_vert_count_local++] = v +
                        jcol*n_local_col;
                    if (next_vert_count_local == LOCAL_QUEUE_SIZE) {
                        uint64_t queue_next_offset = 
                            __sync_fetch_and_add(&next_vert_count,
                            LOCAL_QUEUE_SIZE);
                        next_vert_count_local = 0;
                        
                        memcpy(queue_next+queue_next_offset, 
                            queue_next_local, 
                            LOCAL_QUEUE_SIZE * 4);
                    }

                   // queue_next[next_vert_count++] = v+jcol*n_local_col;
#if STORE_PRED 
#if PRED_CACHE_BYPASS
                    __m128i pv = 
                        _mm_set_epi64(*(__m64 *) &u_global, 
                            *(__m64 *) &v);
                    _mm_stream_si128((__m128i *) 
                        &pred_local[pred_vert_pair_count], pv);
#else
                    pred_local[pred_vert_pair_count] = v;
                    pred_local[pred_vert_pair_count+1] = u_global;
#endif
 
                    pred_vert_pair_count += 2;
                    if (pred_vert_pair_count == LOCAL_QUEUE_SIZE) {
                        uint64_t pred_next_offset = 
                            __sync_fetch_and_add(&pred_count,
                            LOCAL_QUEUE_SIZE);
                        pred_vert_pair_count = 0;
                        
                        memcpy(pred + pred_next_offset, 
                            pred_local, LOCAL_QUEUE_SIZE * 8);
                    }

#endif
                }

#endif
            }
        }
#endif

#pragma omp barrier

        /* Critical section to write out remaining verts */

#pragma omp critical
{
        if (next_vert_count_local > 0) {
            memcpy(queue_next+next_vert_count, queue_next_local, 
                next_vert_count_local * 4);
            next_vert_count += next_vert_count_local;
            next_vert_count_local = 0;
        }

        if (pred_vert_pair_count > 0) {
             uint64_t pred_next_offset = 
                __sync_fetch_and_add(&pred_count,
                            pred_vert_pair_count);
             memcpy(pred + pred_next_offset, 
                    pred_local, pred_vert_pair_count * 8);
             pred_vert_pair_count = 0;
        }
}


#pragma omp barrier
    
        if (tid == 0) {

#if TIME_BFS_SUBROUTINES
        MPI_Barrier((g->comm_data).col_comm);
        if (rank == 0) {
            sift_elt += get_seconds() - sift_elt_part;
            trans_elt_part = get_seconds();
        }
#endif

        /* Transpose queue_next */
        uint64_t next_vert_count_trans = 0; 
#if USE_MPI
        MPI_Status status1, status2;
#endif

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        assert(recv_proc < nproc_rows * nproc_cols);
#if USE_MPI
        MPI_Sendrecv(&next_vert_count, 1, MPI_UNSIGNED_LONG,
                jcol*nproc_rows+irow, rank % (nproc_cols*nproc_rows), 
                &next_vert_count_trans, 1, MPI_UNSIGNED_LONG, 
                recv_proc, recv_proc, (g->comm_data).replicas_comm, 
                &status1);

        MPI_Sendrecv(queue_next, next_vert_count, MPI_UNSIGNED,
                jcol*nproc_rows+irow, rank % (nproc_cols*nproc_rows),
                queue_current, next_vert_count_trans, MPI_UNSIGNED, 
                recv_proc, recv_proc, (g->comm_data).replicas_comm, 
                &status2);
#else
        next_vert_count_trans = next_vert_count;
        memcpy(queue_current, queue_next,
                sizeof(uint32_t)*next_vert_count_trans);
#endif
        /* Allgather */
        sendbuf_counts[0] = next_vert_count_trans;
#if USE_MPI
        MPI_Allgather(sendbuf_counts, 1, MPI_INT, recvbuf_counts, 1, 
                MPI_INT, (g->comm_data).row_comm);
#else
        recvbuf_counts[0] = sendbuf_counts[0];
#endif
        recvbuf_displs[0] = 0; 
        for (int i=1; i<nproc_cols; i++) {
            recvbuf_displs[i] = recvbuf_displs[i-1] +
                recvbuf_counts[i-1];
        }
        current_vert_count = recvbuf_displs[nproc_cols-1] + 
                                recvbuf_counts[nproc_cols-1];

        assert(current_vert_count < n_local_row);

#if USE_MPI
        MPI_Allgatherv(queue_current, next_vert_count_trans,
                MPI_UNSIGNED, queue_next, recvbuf_counts,
                recvbuf_displs, MPI_UNSIGNED, (g->comm_data).row_comm);
#if 0
        uint64_t current_vert_count_total = 0;
        uint64_t current_vert_count_max = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&current_vert_count, &current_vert_count_max, 1, MPI_UNSIGNED_LONG, MPI_MAX,
            (g->comm_data).col_comm);
        MPI_Allreduce(&current_vert_count, &current_vert_count_total, 1, MPI_UNSIGNED_LONG, MPI_SUM,
            (g->comm_data).col_comm);


        if ((rank == 0) && (current_vert_count_total > n_local/4))
            fprintf(stderr, "gather, level %d, max %ld, total %ld, imbalance %3.4lf\n", vis_level, 
                current_vert_count_max, 
                current_vert_count_total, ((double)
                    current_vert_count_max)*nproc_cols/current_vert_count_total);
#endif
#else
        memcpy(queue_next, queue_current,
                next_vert_count_trans*sizeof(uint32_t));
#endif
        
        tmp_queue_ptr = queue_current;
        queue_current = queue_next;
        queue_next = tmp_queue_ptr;
        

#if TIME_BFS_SUBROUTINES
        MPI_Barrier((g->comm_data).row_comm);
        if (rank == 0) {
            trans_elt += get_seconds() - trans_elt_part;
        }
#endif

        *nvisited += current_vert_count;
        //fprintf(stderr, "rank %d, current_vert_count %lu\n", 
        //       rank, current_vert_count);
        //uint64_t tmp_vert_count = current_vert_count;
        //current_vert_count = next_vert_count;
        next_vert_count = 0;
        

#if USE_MPI
        MPI_Allreduce(&current_vert_count, &global_queue_vert_count, 1, 
                MPI_UNSIGNED_LONG, MPI_SUM, (g->comm_data).col_comm);
#else
        global_queue_vert_count = current_vert_count;
#endif
        vis_level++;
        num_a2a_iterations++;

        /*
        if (rank == 0)
            fprintf(stderr, "vis level %lu, global vert count %lu\n", vis_level, 
                global_queue_vert_count);
        */
        /*
        for (i=0; i<nprocs; i++) {
            sendbuf_counts[i] = 0;
            recvbuf_counts[i] = 0;
        }
        */
        assert(vis_level < 255);
        }
#pragma omp barrier

    }

#if USE_MPI
    if (tid == 0) {
        MPI_Allreduce(MPI_IN_PLACE, nvisited, 1, MPI_UNSIGNED_LONG,
            MPI_SUM, (g->comm_data).col_comm);
        MPI_Barrier((g->comm_data).replicas_comm);
    }
#endif

#if TIME_BFS_SUBROUTINES
    if ((rank == 0) && (tid == 0)) {
        full_elt = get_seconds() - full_elt;
        fprintf(stderr, "time: %9.6lf s.\n"
                "sift %9.6lf s. (%3.3lf), load %9.6lf s. (%3.3lf)\n"
                "a2a  %9.6lf s. (%3.3lf), gather %9.6lf s. (%3.3lf)\n"
                "comm %9.6lf s. (%3.3lf), comp %9.6lf s. (%3.3lf)\n"
                "init %9.6lf s., vis levels %lu, nvisited %lu\n",
                full_elt, 
                sift_elt, (sift_elt/full_elt)*100.0,
                load_elt, (load_elt/full_elt)*100.0,
                mpi_elt,  (mpi_elt/full_elt)*100.0,
                trans_elt, (trans_elt/full_elt)*100.0,
                mpi_elt+trans_elt, ((mpi_elt+trans_elt)/full_elt)*100.0,
                sift_elt+load_elt, ((sift_elt+load_elt)/full_elt)*100.0,
                init_elt, vis_level, *nvisited);
    }
#endif
}
    //fprintf(stderr, "rank %d, total adj cut %ld\n", rank, adj_cut_count_total);
    //*pred_array_size_ptr = pred_vert_pair_count/2;
    *pred_array_size_ptr = pred_count/2;
    // fprintf(stderr, "rank %d, pred count: %lu\n", rank, pred_count/2);

    return 0;
}

