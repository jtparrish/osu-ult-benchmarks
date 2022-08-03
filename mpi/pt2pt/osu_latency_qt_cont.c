#define BENCHMARK "OSU MPI%s Multi-threaded Latency Test"
/*
 * Copyright (C) 2002-2019 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>
#include <openmpi/mpiext/mpiext_continue_c.h>

#include <qthread.h>
#include <qthread/qthread.h>
#include <qthread/barrier.h>

//abt ABT_mutex finished_size_mutex;
aligned_t finished_size_mutex;
//abt ABT_cond  finished_size_cond;
qt_barrier_t *finished_size_barrier;
//abt ABT_mutex finished_size_sender_mutex;
aligned_t finished_size_sender_mutex;
//abt ABT_cond  finished_size_sender_cond;
qt_barrier_t *finished_size_sender_barrier;

//abt ABT_barrier sender_barrier;
qt_barrier_t *sender_barrier;

qt_barrier_t *completion_barrier;

aligned_t polling_completion;

double t_start = 0, t_end = 0;

//abt int finished_size;
aligned_t finished_size;
//abt int finished_size_sender;
aligned_t finished_size_sender;

int num_threads_sender=1;
//abt int num_xstreams_sender=1;
int num_sheps_sender=1;
typedef struct {
    //abt ABT_cond  cond;
    aligned_t block_feb;
    //abt ABT_mutex mtx;
    int id;
} thread_state_t;

typedef struct thread_cont {
} thread_cont_t;

aligned_t send_thread(void *arg);
aligned_t recv_thread(void *arg);

static
void block_thread(thread_state_t *thread_state, MPI_Request *req);
static
void unblock_thread(MPI_Status *stat, void *data);
static
aligned_t cont_progress(void* data);

static MPI_Request cont_req;

static volatile int progress = 1;

int main(int argc, char *argv[])
{
    int numprocs, provided, myid, err;
    int i = 0;
    int po_ret = 0;
    //abt ABT_thread sr_threads[MAX_NUM_THREADS];
    //abt ABT_xstream sr_xstreams[MAX_NUM_THREADS];
    thread_state_t tags[MAX_NUM_THREADS];
    //abt ABT_xstream es;
    qthread_shepherd_id_t shep;
    //abt ABT_thread prog_thread;

    options.bench = PT2PT;
    options.subtype = LAT_QT;

    set_header(HEADER);
    set_benchmark_name("osu_latency_mt");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPIX_Continue_init(&cont_req, MPI_INFO_NULL);

    //abt ABT_init(argc, argv);
    if(options.sender_sheps!=-1) {
        num_sheps_sender=options.sender_sheps;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));
    qthread_init( (0 == myid) ? num_sheps_sender : options.num_sheps); 
    //abt ABT_xstream_self(&es);
    //abt sr_xstreams[0] = es;
    // make threads migratable
    //abt ABT_thread_attr attr;
    //abt ABT_thread_attr_create(&attr);
    //abt ABT_thread_attr_set_migratable(attr, ABT_TRUE);

    //abt ABT_mutex_create(&finished_size_mutex);
    //abt ABT_cond_create(&finished_size_cond);
    finished_size_barrier = qt_barrier_create(options.num_threads, REGION_BARRIER);
    //abt ABT_mutex_create(&finished_size_sender_mutex);
    //abt ABT_cond_create(&finished_size_sender_cond);
    finished_size_sender_barrier = qt_barrier_create(num_threads_sender, REGION_BARRIER);


    if(err != MPI_SUCCESS) {
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not available.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not available.\n");
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    /* Check to make sure we actually have a thread-safe
     * implementation 
     */

    finished_size = 1;
    finished_size_sender=1;

    if(provided != MPI_THREAD_MULTIPLE) {
        if(myid == 0) {
            fprintf(stderr,
                "MPI_Init_thread must return MPI_THREAD_MULTIPLE!\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }
    
    
    printf("options.sender_thread %d\n", options.sender_thread);
    if(options.sender_thread!=-1) {
        num_threads_sender=options.sender_thread;
    }

    /*abt
    if(options.sender_xstreams!=-1) {
        num_xstreams_sender=options.sender_xstreams;
    }
    */
   
    //abt ABT_barrier_create(num_threads_sender, &sender_barrier);
    sender_barrier = qt_barrier_create(num_threads_sender, REGION_BARRIER);
    //abt ABT_thread_create_on_xstream(es, cont_progress, NULL, attr, &prog_thread);
    qthread_fork_to(cont_progress, NULL, &polling_completion, shep);

    if(myid == 0) {
        completion_barrier = qt_barrier_create(num_threads_sender + 1, REGION_BARRIER);

        printf("# Number of Sender threads: %d \n# Number of Receiver threads: %d\n",num_threads_sender,options.num_threads );
        //abt printf("# Number of Sender xstreams: %d \n# Number of Receiver xstreams: %d\n",num_xstreams_sender,options.num_xstreams );
        printf("# Number of Sender sheps: %d \n# Number of Receiver sheps: %d\n",num_sheps_sender,options.num_sheps );
    
        print_header(myid, LAT_MT);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);

        /*abt
        for (i = 1; i < num_xstreams_sender; i++) {
            ABT_xstream_create(ABT_SCHED_NULL, &sr_xstreams[i]);
        }
        */

        //abt int es_idx = 0;
        int shep_idx = 0;
        for(i=0;i<num_threads_sender;i++) {
            tags[i].id = i;
            //COMP: replace these
            //abt ABT_mutex_create(&tags[i].mtx);
            //abt ABT_cond_create(&tags[i].cond);
            qthread_empty(&tags[i].block_feb);
            //abt ABT_xstream es = sr_xstreams[(es_idx++)%(num_xstreams_sender)];
            qthread_shepherd_id_t shep = (shep_idx++)%(num_sheps_sender);
            //abt ABT_thread_create_on_xstream(es, send_thread, &tags[i], attr, &sr_threads[i]);
            qthread_fork_to(send_thread, &tags[i], NULL, shep);
        }
        printf("Waiting for all threads to complete\n");
        //Q: why is this in a loop?
        /*abt
        for (i = 0; i < num_threads_sender; ++i) {
            ABT_thread_join(sr_threads[i]);
            ABT_thread_free(&sr_threads[i]);
        }
        */

        qt_barrier_enter(completion_barrier);

        /*abt
        printf("Cancelling xstreams\n");
        for (i = 1; i < num_xstreams_sender; i++) {
            printf("Cancelling xstream %d\n", i);
            ABT_xstream_cancel(sr_xstreams[i]);
        }
        */

        /*abt
        for (i = 1; i < num_xstreams_sender; i++) {
            printf("Joining xstream %d\n", i);
            //ABT_xstream_join(sr_xstreams[i]);
        }
        */

    }

    else {
        completion_barrier=qt_barrier_create(options.num_threads + 1, REGION_BARRIER);
        /*abt
        for (i = 1; i < options.num_xstreams; i++) {
            ABT_xstream_create(ABT_SCHED_NULL, &sr_xstreams[i]);
        }
        */

        //abt int es_idx = 0;
        int shep_idx = 0;
        for(i = 0; i < options.num_threads; i++) {
            tags[i].id = i;
            //COMP: replace these
            //abt ABT_mutex_create(&tags[i].mtx);
            //abt ABT_cond_create(&tags[i].cond);
            qthread_empty(&tags[i].block_feb);
            //abt ABT_xstream es = sr_xstreams[(es_idx++)%(options.num_xstreams)];
            qthread_shepherd_id_t shep = (shep_idx++)%(options.num_sheps);
            //abt ABT_thread_create_on_xstream(es, recv_thread, &tags[i], attr, &sr_threads[i]);
            qthread_fork_to(recv_thread, &tags[i], NULL, shep);
        }
        //Q: why is this in a loop?
        /*abt
        for (i = 0; i < options.num_threads; ++i) {
            ABT_thread_join(sr_threads[i]);
            ABT_thread_free(&sr_threads[i]);
        }
        */

        qt_barrier_enter(completion_barrier);

        /*abt
        printf("Cancelling xstreams\n");
        for (i = 1; i < options.num_xstreams; i++) {
            ABT_xstream_cancel(sr_xstreams[i]);
        }
        */

        /*abt
        for (i = 1; i < options.num_xstreams; i++) {
            //ABT_xstream_join(sr_xstreams[i]);
        }
        */
        
        qt_barrier_enter(completion_barrier);
    }

    progress = 0;
    //abt ABT_thread_join(prog_thread);
    qthread_readFF(NULL, &polling_completion);

    printf("Freeing cont_req\n");
    MPI_Wait(&cont_req, MPI_STATUS_IGNORE);
    MPI_Request_free(&cont_req);

    printf("Tearing down QT\n");

    //abt ABT_thread_attr_free(&attr);
    //abt ABT_finalize();
    qthread_finalize();
    printf("Calling MPI_Finalize\n");
    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

aligned_t recv_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val;
    int iter;
    int myid;
    char *s_buf, *r_buf;
    thread_state_t *thread_state;

    thread_state = (thread_state_t *)arg;
    val = thread_state->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_state->id);
        return 1;
    }

    for(size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        //abt ABT_mutex_lock(finished_size_mutex);

        /*abt
        if(finished_size == options.num_threads) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            finished_size = 1;

            ABT_mutex_unlock(finished_size_mutex);
            ABT_cond_broadcast(finished_size_cond);
        }

        else {
            finished_size++;

            ABT_cond_wait(finished_size_cond, finished_size_mutex);
            ABT_mutex_unlock(finished_size_mutex);
        }
        */

        if (qthread_incr(&finished_size, 1) == options.num_threads) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            const aligned_t one = 1;
            qthread_writeF(&finished_size, &one);
        } 

        qt_barrier_enter(finished_size_barrier);

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        for(i = val; i < (options.iterations + options.skip); i += options.num_threads) {
            MPI_Request req;
            if(options.sender_thread>1) {
                MPI_Irecv (r_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD, &req);
                printf("recv block\n");
                block_thread(thread_state, &req);
                printf("recv block\n");
                MPI_Isend (s_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD, &req);
                block_thread(thread_state, &req);
            }
            else{
                MPI_Irecv (r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &req);
                block_thread(thread_state, &req);
                MPI_Isend (s_buf, size, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &req);
                block_thread(thread_state, &req);
            }
        }

        iter++;
    }

    free_memory(s_buf, r_buf, myid);

    qt_barrier_enter(completion_barrier);

    return 0;
}


aligned_t send_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val, iter;
    int myid;
    char *s_buf, *r_buf;
    double t = 0, latency;
    thread_state_t *thread_state = (thread_state_t *)arg;
    char *ret = NULL;

    val = thread_state->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_state->id);
        return 1;
    }

    for(size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        //abt ABT_mutex_lock(finished_size_sender_mutex);
        
        /*abt
        if(finished_size_sender == num_threads_sender) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        
            finished_size_sender = 1;

            ABT_mutex_unlock(finished_size_sender_mutex);
            ABT_cond_broadcast(finished_size_sender_cond);
        }

        else {
            
            finished_size_sender++;

            ABT_cond_wait(finished_size_sender_cond, finished_size_sender_mutex);
            ABT_mutex_unlock(finished_size_sender_mutex);
        }
        */

        if (qthread_incr(&finished_size_sender, 1) == num_threads_sender) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            const aligned_t one = 1;
            qthread_writeF(&finished_size_sender, &one);
        } 

        printf("Thread entering finished_size_sender_barrier\n");
        qt_barrier_enter(finished_size_sender_barrier);
        printf("Thread exiting finished_size_sender_barrier\n");

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        printf("debug: touch\n");
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);
        printf("debug: touch end\n");

        printf("debug: comm\n");
        int flag_print=0;
        printf("iterations: %zu\n", options.iterations);
        for(i = val; i < options.iterations + options.skip; i+=num_threads_sender) {
            if (i % 1000 == 0) {
                printf("iteration: %d\n", i);
            }
            if(i == options.skip) {
                t_start = MPI_Wtime();
                flag_print =1;
            }

            if(options.sender_thread>1) {
                MPI_Request req;
                
                MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD, &req));
                block_thread(thread_state, &req);
                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD, &req));
                block_thread(thread_state, &req);
            }
            else{
                MPI_Request req;
                MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &req));
                block_thread(thread_state, &req);
                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 1, 2, MPI_COMM_WORLD, &req));
                block_thread(thread_state, &req);
            }
        }
        printf("debug: comm end\n");

        //abt ABT_barrier_wait(sender_barrier);
        qt_barrier_enter(sender_barrier);
        if(flag_print==1) {
            t_end = MPI_Wtime ();
            t = t_end - t_start;

            latency = (t) * 1.0e6 / (2.0 * options.iterations / num_threads_sender);
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                    latency);
            fflush(stdout);
        }
        iter++;
    }

    free_memory(s_buf, r_buf, myid);

    qt_barrier_enter(completion_barrier);

    return 0;
}

static _Thread_local int completed = 0;

static
aligned_t cont_progress(void* data)
{
    int flag; // <- ignored
    while (progress) {
        //do {
        //completed = 0;
        MPI_Test(&cont_req, &flag, MPI_STATUS_IGNORE);
        //} while (completed > 0);
        //abt ABT_thread_yield();
        qthread_yield();
    }

    return 0;
}
static
void block_thread(thread_state_t *thread_state, MPI_Request *req)
{
    int flag;
    //abt ABT_mutex_lock(thread_state->mtx);
    MPI_Request _req = *req; // DEBUG
    MPIX_Continue(&_req, &unblock_thread, thread_state, MPI_STATUS_IGNORE, cont_req);
    //printf("Blocking thread %d, flag %d\n", thread_state->id, flag);
    /*abt
    if (!flag) {
        ABT_cond_wait(thread_state->cond, thread_state->mtx);
    }
    */
    qthread_readFE(NULL, &thread_state->block_feb);
    //abt ABT_mutex_unlock(thread_state->mtx);
    //printf("Unblocked thread %d\n", thread_state->id);
}
static
void unblock_thread(MPI_Status *stat, void *data)
{
    thread_state_t *thread_state = (thread_state_t *)data;
    //printf("Unblocking thread %d\n", thread_state->id);
    //abt ABT_mutex_lock(thread_state->mtx);
    //abt ABT_cond_signal(thread_state->cond);
    qthread_fill(&thread_state->block_feb); // or just fill
    //abt ABT_mutex_unlock(thread_state->mtx);
    completed++;
}



/* vi: set sw=4 sts=4 tw=80: */
