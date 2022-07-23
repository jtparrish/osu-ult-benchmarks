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

#include <abt.h>

ABT_mutex finished_size_mutex;
//# aligned_t finished_size_mutex;
ABT_cond  finished_size_cond;
//# aligned_t finished_size_cond;
ABT_mutex finished_size_sender_mutex;
//# aligned_t finished_size_sender_mutex;
ABT_cond  finished_size_sender_cond;
//# aligned_t finished_size_sender_cond;

ABT_barrier sender_barrier;
//# qt_feb_barrier_t *sender_barrier;

double t_start = 0, t_end = 0;

int finished_size;
int finished_size_sender;

int num_threads_sender=1;
int num_xstreams_sender=1;
//# int num_sheps_sender=1;
typedef struct thread_tag  {
        int id;
} thread_tag_t;

void send_thread(void *arg);
void recv_thread(void *arg);

int main(int argc, char *argv[])
{
    int numprocs, provided, myid, err;
    int i = 0;
    int po_ret = 0;
    ABT_thread sr_threads[MAX_NUM_THREADS];
    //-
    ABT_xstream sr_xstreams[MAX_NUM_THREADS];
    //-
    thread_tag_t tags[MAX_NUM_THREADS];
    ABT_xstream es;
    //# qthread_shepherd_id_t shep;
    

    options.bench = PT2PT;
    options.subtype = LAT_ABT;

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

    ABT_init(argc, argv);
    //TODO: make sure these are initialized before using them in init
    //# qthread_init( (0 == myid) ? num_sender_sheps : num_sheps); 
    ABT_xstream_self(&es);
    //# shep = qthread_shep()
    sr_xstreams[0] = es;
    //ABT_xstream_set_cpubind(es, 0);
    // make threads migratable
    // TODO: figure out how to replicate this in Qthreads
    ABT_thread_attr attr;
    //-
    ABT_thread_attr_create(&attr);
    //-
    ABT_thread_attr_set_migratable(attr, ABT_TRUE);
    //-

    ABT_mutex_create(&finished_size_mutex);
    //-
    ABT_cond_create(&finished_size_cond);
    //-
    ABT_mutex_create(&finished_size_sender_mutex);
    //-
    ABT_cond_create(&finished_size_sender_cond);
    //-


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

    if(options.sender_xstreams!=-1) {
        num_xstreams_sender=options.sender_xstreams;
    }
   
    ABT_barrier_create(num_threads_sender, &sender_barrier);
    //# sender_barrier = qthread_feb_barrier_create(num_threads_sender)


    if(myid == 0) {
        printf("# Number of Sender threads: %d \n# Number of Receiver threads: %d\n",num_threads_sender,options.num_threads );
        printf("# Number of Sender xstreams: %d \n# Number of Receiver xstreams: %d\n",num_xstreams_sender,options.num_xstreams );
    
        print_header(myid, LAT_MT);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);

        for (i = 1; i < num_xstreams_sender; i++) {
            //TODO: find the qthreads equivalent for this
            //IDEA: delete this loop
            ABT_xstream_create(ABT_SCHED_NULL, &sr_xstreams[i]);
            //ABT_xstream_set_cpubind(sr_xstreams[i], i);
        }

        int es_idx = 0;
        for(i=0;i<num_threads_sender;i++) {
            tags[i].id = i;
            //COMP: find the qthreads equivalent for this
            ABT_xstream es = sr_xstreams[(es_idx++)%(num_xstreams_sender)];
            //# qthread_shepherd_id_t shep = (shep_idx++)%(num_sheps_sender);
            ABT_thread_create_on_xstream(es, send_thread, &tags[i], attr, &sr_threads[i]);
            //# qthread_fork_to(send_thread, &tags[i], NULL, shep);
        }
        printf("Waiting for all threads to complete\n");
        //TODO: find the qthreads equivalent for this
        //IDEA: sincs
        ABT_thread_join_many(num_threads_sender, sr_threads);
        printf("Cancelling xstreams\n");
        for (i = 1; i < num_xstreams_sender; i++) {
            printf("Cancelling xstream %d\n", i);
            //TODO: find the qthreads equivalent for this
            //IDEA: delete?
            ABT_xstream_cancel(sr_xstreams[i]);
        }
        for (i = 1; i < num_xstreams_sender; i++) {
            printf("Joining xstream %d\n", i);
            //ABT_xstream_join(sr_xstreams[i]);
        }
    }

    else {
        for (i = 1; i < options.num_xstreams; i++) {
            //TODO: find the qthreads equivalent for this
            //IDEA: delete this loop
            ABT_xstream_create(ABT_SCHED_NULL, &sr_xstreams[i]);
        }
        int es_idx = 0;
        for(i = 0; i < options.num_threads; i++) {
            tags[i].id = i;
            //TODO: find the qthreads equivalent for this
            ABT_xstream es = sr_xstreams[(es_idx++)%(options.num_xstreams)];
            ABT_thread_create_on_xstream(es, recv_thread, &tags[i], attr, &sr_threads[i]);
        }
        //TODO: find the qthreads equivalent for this
        //IDEA: sincs
        ABT_thread_join_many(options.num_threads, sr_threads);

        for (i = 1; i < options.num_xstreams; i++) {
            //TODO: find the qthreads equivalent for this
            //IDEA: delete?
            ABT_xstream_cancel(sr_xstreams[i]);
        }
        for (i = 1; i < options.num_xstreams; i++) {
            //ABT_xstream_join(sr_xstreams[i]);
        }
        
    }

    ABT_thread_attr_free(&attr);
    ABT_finalize();
    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

void recv_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val;
    int iter;
    int myid;
    char *s_buf, *r_buf;
    thread_tag_t *thread_id;

    thread_id = (thread_tag_t *)arg;
    val = thread_id->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_id->id);
        return;
    }

    for(size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        ABT_mutex_lock(finished_size_mutex);
        //# qthread_lock(&finished_size_mutex);

        if(finished_size == options.num_threads) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            finished_size = 1;

            ABT_mutex_unlock(finished_size_mutex);
            //# qthread_unlock(&finished_size_mutex);
            ABT_cond_broadcast(finished_size_cond);
            //# qthread_fill(&finished_size_cond);
        }

        else {
            finished_size++;

            ABT_cond_wait(finished_size_cond, finished_size_mutex);
            //TODO: find how to release the mutex
            //IDEA: is the mutex even necessary it gets released?
            //# qthread_readFF(NULL, &finished_size_cond)
            ABT_mutex_unlock(finished_size_mutex);
            //TODO: see above
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        for(i = val; i < (options.iterations + options.skip); i += options.num_threads) {
            if(options.sender_thread>1) {
                MPI_Recv (r_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD,
                        &reqstat[val]);
                MPI_Send (s_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD);
            }
            else{
                MPI_Recv (r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD,
                        &reqstat[val]);
                MPI_Send (s_buf, size, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
            }
        }

        iter++;
    }

    free_memory(s_buf, r_buf, myid);

}


void send_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val, iter;
    int myid;
    char *s_buf, *r_buf;
    double t = 0, latency;
    thread_tag_t *thread_id = (thread_tag_t *)arg;
    char *ret = NULL;

    val = thread_id->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_id->id);
        return;
    }

    for(size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        ABT_mutex_lock(finished_size_sender_mutex);
        //# qthread_lock(&finished_size_sender_mutex);
        
        if(finished_size_sender == num_threads_sender) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        
            finished_size_sender = 1;

            ABT_mutex_unlock(finished_size_sender_mutex);
            //# qthread_unlock(&finished_size_sender_mutex);
            ABT_cond_broadcast(finished_size_sender_cond);
            //# qthread_fill(&finished_size_sender_cond);
        }

        else {
            
            finished_size_sender++;

            ABT_cond_wait(finished_size_sender_cond, finished_size_sender_mutex);
            //TODO: find how to release the mutex
            //IDEA: is the mutex even necessary it gets released?
            //# qthread_readFF(NULL, &finished_size_cond)
            ABT_mutex_unlock(finished_size_sender_mutex);
            //TODO: see above
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        int flag_print=0;
        for(i = val; i < options.iterations + options.skip; i+=num_threads_sender) {
            if(i == options.skip) {
                t_start = MPI_Wtime();
                flag_print =1;
            }

            if(options.sender_thread>1) {
                
                
                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD,
                        &reqstat[val]));
            }
            else{

                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, 2, MPI_COMM_WORLD,
                        &reqstat[val]));
            
            }
        }

        ABT_barrier_wait(sender_barrier);
        if(flag_print==1) {
            t_end = MPI_Wtime();
            t = t_end - t_start;

            latency = (t) * 1.0e6 / (2.0 * options.iterations / num_threads_sender);
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                    latency);
            fflush(stdout);
        }
        iter++;
    }

    free_memory(s_buf, r_buf, myid);

}

/* vi: set sw=4 sts=4 tw=80: */
