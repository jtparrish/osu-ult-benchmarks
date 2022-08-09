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

#define _POSIX_C_SOURCE 199309L
//#define USE_NEW_CONT_API 1

#include <osu_util_mpi.h>

//#if USE_NEW_CONT_API
#include <mpi-ext.h>
//#endif // 0
#include <mpi.h>

#include <time.h>

#include <omp.h>

int myid;


double t_start = 0, t_end = 0;

int finished_size;
int finished_size_sender;

int num_threads_sender=1;
int num_xstreams_sender=1;
_Atomic int active_thread_count;
typedef struct thread_tag  {
    int id;
    int myid;
    int size;
    int tag;
    omp_event_handle_t event;
    char *s_buf, *r_buf;
    MPI_Status status;
} thread_tag_t;

thread_tag_t *tags = NULL;

MPI_Request cont_req;

void send_thread();
void recv_thread();

typedef void (task_cb_t)(void);

static char *dep_buffer = NULL;

#define TIMESPEC_NS(_ts) (_ts.tv_sec*1E9 + _ts.tv_nsec)

static inline
double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return TIMESPEC_NS(ts);
}

static volatile int do_progress = 1; // whether to keep triggering progress
static volatile int in_compute  = 0; // whether the compute task has been

static void* progress_thread(void *ptr)
{
      struct timespec sleep_ts = {.tv_sec = 0, .tv_nsec = 10};
      double poll_interval_ns = 100;
      //using clock = std::chrono::high_resolution_clock;
      printf("Progress task starting\n");
      double last_progress_ts = get_time_ns();
      while(do_progress) {
        double ts = get_time_ns();
        if (poll_interval_ns < ts - last_progress_ts)
        {
            int flag = 0;
            MPI_Test(&cont_req, &flag, MPI_STATUS_IGNORE);
            last_progress_ts = get_time_ns();
#if USE_NEW_CONT_API
            if (flag) {
                // restart the CR to restart processing them
                MPI_Start(&cont_req);
            }
#endif
        }
#if 0
        // it is not safe to yield before compute has started because the thread might steal the compute task
        if (in_compute) {
            #pragma omp taskyield
        } else {
            nanosleep(&sleep_ts, NULL);
        }
#endif // 0
      }
      return NULL;
}

static void start_with_progress(task_cb_t *task_cb)
{
    static volatile int do_progress = 1; // whether to keep triggering progress
    static volatile int in_compute  = 0; // whether the compute task has been

    printf("start_with_progress %d\n", omp_get_thread_num());

    #pragma omp single nowait
{
    #pragma omp task default(shared) untied
    {
      struct timespec sleep_ts = {.tv_sec = 0, .tv_nsec = 10};
      double poll_interval_ns = 100;
      //using clock = std::chrono::high_resolution_clock;
      printf("Progress task starting\n");
      double last_progress_ts = get_time_ns();
      while(do_progress) {
        double ts = get_time_ns();
        if (poll_interval_ns < ts - last_progress_ts)
        {
            int flag = 0;
            MPI_Test(&cont_req, &flag, MPI_STATUS_IGNORE);
            last_progress_ts = get_time_ns();
            if (flag) {
                // restart the CR to restart processing them
                MPI_Start(&cont_req);
            }
        }
        // it is not safe to yield before compute has started because the thread might steal the compute task
        if (in_compute) {
            #pragma omp taskyield
        } else {
            nanosleep(&sleep_ts, NULL);
        }
      }
    }
}
    {
      #pragma omp task default(shared) untied
      {
          printf("Compute task\n");
          in_compute = 1; // signal that the progress task can start yielding without risking to take away this task
          task_cb();
          // wait for all tasks created inside this task
          #pragma omp taskwait
          // signal that the progress task can stop working
          do_progress = 0;
      }
    }
    // wait for the progress task to complete
    printf("Waiting for all tasks to complete %d\n", omp_get_thread_num());
    #pragma omp taskwait
}

int main(int argc, char *argv[])
{
    int numprocs, provided, err;
    int i = 0;
    int po_ret = 0;
    pthread_t progress_tid;

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


    if(err != MPI_SUCCESS) {
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

#if USE_NEW_CONT_API
    MPI_CHECK(MPIX_Continue_init(0, 0, &cont_req, MPI_INFO_NULL));
#else
    MPI_CHECK(MPIX_Continue_init(&cont_req, MPI_INFO_NULL));
    //MPI_CHECK(MPI_Continue_init(&cont_req));
#endif

#if USE_NEW_CONT_API
    MPI_Start(&cont_req);
#endif // USE_NEW_CONT_API

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
        dep_buffer = malloc(num_threads_sender*sizeof(char));
        tags = malloc(num_threads_sender*sizeof(tags[0]));
    } else {
        active_thread_count = options.num_threads;
        dep_buffer = malloc(options.num_threads*sizeof(char));
        tags = malloc(options.num_threads*sizeof(tags[0]));
    }

    if(options.sender_xstreams!=-1) {
        num_xstreams_sender=options.sender_xstreams;
    }

    pthread_create(&progress_tid, NULL, &progress_thread, NULL);

    if(myid == 0) {
        printf("# Number of Sender threads: %d \n# Number of Receiver threads: %d\n",num_threads_sender,options.num_threads );
        printf("# Number of Sender xstreams: %d \n# Number of Receiver xstreams: %d\n",num_xstreams_sender,options.num_xstreams );

        print_header(myid, LAT_MT);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);


        // an OpenMP thread is an execution stream
#pragma omp parallel num_threads(num_xstreams_sender)
{
        //start_with_progress(&send_thread);
        send_thread();
}
    }

    else {

#pragma omp parallel num_threads(options.num_xstreams)
{
        //start_with_progress(&recv_thread);
        recv_thread();
}
    }

    do_progress = 0;

    pthread_join(progress_tid, NULL);

    free(dep_buffer);
    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

#if USE_NEW_CONT_API
static int final_cb(int rc, void *data)
#else
static int final_cb(MPI_Status *status, void *data)
#endif
{
    MPI_Request req;
    thread_tag_t *tag = (thread_tag_t*)data;
    printf("[%d] final_cb tag %d event %p\n", myid, tag->tag, tag->event);
    omp_fulfill_event(tag->event);

    return MPI_SUCCESS;
}

#if USE_NEW_CONT_API
static int receive_thread_recv_cb(int rc, void *data)
#else
static int receive_thread_recv_cb(MPI_Status *status, void *data)
#endif
{
    MPI_Request req;
    thread_tag_t *tag = (thread_tag_t*)data;
    if(options.sender_thread == 1) {
        tag->tag = 2;
    }
    //printf("receive_thread_recv_cb\n");
    printf("[%d] Pong tag %d size %d thread %d \n", myid, tag->tag, tag->size, omp_get_thread_num());
    MPI_Isend (tag->s_buf, tag->size, MPI_CHAR, 0, tag->tag, MPI_COMM_WORLD, &req);
#if USE_NEW_CONT_API
    MPI_CHECK(MPIX_Continue(&req, &final_cb, tag, 0, &tag->status, cont_req));
#else  // MPIX_CONT_REQBUF_VOLATILE
    MPI_CHECK(MPIX_Continue(&req, &final_cb, tag, &tag->status, cont_req));
//    int flag;
//    MPI_CHECK(MPI_Continue(&req, &flag, &receive_thread_recv_cb, tag, &tag->status, cont_req));
//    if (flag) {
//        final_cb(MPI_SUCCESS, tag);
//    }
#endif // MPIX_CONT_REQBUF_VOLATILE

    return MPI_SUCCESS;
}


void recv_thread() {
    int myid;
    char *s_buf, *r_buf;

#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_id);
        return;
    }

    /* touch the data */
    set_buffer_pt2pt(s_buf, myid, options.accel, 'a', options.max_message_size);
    set_buffer_pt2pt(r_buf, myid, options.accel, 'b', options.max_message_size);

    for (int t = thread_id; t < options.num_threads; t += options.num_xstreams) {
        thread_tag_t *tag = &tags[t];
        tag->myid = myid;
        tag->id = t;
        tag->s_buf = s_buf;
        tag->r_buf = r_buf;
    }

    for(size_t size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {

        int finished;

        //printf("Recv size %zu\n", size);

#pragma omp critical
        finished = finished_size++;

        if(finished == options.num_threads) {
            printf("[%d] MPI Barrier\n", myid);
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            finished_size = 1;
        }
        printf("[%d] Waiting for thread barrier %d\n", myid, omp_get_thread_num());
        // wait for all threads to arrive
#pragma omp barrier

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        omp_event_handle_t event;

        for(int i = thread_id, t = thread_id; i < (options.iterations + options.skip); i += options.num_threads, t += options.num_xstreams) {
            if (t >= options.num_threads) {
                t = thread_id;
            }
            // a thread is a stream of serialized tasks
#pragma omp task depend(inout:dep_buffer[t]) detach(event) firstprivate(i, t)
{
            thread_tag_t *tag = &tags[t];
            tag->size = size;
            tag->event = event;
            if(options.sender_thread>1) {
                tag->tag = i;
            } else {
                tag->tag = 1;
            }
            MPI_Request req;
            printf("[%d] Ping tag %d size %d thread %d event %p\n", myid, tag->tag, size, omp_get_thread_num(), event);
            MPI_Irecv(tag->r_buf, size, MPI_CHAR, 0, tag->tag, MPI_COMM_WORLD,
                      &req);
#if USE_NEW_CONT_API
            MPI_CHECK(MPIX_Continue(&req, &receive_thread_recv_cb, tag, MPIX_CONT_REQBUF_VOLATILE, &tag->status, cont_req));
#else  // MPIX_CONT_REQBUF_VOLATILE
            MPI_CHECK(MPIX_Continue(&req, &receive_thread_recv_cb, tag, &tag->status, cont_req));
//            int flag;
//            MPI_CHECK(MPI_Continue(&req, &flag, &receive_thread_recv_cb, tag, &tag->status, cont_req));
//            if (flag) {
//                receive_thread_recv_cb(MPI_SUCCESS, tag);
//            }
#endif // MPIX_CONT_REQBUF_VOLATILE
}
            printf("[%d] TASK tag %d size %d thread %d event %p\n", myid, i, size, omp_get_thread_num(), event);
        }
        // wait for all tasks in this iteration to complete
#pragma omp taskwait

        iter++;
    }

    free_memory(s_buf, r_buf, myid);

}


#if USE_NEW_CONT_API
static int send_thread_send_cb(int rc, void *data)
#else
static int send_thread_send_cb(MPI_Status *status, void *data)
#endif
{
    MPI_Request req;
    thread_tag_t *tag = (thread_tag_t*)data;

    if(options.sender_thread == 1) {
        tag->tag = 2;
    }

    printf("[%d] Pong tag %d size %d thread %d \n", myid, tag->tag, tag->size, omp_get_thread_num());

    MPI_CHECK(MPI_Irecv(tag->r_buf, tag->size, MPI_CHAR, 1, tag->tag, MPI_COMM_WORLD,
              &req));
#if USE_NEW_CONT_API
    MPI_CHECK(MPIX_Continue(&req, &final_cb, tag, MPIX_CONT_REQBUF_VOLATILE, &tag->status, cont_req));
#else  // MPIX_CONT_REQBUF_VOLATILE
    MPI_CHECK(MPIX_Continue(&req, &final_cb, tag, &tag->status, cont_req));
//    int flag;
//    MPI_CHECK(MPI_Continue(&req, &flag, &final_cb, tag, &tag->status, cont_req));
//    if (flag) {
//        final_cb(MPI_SUCCESS, tag);
//    }
#endif // MPIX_CONT_REQBUF_VOLATILE
    return MPI_SUCCESS;
}


void send_thread() {
    int myid;
    char *s_buf, *r_buf;
    double t = 0, latency;

    double t_start = 0, t_end = 0;

    int thread_id = omp_get_thread_num();

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n", myid, thread_id);
        return;
    }

    /* touch the data */
    set_buffer_pt2pt(s_buf, myid, options.accel, 'a', options.max_message_size);
    set_buffer_pt2pt(r_buf, myid, options.accel, 'b', options.max_message_size);

    for (int t = thread_id; t < options.num_threads; t += options.num_xstreams) {
        thread_tag_t *tag = &tags[t];
        tag->myid = myid;
        tag->id = t;
        tag->s_buf = s_buf;
        tag->r_buf = r_buf;
    }

    for(size_t size = options.min_message_size, iter = 0; size <= options.max_message_size; size = (size ? size * 2 : 1)) {

        //printf("Send size %zu\n", size);
        int finished;

#pragma omp critical
        finished = finished_size++;

        if(finished == options.num_threads) {
            printf("[%d] MPI Barrier\n", myid);
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            finished_size = 1;
        }
        printf("[%d] Waiting for thread barrier %d\n", myid, omp_get_thread_num());
        // wait for all threads to arrive
#pragma omp barrier

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        omp_event_handle_t event;

        int flag_print=0;
        for(int i = thread_id, t = thread_id; i < (options.iterations + options.skip); i += num_threads_sender, t += num_xstreams_sender) {
            if (t >= options.num_threads) {
                t = thread_id;
            }
            // a thread is a stream of serialized tasks
#pragma omp task depend(inout:dep_buffer[t]) firstprivate(i, t) default(shared) detach(event)
{
            thread_tag_t *tag = &tags[t];
            if(i == options.skip) {
                t_start = MPI_Wtime();
                flag_print =1;
            }

            tag->size = size;
            tag->event = event;
            if(options.sender_thread>1) {
                tag->tag = i;
            } else {
                tag->tag = 1;
            }
            MPI_Request req;
            printf("[%d] Ping tag %d size %d thread %d event %p\n", myid, tag->tag, size, omp_get_thread_num(), event);
            MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, 1, tag->tag, MPI_COMM_WORLD, &req));
#if USE_NEW_CONT_API
            MPI_CHECK(MPIX_Continue(&req, &send_thread_send_cb, tag, MPIX_CONT_REQBUF_VOLATILE, &tag->status, cont_req));
#else  // MPIX_CONT_REQBUF_VOLATILE
            MPI_CHECK(MPIX_Continue(&req, &send_thread_send_cb, tag, &tag->status, cont_req));
//            int flag;
//            MPI_CHECK(MPI_Continue(&req, &flag, &send_thread_send_cb, tag, &tag->status, cont_req));
//            if (flag) {
//                send_thread_send_cb(MPI_SUCCESS, tag);
//            }
#endif // MPIX_CONT_REQBUF_VOLATILE
}
            printf("[%d] TASK tag %d size %d thread %d event %p\n", myid, i, size, omp_get_thread_num(), event);
        }

#pragma omp taskwait

#pragma omp barrier

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

}

/* vi: set sw=4 sts=4 tw=80: */
