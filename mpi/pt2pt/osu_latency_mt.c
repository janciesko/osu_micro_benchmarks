#define BENCHMARK "OSU MPI%s Multi-threaded Latency Test"
/*
 * Copyright (C) 2002-2021 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>

#define CSV_OUTPUT

#ifndef OSU_USE_QTHREADS

typedef pthread_mutex_t thread_mutex_t;
typedef pthread_cond_t thread_cond_t;
typedef pthread_barrier_t thread_barrier_t;
typedef pthread_t thread_t;
typedef void ret_type;

void init_threading()
{
    ;
}
void finalize_threading()
{
    ;
}

int thread_mutex_init(thread_mutex_t *mutex, const void *attr)
{
    return pthread_mutex_init(mutex, (const pthread_mutexattr_t *)attr);
}

int thread_cond_init(thread_cond_t *cond, const void *attr)
{
    return pthread_cond_init(cond, (const pthread_condattr_t *)attr);
}

int thread_barrier_init(thread_barrier_t *barrier, const void *attr,
                        unsigned count)
{
    return pthread_barrier_init(barrier, (const pthread_barrierattr_t *)attr,
                                count);
}

int thread_create(thread_t *thread, const void *attr,
                  void *(*start_routine)(void *), void *arg)
{
    return pthread_create(thread, (const pthread_attr_t *)attr, start_routine,
                          arg);
}

int thread_join(thread_t thread, void **retval)
{
    return pthread_join(thread, retval);
}

int thread_mutex_lock(thread_mutex_t *mutex)
{
    return pthread_mutex_lock(mutex);
}

int thread_mutex_unlock(thread_mutex_t *mutex)
{
    return pthread_mutex_unlock(mutex);
}

int thread_cond_broadcast(thread_cond_t *cond)
{
    return pthread_cond_broadcast(cond);
}

int thread_cond_wait(thread_cond_t *cond, thread_mutex_t *mutex)
{
    return pthread_cond_wait(cond, mutex);
}

int thread_barrier_wait(thread_barrier_t *barrier)
{
    return pthread_barrier_wait(barrier);
}

#else

#include <qthread.h>
#include <qthread/barrier.h>

typedef aligned_t atomic_int32_t;
typedef atomic_int32_t thread_lock_t;
typedef atomic_int32_t thread_mutex_t;
typedef atomic_int32_t thread_cond_t;
typedef aligned_t ret_type;

typedef struct {
    aligned_t retval;
    aligned_t * retval_ptr;
} qt_thread_t;

typedef qt_barrier_t thread_barrier_t;
typedef qt_thread_t thread_t;
#define SUCCESS 1

void init_threading()
{
    qthread_initialize();
}

void finalize_threading()
{
    qthread_finalize ();
}

int atomic_lock_init(thread_lock_t *lock, const void *attr)
{
    * lock = 0;
    return SUCCESS;
}

int thread_mutex_init(thread_mutex_t *mutex, const void *attr)
{
    * mutex = 0;
    return SUCCESS;
}

int thread_cond_init(thread_cond_t *cond, const void *attr)
{
    qthread_empty(cond);
    return SUCCESS;
}

int thread_barrier_init(thread_barrier_t ** barrier, const void *attr,
                        unsigned count)
{
    * barrier = qt_barrier_create(count, REGION_BARRIER);
}

int thread_create(thread_t *thread, const void *attr,
                   qthread_f task, void *arg)
{
    thread->retval_ptr = &thread->retval;
    qthread_fork(task, arg, thread->retval_ptr);
    return SUCCESS;
}

int thread_join(thread_t thread, void **retval)
{
    qthread_readFF(NULL, thread.retval_ptr);
    if (retval)
        * retval = thread.retval_ptr;
    return SUCCESS;
}

int thread_mutex_lock(thread_mutex_t *mutex)
{
    qthread_lock(mutex);
    return SUCCESS;
}

int thread_mutex_unlock(thread_mutex_t *mutex)
{
    qthread_unlock(mutex);
    return SUCCESS;
}

int thread_cond_broadcast(thread_cond_t *cond)
{
    qthread_writeF(cond, cond);
    qthread_empty(cond);
    return SUCCESS;
}

int thread_cond_wait(thread_cond_t *cond, thread_mutex_t *lock)
{
    qthread_unlock(lock);
    qthread_readFF(cond, cond);
    qthread_lock(lock);
    return SUCCESS;    
}

int thread_barrier_wait(thread_barrier_t ** barrier)
{
    qt_barrier_enter(*barrier);
    return SUCCESS;    
}

#endif

thread_barrier_t * sender_barrier;
thread_barrier_t * all_barrier;
thread_mutex_t finished_size_mutex;
thread_cond_t finished_size_cond;
thread_mutex_t finished_size_sender_mutex;
thread_cond_t finished_size_sender_cond;

double t_start = 0, t_end = 0;

int finished_size = 0;
int finished_size_sender = 0;

int num_threads_sender = 1;
typedef struct thread_tag  {
        int id;
} thread_tag_t;

ret_type send_thread(void *arg);
ret_type recv_thread(void *arg);

int main(int argc, char *argv[])
{
    init_threading();
    int numprocs = 0, provided = 0, myid = 0, err = 0;
    int i = 0;
    int po_ret = 0;
    thread_t *sr_threads = (thread_t *)malloc(sizeof(thread_t) * MAX_NUM_THREADS);
    thread_tag_t *tags = (thread_tag_t *)malloc(sizeof(thread_tag_t) * MAX_NUM_THREADS);

    thread_mutex_init(&finished_size_mutex, NULL);
    thread_cond_init(&finished_size_cond, NULL);
    thread_mutex_init(&finished_size_sender_mutex, NULL);
    thread_cond_init(&finished_size_sender_cond, NULL);

    options.bench = PT2PT;
    options.subtype = LAT_MT;

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

    if (err != MPI_SUCCESS) {
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

    if (numprocs != 2) {
        if (myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    /* Check to make sure we actually have a thread-safe
     * implementation
     */

    finished_size = 1;
    finished_size_sender = 1;

    if (provided != MPI_THREAD_MULTIPLE) {
        if (myid == 0) {
            fprintf(stderr,
                    "MPI_Init_thread must return MPI_THREAD_MULTIPLE!\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    if (options.sender_thread != -1) {
        num_threads_sender = options.sender_thread;
    }

    thread_barrier_init(&sender_barrier, NULL, num_threads_sender);
    thread_barrier_init(&all_barrier, NULL, num_threads_sender + options.num_threads);

    if (myid == 0) {
        #ifdef CSV_OUTPUT
        #else
        printf("# Number of Sender threads: %d \n# Number of Receiver threads: "
            "%d\n",
            num_threads_sender, options.num_threads);
        print_header(myid, LAT_MT);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
        #endif

        for (i = 0; i < num_threads_sender; i++) {
            tags[i].id = i;
            thread_create(&sr_threads[i], NULL, send_thread, &tags[i]);
        }
        for (i = 0; i < num_threads_sender; i++) {
            thread_join(sr_threads[i], NULL);
        }
    } else {
        for (i = 0; i < options.num_threads; i++) {
            tags[i].id = i;
            thread_create(&sr_threads[i], NULL, recv_thread, &tags[i]);
        }

        for (i = 0; i < options.num_threads; i++) {
            thread_join(sr_threads[i], NULL);
        }
    }

   // thread_barrier_wait(&all_barrier); //FIXME
    MPI_CHECK(MPI_Finalize());
    finalize_threading();

    return EXIT_SUCCESS;
}

ret_type recv_thread(void *arg)
{
    int size = 0, i = 0, val = 0;
    int iter = 0;
    int myid = 0;

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
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n",
                myid, thread_id->id);
        return 1;
    }
    
    for (size = options.min_message_size, iter = 0;
         size <= options.max_message_size; size = (size ? size * 2 : 1)) {

        thread_mutex_lock(&finished_size_mutex);

        //Last thread to enter syncs procs
        if (finished_size == options.num_threads) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            finished_size = 1;

            thread_mutex_unlock(&finished_size_mutex);
            thread_cond_broadcast(&finished_size_cond);
        }

        else {
            finished_size++;
            thread_cond_wait(&finished_size_cond, &finished_size_mutex);           
            thread_mutex_unlock(&finished_size_mutex);
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        fflush(stdout);

        for (i = val; i < (options.iterations + options.skip);
             i += options.num_threads) {
            if (options.sender_thread > 1) {
                MPI_Recv(r_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD,
                         &reqstat[val]);
                MPI_Send(s_buf, size, MPI_CHAR, 0, i, MPI_COMM_WORLD);
            } else {
                MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD,
                         &reqstat[val]);
                MPI_Send(s_buf, size, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
            }
        }
        iter++;
    }

    free_memory(s_buf, r_buf, myid);

    sleep(1);

    return 0;
}

ret_type send_thread(void *arg)
{
    int size = 0, i = 0, val = 0, iter = 0;
    int myid = 0;
    char *s_buf, *r_buf;
    double t = 0, latency = 0;
    thread_tag_t *thread_id = (thread_tag_t *)arg;
    val = thread_id->id;

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (NONE != options.accel && init_accel()) {
        fprintf(stderr, "Error initializing device\n");
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        fprintf(stderr, "Error allocating memory on Rank %d, thread ID %d\n",
                myid, thread_id->id);
        return 1;
    }

    for (size = options.min_message_size, iter = 0;
         size <= options.max_message_size; size = (size ? size * 2 : 1)) {
         thread_mutex_lock(&finished_size_sender_mutex);

        //Last thread to enter syncs procs
        if (finished_size_sender == num_threads_sender) {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            finished_size_sender = 1;

            thread_mutex_unlock(&finished_size_sender_mutex);
            thread_cond_broadcast(&finished_size_sender_cond);
        } else {
            finished_size_sender++;

            thread_cond_wait(&finished_size_sender_cond,
                             &finished_size_sender_mutex);
            thread_mutex_unlock(&finished_size_sender_mutex);
        }

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        /* touch the data */
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        int flag_print = 0;
        for (i = val; i < options.iterations + options.skip;
             i += num_threads_sender) {
            if (i == options.skip) {
                t_start = MPI_Wtime();
                flag_print = 1;
            }

            if (options.sender_thread > 1) {
                MPI_CHECK(
                    MPI_Send(s_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, i, MPI_COMM_WORLD,
                                  &reqstat[val]));
            } else {
                MPI_CHECK(
                    MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, 2, MPI_COMM_WORLD,
                    &reqstat[val]));
            }
        }

        thread_barrier_wait(&sender_barrier);

        if (flag_print == 1) {
            t_end = MPI_Wtime();
            t = t_end - t_start;

            latency =
                (t)*1.0e6 / (2.0 * options.iterations / num_threads_sender);
            #ifdef CSV_OUTPUT
            #ifdef OSU_USE_QTHREADS
            fprintf(stdout, "%i,%i,%i,%f\n",options.num_threads, qthread_num_shepherds(), size, latency);
            #else //OSU_USE_QTHREADS
            fprintf(stdout, "%i,%i,%f\n",options.num_threads, size, latency);
            #endif //OSU_USE_QTHREADS
            #else
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION, latency);
            #endif
                       
            fflush(stdout);
        }
        iter++;
    }

    free_memory(s_buf, r_buf, myid);

    return 0;
}

/* vi: set sw=4 sts=4 tw=80: */