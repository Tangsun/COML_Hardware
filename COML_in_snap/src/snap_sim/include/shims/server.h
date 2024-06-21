/**
 * @file server.h
 * @brief Templated shared memory server for inter-process communication (IPC)
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 27 December 2019
 */

#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <exception>

#include <time.h>
#include <pthread.h>

// POSIX shared memory
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

namespace acl {
namespace ipc {

  template <class MsgData>
  class Server
  {
  public:
    struct msg_t
    {
      MsgData data;

      pthread_mutex_t mutex;
      pthread_cond_t condvar;
    };
  public:
    Server(key_t key)
    {
      // initialize shared memory
      initSHM(key);

      // initialize the mutex and condvar to synchronize
      // across process boundaries for concurrency-safe
      // shared memory read and write.
      initMutex();
    }

    ~Server()
    {
      // n.b. order matters
      deinitMutex();
      deinitSHM();
    }

    void send(const MsgData& msg)
    {
      pthread_mutex_lock(&msg_ptr_->mutex);

      // copy data to memory-mapped file
      std::memcpy(&msg_ptr_->data, &msg, sizeof(MsgData));

      // signal client to read data
      pthread_cond_signal(&msg_ptr_->condvar);

      pthread_mutex_unlock(&msg_ptr_->mutex);
    }

  private:
    int shmid_; ///< shared memory segment id
    msg_t * msg_ptr_ = nullptr;

    bool initSHM(key_t key)
    {
      // create and connect to a shared memory segment (check with 'ipcs -m')
      shmid_ = shmget(key, sizeof(msg_t), IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR);

      // attach to the shared memory segment
      void * ptr = shmat(shmid_, 0, 0);
      msg_ptr_ = reinterpret_cast<msg_t *>(ptr);

      return true;
    }

    void deinitSHM()
    {
      // detach
      shmdt(reinterpret_cast<void *>(msg_ptr_));
    }

    void deinitMutex()
    {
      pthread_mutex_destroy(&msg_ptr_->mutex);
      // If client is killed before server, this will block due to undef behavior
      // pthread_cond_destroy(&msg_ptr_->condvar);
    }

    void initMutex()
    {
      // n.b.: this assumes that msg_ptr_ already points at allocated memory
      //       (e.g., from initMMF).
      //       Also, only the server will init and destroy mutex/condvar

      // mutex setup
      pthread_mutexattr_t mattr;
      pthread_mutexattr_init(&mattr);
      pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
      pthread_mutex_init(&msg_ptr_->mutex, &mattr);

      // conditional variable setup
      pthread_condattr_t cvattr;
      pthread_condattr_init(&cvattr);
      pthread_condattr_setpshared(&cvattr, PTHREAD_PROCESS_SHARED);
      pthread_cond_init(&msg_ptr_->condvar, &cvattr);
    }
  };

} // ns ipc
} // ns acl
