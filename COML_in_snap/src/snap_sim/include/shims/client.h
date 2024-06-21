/**
 * @file client.h
 * @brief Templated shared memory client for inter-process communication (IPC)
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 27 December 2019
 */

#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <thread>
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
  class Client
  {
  public:
    struct msg_t
    {
      MsgData data;

      pthread_mutex_t mutex;
      pthread_cond_t condvar;
    };
  public:
    Client(key_t key)
    {
      if (!init(key)) {
        std::throw_with_nested(std::runtime_error("IPC Client: could not initialize"));
      }
    }

    ~Client()
    {
      // detach
      shmdt(reinterpret_cast<void *>(msg_ptr_));
    }

    bool read(MsgData * msg)
    {
      struct timespec timeout;
      clock_gettime(CLOCK_REALTIME, &timeout);
      timeout.tv_sec += TIMEOUT_SEC;

      // lock -- predicate (the mutex / condvar) -- unlock pattern
      pthread_mutex_lock(&msg_ptr_->mutex);
      int ret = pthread_cond_timedwait(&msg_ptr_->condvar, &msg_ptr_->mutex, &timeout);

      // we timed out!
      if (ret != 0) {
        pthread_mutex_unlock(&msg_ptr_->mutex);
        return false;
      }

      // copy data locally
      std::memcpy(msg, &msg_ptr_->data, sizeof(MsgData));

      pthread_mutex_unlock(&msg_ptr_->mutex);
      return true;
    }

  private:
    int shmid_ = -1;
    msg_t * msg_ptr_ = nullptr;

    static constexpr int TIMEOUT_SEC = 3; ///< read timeout

    bool init(key_t key)
    {
      while (shmid_ == -1) {
        // connect to an already created shared memory segment (use 'ipcs -m')
        shmid_ = shmget(key, sizeof(msg_t), /*IPC_CREAT |*/ S_IRUSR | S_IWUSR);

        if (shmid_ == -1) {
          std::cout << "Waiting on server to initialize shared memory..." << std::endl;
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
      }

      // attach to the shared memory segment
      void * ptr = shmat(shmid_, 0, 0);
      msg_ptr_ = reinterpret_cast<msg_t *>(ptr);

      // we don't want anyone else to connect; mark shmem for destruction
      shmctl(shmid_, IPC_RMID, NULL);

      return true;
    }
  };

} // ns ipc
} // ns acl
