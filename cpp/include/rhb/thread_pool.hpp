// Minimal fork/join thread pool: run(fn) executes fn(chunk_index) on every
// chunk, with the calling thread taking chunk 0. Workers are persistent, so a
// batched step costs two condition-variable handshakes rather than thread spawns.
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rhb {

class ThreadPool {
public:
    explicit ThreadPool(int n) : n_(n < 1 ? 1 : n) {
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return n_; }

    // Runs fn(i) for i in [0, size()) and blocks until all are done.
    void run(const std::function<void(int)>& fn) {
        if (n_ == 1) {
            fn(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            task_ = &fn;
            pending_ = n_ - 1;
            ++generation_;
        }
        start_cv_.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        task_ = nullptr;
    }

private:
    void loop(int idx) {
        unsigned long seen = 0;
        for (;;) {
            const std::function<void(int)>* task;
            {
                std::unique_lock<std::mutex> lk(m_);
                start_cv_.wait(lk, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
                task = task_;
            }
            (*task)(idx);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable start_cv_, done_cv_;
    const std::function<void(int)>* task_ = nullptr;
    int pending_ = 0;
    unsigned long generation_ = 0;
    bool stop_ = false;
};

}  // namespace rhb
