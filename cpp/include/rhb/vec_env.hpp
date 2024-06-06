// Batched, multi-threaded environment. All buffers live in C++ and are exposed
// to NumPy zero-copy; step() partitions envs into contiguous chunks, one per
// thread, with no shared mutable state between chunks.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rhb/gridworld.hpp"
#include "rhb/thread_pool.hpp"

namespace rhb {

class VecEnv {
public:
    VecEnv(int num_envs, int num_threads, const EnvConfig& cfg, uint64_t seed)
        : n_(num_envs), cfg_(cfg), pool_(num_threads),
          envs_(num_envs),
          obs_(static_cast<size_t>(num_envs) * cfg.obs_dim(), 0.f),
          proxy_(num_envs, 0.f), true_(num_envs, 0.f),
          done_(num_envs, 0), stats_(static_cast<size_t>(num_envs) * kNumStats, 0.f),
          visited_(static_cast<size_t>(num_envs) * cfg.cells(), 0) {
        if (num_envs < 1) throw std::invalid_argument("num_envs must be >= 1");
        if (cfg.size < 3) throw std::invalid_argument("size must be >= 3");
        if (cfg.variant < 0 || cfg.variant > 2) throw std::invalid_argument("variant must be 0, 1 or 2");
        if (cfg.max_steps < 1) throw std::invalid_argument("max_steps must be >= 1");
        for (int i = 0; i < n_; ++i) {
            envs_[i].rng = Rng(seed * 0x100000001B3ULL + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL + 1);
            envs_[i].visited = &visited_[static_cast<size_t>(i) * cfg.cells()];
        }
        reset();
    }

    void reset() {
        run_chunks([this](int b, int e) {
            for (int i = b; i < e; ++i) {
                envs_[i].reset(cfg_, obs_row(i));
                proxy_[i] = true_[i] = 0.f; done_[i] = 0;
            }
        });
    }

    // actions: num_envs ints in [0, kNumActions). Finished envs auto-reset; their
    // statistics are in stats row i and done[i]==1, obs holds the new episode.
    void step(const int32_t* actions) {
        for (int i = 0; i < n_; ++i)
            if (actions[i] < 0 || actions[i] >= kNumActions)
                throw std::out_of_range("action " + std::to_string(actions[i]) + " out of range");
        run_chunks([this, actions](int b, int e) {
            for (int i = b; i < e; ++i) {
                float* o = obs_row(i);
                const bool fin = envs_[i].step(cfg_, actions[i], o, proxy_[i], true_[i]);
                done_[i] = fin ? 1 : 0;
                if (fin) {
                    envs_[i].write_stats(cfg_, &stats_[static_cast<size_t>(i) * kNumStats]);
                    envs_[i].reset(cfg_, o);
                }
            }
        });
    }

    // Force env i into an explicit layout (for tests / parity checks).
    void set_layout(int i, int agent, int goal, int sparkle) {
        const int n = cfg_.cells();
        if (i < 0 || i >= n_) throw std::out_of_range("env index");
        if (agent < 0 || agent >= n || goal < 0 || goal >= n || agent == goal ||
            (cfg_.variant == kSparkle && (sparkle < 0 || sparkle >= n || sparkle == agent || sparkle == goal)))
            throw std::invalid_argument("invalid layout");
        envs_[i].start(cfg_, agent, goal, sparkle, obs_row(i));
        proxy_[i] = true_[i] = 0.f; done_[i] = 0;
    }

    int num_envs() const { return n_; }
    int num_threads() const { return pool_.size(); }
    const EnvConfig& config() const { return cfg_; }
    float* obs() { return obs_.data(); }
    float* proxy_reward() { return proxy_.data(); }
    float* true_reward() { return true_.data(); }
    uint8_t* done() { return done_.data(); }
    float* stats() { return stats_.data(); }
    int agent_pos(int i) const { return envs_.at(i).agent; }
    int goal_pos(int i) const { return envs_.at(i).goal; }
    int sparkle_pos(int i) const { return envs_.at(i).sparkle; }

private:
    float* obs_row(int i) { return &obs_[static_cast<size_t>(i) * cfg_.obs_dim()]; }

    template <class F>
    void run_chunks(F&& f) {
        const int T = pool_.size();
        std::function<void(int)> task = [&](int k) {
            const int b = static_cast<int>(static_cast<long long>(n_) * k / T);
            const int e = static_cast<int>(static_cast<long long>(n_) * (k + 1) / T);
            f(b, e);
        };
        pool_.run(task);
    }

    int n_;
    EnvConfig cfg_;
    ThreadPool pool_;
    std::vector<GridEnv> envs_;
    std::vector<float> obs_, proxy_, true_;
    std::vector<uint8_t> done_;
    std::vector<float> stats_;
    std::vector<uint8_t> visited_;
};

}  // namespace rhb
