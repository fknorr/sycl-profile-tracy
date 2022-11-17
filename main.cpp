#include <iostream>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <SYCL/sycl.hpp>
#include <fmt/core.h>

#include <unordered_map>
#include <stack>


class tracy_sink
    : public sycl::profile::sink {
    public:
        void register_backend_queue(sycl::profile::backend_queue_id id, std::string name, bool in_order) override {
          assert(in_order);
          m_backend_queues[id] = backend_queue_state{std::move(name), {}};
        }

        void unregister_backend_queue(sycl::profile::backend_queue_id id) override {
          m_backend_queues.erase(id);
        }

        void register_command_group(sycl::profile::command_group_id id, std::optional<std::string> name) override {
          m_command_groups[id] = command_group_info{std::move(name)};
        }

        void unregister_command_group(sycl::profile::command_group_id id) override {
          m_command_groups.erase(id);
        }

        void register_runtime_thread(std::string name) override {
          TracyCSetThreadName(name.c_str());
          m_threads[std::this_thread::get_id()] = thread_state{std::move(name), {}};
        }

        void unregister_runtime_thread() override {
          m_threads.erase(std::this_thread::get_id());
        }

        void
        frontend_thread_begin(sycl::profile::frontend_operation operation,
            std::vector<sycl::profile::command_group_id> cgs) override {
          TracyCZone(ctx, true);
          const auto name = sycl::profile::frontend_operation_string(operation);
          TracyCZoneName(ctx, name.data(), name.size());
          auto &zones = m_threads[std::this_thread::get_id()].active_zones;
          zones.push(zone{ctx, std::move(cgs)});
        }

        void frontend_thread_end() override {
          auto &zones = m_threads[std::this_thread::get_id()].active_zones;
          std::string zone_text;
          for (auto cgid: zones.top().cgs) {
            if (auto cg = m_command_groups.find(cgid); cg != m_command_groups.end()) {
              if (auto &name = cg->second.name; name.has_value()) {
                zone_text += *name;
                zone_text.push_back('\n');
              }
            } else {
            }
          }

          TracyCZoneText(zones.top().ctx, zone_text.data(), zone_text.size());
          TracyCZoneEnd(zones.top().ctx);
          zones.pop();
        }

        void
        runtime_thread_begin(sycl::profile::runtime_operation operation,
            std::vector<sycl::profile::command_group_id> cgs) override {
          TracyCZone(ctx, true);
          const auto name = sycl::profile::runtime_operation_name(operation);
          TracyCZoneName(ctx, name.data(), name.size());
          auto &zones = m_threads[std::this_thread::get_id()].active_zones;
          zones.push(zone{ctx, std::move(cgs)});
        }

        void runtime_thread_end() override {
          frontend_thread_end();
        }

        void backend_queue_begin(sycl::profile::backend_queue_id id, sycl::profile::backend_operation operation,
            std::vector<sycl::profile::command_group_id> cgs) override {
          TracyCFiberEnter(m_backend_queues.at(id).name.c_str());
          TracyCZone(ctx, true);
          const auto name = sycl::profile::backend_operation_string(operation);
          TracyCZoneName(ctx, name.data(), name.size());
          auto &zones = m_backend_queues[id].active_zones;
          zones.push(zone{ctx, std::move(cgs)});
          TracyFiberLeave;
        }

        void backend_queue_end(sycl::profile::backend_queue_id id) override {
          TracyCFiberEnter(m_backend_queues.at(id).name.c_str());

          auto &zones = m_backend_queues[id].active_zones;
          std::string zone_text;
          for (auto cgid: zones.top().cgs) {
            if (auto cg = m_command_groups.find(cgid); cg != m_command_groups.end()) {
              if (auto &name = cg->second.name; name.has_value()) {
                zone_text += *name;
                zone_text.push_back('\n');
              }
            } else {
            }
          }

          TracyCZoneText(zones.top().ctx, zone_text.data(), zone_text.size());
          TracyCZoneEnd(zones.top().ctx);
          zones.pop();

          TracyFiberLeave;
        }

    private:
        struct zone {
            TracyCZoneCtx ctx{};
            std::vector<sycl::profile::command_group_id> cgs;
        };
        struct backend_queue_state {
            std::string name;
            std::stack<zone> active_zones;
        };
        struct thread_state {
            std::optional<std::string> name;
            TracyCZoneCtx zone_ctx{};
            std::stack<zone> active_zones;
        };
        struct command_group_info {
            std::optional<std::string> name;
        };

        std::unordered_map<sycl::profile::backend_queue_id, backend_queue_state> m_backend_queues;
        std::unordered_map<sycl::profile::command_group_id, command_group_info> m_command_groups;
        std::unordered_map<std::thread::id, thread_state> m_threads;
};


int main() {
  sycl::profile::the_sink = new tracy_sink;

  sycl::queue q;
  constexpr size_t N = 1024;

  TracyCZoneN(malloc_ab_ctx, "malloc", true);
  std::vector<float> in_a(N * N);
  std::vector<float> in_b(N * N);
  TracyCZoneEnd(malloc_ab_ctx);

  TracyCZoneN(init_ctx, "init", true);
  for (int i = 0; i < N; ++i) {
    in_a[i * N + i] = 2;
    in_b[i * N + i] = 3;
  }
  TracyCZoneEnd(init_ctx);

  sycl::buffer<float, 2> buf_a(in_a.data(), sycl::range<2>(N, N));
  sycl::buffer<float, 2> buf_b(in_b.data(), sycl::range<2>(N, N));
  sycl::buffer<float, 2> buf_c(sycl::range<2>(N, N));

  for (size_t n = 0; n < 2; ++n) {
    for (size_t k = 0; k < 3; ++k) {
      q.submit([&](sycl::handler &cgh) {
          sycl::accessor a(buf_a, cgh, sycl::read_only);
          sycl::accessor b(buf_b, cgh, sycl::read_only);
          sycl::accessor c(buf_c, cgh, sycl::write_only, sycl::no_init);
          cgh.parallel_for(sycl::range<2>(N, N), [=](sycl::item<2> it) {
              c[it] = 0;
              for (size_t i = 0; i < N; ++i) {
                c[it] += a[{i, it[1]}] * b[{it[0], i}];
              }
          });
      });
    }
    q.wait();
  }

  TracyCZoneN(malloc_c_ctx, "malloc", true);
  std::vector<float> out_c(N * N);
  TracyCZoneEnd(malloc_c_ctx);

  q.submit([&](sycl::handler &cgh) {
    cgh.copy(sycl::accessor{buf_a, cgh, sycl::read_only}, out_c.data());
  }).wait();

  ZoneScopedN("sleep");
  sleep(1);

  return 0;
}
