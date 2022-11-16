#include <iostream>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <SYCL/sycl.hpp>
#include <fmt/core.h>

#include <unordered_map>


// Workaround for https://github.com/wolfpld/tracy/issues/426
class tracy_async_lane {
    public:
        void initialize() {
          assert(!m_started);
          m_lane_id = get_free_lane();
          m_started = true;
        }

        void destroy() {
          assert(m_started);
          TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
          if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
          return_lane(m_lane_id);
          TracyFiberLeave;
          m_started = false;
          m_current_zone = std::nullopt;
        }

        void activate() {
          assert(m_started);
          TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
        }

        void deactivate() {
          assert(m_started);
          TracyFiberLeave;
        }

        void begin_phase(const std::string& name, const std::string& description, const tracy::Color::ColorType color) {
          assert(m_started);
          if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
          TracyCZone(t_ctx, true);
          TracyCZoneName(t_ctx, name.c_str(), name.size());
          TracyCZoneText(t_ctx, description.c_str(), description.size());
          TracyCZoneColor(t_ctx, color);
          m_current_zone = t_ctx;
        }

    private:
        bool m_started = false;
        size_t m_lane_id = -1;
        std::optional<TracyCZoneCtx> m_current_zone;

        struct lane_info {
            std::unique_ptr<std::string> name;
            bool is_free;
        };

        inline static std::vector<lane_info> tracy_lanes = {};

        static size_t get_free_lane() {
          for(size_t lane = 0; lane < tracy_lanes.size(); ++lane) {
            if(tracy_lanes[lane].is_free) {
              tracy_lanes[lane].is_free = false;
              return lane;
            }
          }
          tracy_lanes.push_back({std::make_unique<std::string>(fmt::format("celerity async {:02}", tracy_lanes.size())), false});
          return tracy_lanes.size() - 1;
        }

        static void return_lane(size_t lane_id) {
          assert(!tracy_lanes.at(lane_id).is_free);
          tracy_lanes[lane_id].is_free = true;
        }
};

class tracy_sink
    : public sycl::profile::sink {
    public:
        sycl::profile::device_id register_device(std::string name) override {
          const auto did = sycl::profile::device_id(m_devices.size());
          m_devices.push_back(device_state{std::move(name)});
          return did;
        }

        sycl::profile::backend_queue_id register_device_queue(sycl::profile::device_id device, bool in_order) override {
          assert(in_order);
          auto &dev = m_devices.at(size_t(device));
          dev.num_queues += 1;
          const auto bqid = sycl::profile::backend_queue_id(m_backend_queues.size());
          m_backend_queues.push_back(backend_queue_state{fmt::format("{} queue {}", dev.name, dev.num_queues), {}});
          return bqid;
        }

        void set_buffer_name(sycl::profile::identifier buffer_id, std::string name) override {

        }

        void task_submit_begin(sycl::profile::identifier task_id) override {
          auto &state = m_threads[std::this_thread::get_id()];
          TracyCZoneN(ctx, "submit", true);
          state.zone_ctx = ctx;
        }

        void task_submit_end(sycl::profile::task task) override {
          auto &state = m_threads.at(std::this_thread::get_id());
          TracyCZoneEnd(state.zone_ctx);
        }

        void task_schedule_begin(sycl::profile::identifier task_id) override {
          auto &state = m_threads[std::this_thread::get_id()];
          TracyCZoneN(ctx, "schedule", true);
          state.zone_ctx = ctx;
        }

        void task_schedule_end(sycl::profile::identifier task_id) override {
          auto &state = m_threads.at(std::this_thread::get_id());
          TracyCZoneEnd(state.zone_ctx);
        }

        void task_execute_begin(sycl::profile::backend_queue_id bqid, sycl::profile::identifier task_id) override {
          auto &state = m_backend_queues.at(size_t(bqid));
          TracyFiberEnter(state.name.c_str());
          TracyCZoneN(ctx, "execute", true);
          state.zone_ctx = ctx;
        }

        void task_execute_end(sycl::profile::backend_queue_id bqid, sycl::profile::identifier task_id) override {
          auto &state = m_backend_queues.at(size_t(bqid));
          TracyCZoneEnd(state.zone_ctx);
          TracyFiberLeave;
        }

        void transfer_begin(sycl::profile::backend_queue_id bqid, sycl::profile::transfer transfer) override {

        }

        void transfer_end(sycl::profile::backend_queue_id bqid, sycl::profile::identifier transfer_id) override {

        }

        void host_access_request(sycl::profile::host_access access) override {

        }

        void host_access_begin(sycl::profile::identifier access_id) override {

        }

        void host_access_end(sycl::profile::identifier access_id) override {

        }

        void wait_begin() override {
          auto &state = m_threads[std::this_thread::get_id()];
          TracyCZoneN(ctx, "wait", true);
          state.zone_ctx = ctx;
        }

        void wait_begin(std::vector<sycl::profile::identifier> dependencies) override {
          wait_begin();
        }

        void wait_end() override {
          auto &state = m_threads.at(std::this_thread::get_id());
          TracyCZoneEnd(state.zone_ctx);
        }

        void idle_begin(sycl::profile::device device) override {

        }

        void idle_end(sycl::profile::device device) override {

        }

    private:
        struct device_state {
            std::string name;
            size_t num_queues = 0;
        };
        struct backend_queue_state {
            std::string name;
            TracyCZoneCtx zone_ctx{};
        };
        struct thread_state {
            TracyCZoneCtx zone_ctx{};
        };

        std::vector<device_state> m_devices;
        std::vector<backend_queue_state> m_backend_queues;
        std::unordered_map<std::thread::id, thread_state> m_threads;
};


int main() {
  sycl::profile::the_sink = new tracy_sink;

  sycl::queue q;
  constexpr size_t N = 1024;

  sycl::buffer<float, 2> buf_a(sycl::range<2>(N, N));
  sycl::buffer<float, 2> buf_b(sycl::range<2>(N, N));
  sycl::buffer<float, 2> buf_c(sycl::range<2>(N, N));

  for (size_t k = 0; k < 3; ++k) {
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor a(buf_a, cgh, sycl::read_only);
        sycl::accessor b(buf_b, cgh, sycl::write_only, sycl::no_init);
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

  sleep(1);

  return 0;
}
