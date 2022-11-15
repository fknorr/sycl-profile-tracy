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
          TracyFiberEnter(tracy_lanes[m_lane_id].name->c_str());
          if(m_current_zone.has_value()) { TracyCZoneEnd(*m_current_zone); }
          TracyCZone(t_ctx, true);
          TracyCZoneName(t_ctx, name.c_str(), name.size());
          TracyCZoneText(t_ctx, description.c_str(), description.size());
          TracyCZoneColor(t_ctx, color);
          TracyFiberLeave;
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
        void register_device(sycl::profile::identifier device_id, const sycl::device &device) override {

        }

        void set_buffer_name(sycl::profile::identifier buffer_id, std::string name) override {

        }

        void task_submit(sycl::profile::task task) override {
          tracy_async_lane lane;
          lane.initialize();
          lane.begin_phase("submit task", fmt::format("T{}", task.id), tracy::Color::Aqua);
          m_lanes.emplace(task.id, lane);
        }

        void task_begin_execute(sycl::profile::identifier task_id) override {
          auto &lane = m_lanes.at(task_id);
          lane.begin_phase("execute task", fmt::format("T{}", task_id), tracy::Color::Coral);
        }

        void task_end_execute(sycl::profile::identifier task_id) override {
          auto &lane = m_lanes.at(task_id);
          lane.destroy();
        }

        void transfer_begin(sycl::profile::transfer transfer) override {

        }

        void transfer_end(sycl::profile::identifier transfer_id) override {

        }

        void host_access_request(sycl::profile::host_access access) override {

        }

        void host_access_begin(sycl::profile::identifier access_id) override {

        }

        void host_access_end(sycl::profile::identifier access_id) override {

        }

        void wait_begin() override {
          TracyCZoneN(ctx, "wait", true)
          m_wait_ctx = ctx;
        }

        void wait_begin(std::vector<sycl::profile::identifier> dependencies) override {
          TracyCZoneN(ctx, "wait", true)
          m_wait_ctx = ctx;
        }

        void wait_end() override {
          TracyCZoneEnd(m_wait_ctx)
        }

        void idle_begin(sycl::profile::device device) override {

        }

        void idle_end(sycl::profile::device device) override {

        }

    private:
        std::unordered_map<sycl::profile::identifier, tracy_async_lane> m_lanes;
        ___tracy_c_zone_context m_wait_ctx;
};


int main() {
  sycl::profile::the_sink = new tracy_sink;

  sycl::queue q;
  constexpr size_t N = 256;

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

  return 0;
}
