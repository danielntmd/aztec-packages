#include "bn254_test_utils.hpp"

#ifdef BB_GPU_NATIVE

#include "barretenberg/gpu/common/device_buffer.hpp"
#include "common/gpu_msm_context.hpp"

#include <cstring>
#include <vector>

namespace {

using namespace bb;
using namespace bb::gpu;
using namespace bb::gpu::bn254;
namespace gpu_testing = bb::gpu::bn254::testing;

TEST(GpuBn254, DeviceBufferCopiesRoundTrip) {
  BB_REQUIRE_CUDA_DEVICE();

  std::vector<fq32_t> input = {gpu_testing::to_fq32_standard(fq::zero()),
                               gpu_testing::to_fq32_standard(fq::one()),
                               gpu_testing::to_fq32_standard(fq(17))};
  std::vector<fq32_t> output(input.size());
  DeviceBuffer<fq32_t> buffer;

  copy_to_device(buffer, std::span<const fq32_t>(input.data(), input.size()),
                 default_msm_context().stream());
  copy_to_host(std::span<fq32_t>(output.data(), output.size()), buffer,
               default_msm_context().stream());
  default_msm_context().sync();

  EXPECT_EQ(
      std::memcmp(input.data(), output.data(), sizeof(fq32_t) * input.size()),
      0);
}

} // namespace

#endif // BB_GPU_NATIVE
