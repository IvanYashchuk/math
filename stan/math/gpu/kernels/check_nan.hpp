#ifndef STAN_MATH_GPU_KERNELS_CHECK_NAN_HPP
#define STAN_MATH_GPU_KERNELS_CHECK_NAN_HPP
#ifdef STAN_OPENCL

#include <stan/math/gpu/kernel_cl.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
const char *is_nan_kernel_code = STRINGIFY(
    // \endcond
    /**
     * Check if the <code>matrix_gpu</code> has NaN values
     *
     * @param[in] A The matrix to check.
     * @param rows The number of rows in matrix A.
     * @param cols The number of columns in matrix A.
     * @param[out] flag the flag to be written to if any diagonal is zero.
     * @note Code is a <code>const char*</code> held in
     * <code>is_nan_kernel_code.</code>
     *  Kernel for stan/math/gpu/err/check_nan.hpp.
     *  This kernel uses the helper macros available in helpers.cl.
     */
    __kernel void is_nan(
        __global read_only double *A, __global write_only int *flag,
        read_only unsigned int rows, read_only unsigned int cols) {
      const int i = get_global_id(0);
      const int j = get_global_id(1);
      if (i < rows && j < cols) {
        if (isnan(A(i, j))) {
          flag[0] = 1;
        }
      }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/check_nan.hpp is_nan() \endlink
 */
const global_range_kernel<cl::Buffer, cl::Buffer, int, int> check_nan(
    "is_nan", is_nan_kernel_code);

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
