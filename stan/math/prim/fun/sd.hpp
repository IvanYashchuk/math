#ifndef STAN_MATH_PRIM_FUN_SD_HPP
#define STAN_MATH_PRIM_FUN_SD_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/variance.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <vector>
#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the unbiased sample standard deviation of the
 * coefficients in the specified std vector, column vector, row vector, or
 * matrix.
 *
 * @tparam T type of the container
 *
 * @param m Specified container.
 * @return Sample variance.
 */
template <typename T, require_container_t<T>* = nullptr,
          require_not_vt_var<T>* = nullptr>
inline return_type_t<T> sd(const T& m) {
  using std::sqrt;
  check_nonzero_size("sd", "m", m);
  if (m.size() == 1) {
    return 0.0;
  }
  return sqrt(variance(m));
}

}  // namespace math
}  // namespace stan

#endif
