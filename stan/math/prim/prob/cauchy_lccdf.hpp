#ifndef STAN_MATH_PRIM_PROB_CAUCHY_LCCDF_HPP
#define STAN_MATH_PRIM_PROB_CAUCHY_LCCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/operands_and_partials.hpp>
#include <cmath>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Returns the cauchy log complementary cumulative distribution function
 * for the given location, and scale. If given containers of matching sizes
 * returns the log sum of probabilities.
 *
 * @tparam T_y type of real parameter
 * @tparam T_loc type of location parameter
 * @tparam T_scale type of scale parameter
 * @param y real parameter
 * @param mu location parameter
 * @param sigma scale parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if sigma is nonpositive or y, mu are nan
 * @throw std::invalid_argument if container sizes mismatch
 */
template <typename T_y, typename T_loc, typename T_scale>
return_type_t<T_y, T_loc, T_scale> cauchy_lccdf(const T_y& y, const T_loc& mu,
                                                const T_scale& sigma) {
  using T_partials_return = partials_return_t<T_y, T_loc, T_scale>;
  using std::atan;
  using std::log;
  static const char* function = "cauchy_lccdf";
  check_not_nan(function, "Random variable", y);
  check_finite(function, "Location parameter", mu);
  check_positive_finite(function, "Scale parameter", sigma);
  check_consistent_sizes(function, "Random variable", y, "Location parameter",
                         mu, "Scale Parameter", sigma);

  if (size_zero(y, mu, sigma)) {
    return 0;
  }

  T_partials_return ccdf_log(0.0);
  operands_and_partials<T_y, T_loc, T_scale> ops_partials(y, mu, sigma);

  scalar_seq_view<T_y> y_vec(y);
  scalar_seq_view<T_loc> mu_vec(mu);
  scalar_seq_view<T_scale> sigma_vec(sigma);
  size_t N = max_size(y, mu, sigma);

  for (size_t n = 0; n < N; n++) {
    const T_partials_return y_dbl = value_of(y_vec[n]);
    const T_partials_return mu_dbl = value_of(mu_vec[n]);
    const T_partials_return sigma_inv_dbl = 1.0 / value_of(sigma_vec[n]);
    const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
    const T_partials_return z = (y_dbl - mu_dbl) * sigma_inv_dbl;

    const T_partials_return Pn = 0.5 - atan(z) / pi();
    ccdf_log += log(Pn);

    const T_partials_return rep_deriv
        = 1.0 / (Pn * pi() * (z * z * sigma_dbl + sigma_dbl));
    if (!is_constant_all<T_y>::value) {
      ops_partials.edge1_.partials_[n] -= rep_deriv;
    }
    if (!is_constant_all<T_loc>::value) {
      ops_partials.edge2_.partials_[n] += rep_deriv;
    }
    if (!is_constant_all<T_scale>::value) {
      ops_partials.edge3_.partials_[n] += rep_deriv * z;
    }
  }
  return ops_partials.build(ccdf_log);
}

}  // namespace math
}  // namespace stan
#endif
