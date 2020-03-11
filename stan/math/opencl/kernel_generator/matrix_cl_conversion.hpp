#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_MATRIX_CL_CONVERSION
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_MATRIX_CL_CONVERSION
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/multi_result_kernel.hpp>
#include <stan/math/opencl/matrix_cl.hpp>

namespace stan{
namespace math{

template<typename T>
template<typename Expr, require_all_valid_expressions_and_none_scalar_t<Expr>*>
matrix_cl<T, require_arithmetic_t<T>>::matrix_cl(const Expr& expresion):
  rows_(0), cols_(0) {
  results(*this) = expressions(expresion);
}

template<typename T>
template<typename Expr, require_all_valid_expressions_and_none_scalar_t<Expr>*>
matrix_cl<T>& matrix_cl<T, require_arithmetic_t<T>>::operator=(const Expr& expresion){
  results(*this) = expressions(expresion);
  return *this;
}

}
}

#endif
#endif
