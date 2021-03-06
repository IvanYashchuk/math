#ifndef STAN_MATH_PRIM_FUN_CSR_EXTRACT_V_HPP
#define STAN_MATH_PRIM_FUN_CSR_EXTRACT_V_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <vector>
#include <numeric>

namespace stan {
namespace math {

/** \addtogroup csr_format
 *  @{
 */

/**
 * Extract the column indexes for non-zero value from a sparse
 * matrix.
 *
 * @tparam T type of elements in the matrix
 * @param A Sparse matrix.
 * @return Array of column indexes for non-zero entries of A.
 */
template <typename T>
const std::vector<int> csr_extract_v(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& A) {
  std::vector<int> v(A.nonZeros());
  for (int nze = 0; nze < A.nonZeros(); ++nze) {
    v[nze] = *(A.innerIndexPtr() + nze) + stan::error_index::value;
  }
  return v;
}

/**
 * Extract the column indexes for non-zero values from a dense
 * matrix by converting to sparse and calling the sparse matrix
 * extractor.
 *
 * @tparam T type of elements in the matrix
 * @tparam R number of rows, can be Eigen::Dynamic
 * @tparam C number of columns, can be Eigen::Dynamic
 *
 * @param[in] A dense matrix.
 * @return Array of column indexes to non-zero entries of A.
 */
template <typename T, int R, int C>
const std::vector<int> csr_extract_v(const Eigen::Matrix<T, R, C>& A) {
  Eigen::SparseMatrix<T, Eigen::RowMajor> B = A.sparseView();
  std::vector<int> v = csr_extract_v(B);
  return v;
}

/** @} */  // end of csr_format group

}  // namespace math
}  // namespace stan

#endif
