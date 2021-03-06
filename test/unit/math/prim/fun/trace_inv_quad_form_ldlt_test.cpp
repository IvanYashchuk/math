#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, trace_inv_quad_form_ldlt) {
  stan::math::matrix_d A(4, 4), B(4, 2);
  stan::math::LDLT_factor<double, -1, -1> ldlt_A;
  A << 9.0, 3.0, 3.0, 3.0, 3.0, 10.0, 2.0, 2.0, 3.0, 2.0, 7.0, 1.0, 3.0, 2.0,
      1.0, 112.0;
  B << 100, 10, 0, 1, -3, -3, 5, 2;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  EXPECT_FLOAT_EQ(1439.1061766207, trace_inv_quad_form_ldlt(ldlt_A, B));
  EXPECT_FLOAT_EQ((B.transpose() * A.inverse() * B).trace(),
                  trace_inv_quad_form_ldlt(ldlt_A, B));
}

TEST(MathMatrixPrimMat, trace_inv_quad_form_ldlt_0x0) {
  stan::math::LDLT_factor<double, -1, -1> ldlt_A;
  stan::math::matrix_d B(0, 0), C(0, 2), D(2, 0);

  EXPECT_FLOAT_EQ(0, trace_inv_quad_form_ldlt(ldlt_A, B));
  EXPECT_FLOAT_EQ(0, trace_inv_quad_form_ldlt(ldlt_A, C));

  EXPECT_THROW(trace_inv_quad_form_ldlt(ldlt_A, D), std::invalid_argument);
}
