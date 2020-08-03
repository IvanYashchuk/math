#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <petsc.h>

TEST(MathStanPetscInterface, eigen_to_petsc) {
  // initialize random Eigen array
  Eigen::VectorXd eigen_vec1 = Eigen::VectorXd::Random(5);
  Eigen::VectorXd eigen_vec2 = Eigen::VectorXd::Random(6);

  // now transform to petsc
  using stan::math::petsc::EigenVectorToPetscVec;
  Vec petsc_vec1 = EigenVectorToPetscVec(eigen_vec1);
  Vec petsc_vec2 = EigenVectorToPetscVec(eigen_vec2);

  // Try to modify vec2, eigen array should change as well
  PetscRandom rctx;
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  VecSetRandom(petsc_vec2, rctx);
  PetscRandomDestroy(&rctx);

  // check eigen_vec1 and petsc_vec1 have same values
  PetscScalar* petsc_vec_array;
  VecGetArray(petsc_vec1, &petsc_vec_array);
  Eigen::Map<Eigen::VectorXd> eigen_petsc_vec1(petsc_vec_array, eigen_vec1.size());
  ASSERT_TRUE( eigen_vec1.isApprox(eigen_petsc_vec1) );
  VecRestoreArray(petsc_vec1, &petsc_vec_array);

  // check eigen_vec2 and petsc_vec2 have same values
  VecGetArray(petsc_vec2, &petsc_vec_array);
  Eigen::Map<Eigen::VectorXd> eigen_petsc_vec2(petsc_vec_array, eigen_vec2.size());
  ASSERT_TRUE( eigen_vec2.isApprox(eigen_petsc_vec2) );
  VecRestoreArray(petsc_vec2, &petsc_vec_array);

  VecDestroy(&petsc_vec1);
  VecDestroy(&petsc_vec2);
}

TEST(MathStanPetscInterface, petsc_to_eigen) {
  // initialize random PETSc Vec
  Vec petsc_vec;
  VecCreateSeq(PETSC_COMM_SELF, 5, &petsc_vec);
  PetscRandom rctx;
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  VecSetRandom(petsc_vec, rctx);

  // now transform to eigen
  Eigen::VectorXd eigen_vec;
  using stan::math::petsc::PetscVecToEigenVector;
  PetscVecToEigenVector(petsc_vec, eigen_vec);

  // check eigen_vec and petsc_vec have same values
  PetscScalar* petsc_vec_array;
  VecGetArray(petsc_vec, &petsc_vec_array);
  Eigen::Map<Eigen::VectorXd> eigen_petsc_vec(petsc_vec_array, eigen_vec.size());
  ASSERT_TRUE( eigen_vec.isApprox(eigen_petsc_vec) );
  VecRestoreArray(petsc_vec, &petsc_vec_array);

  // Try to modify petsc_vec, eigen array shouldn't change
  VecSetRandom(petsc_vec, rctx);

  // check eigen_vec and petsc_vec have different values
  VecGetArray(petsc_vec, &petsc_vec_array);
  Eigen::Map<Eigen::VectorXd> eigen_petsc_vec2(petsc_vec_array, eigen_vec.size());
  ASSERT_FALSE( eigen_vec.isApprox(eigen_petsc_vec2) );
  VecRestoreArray(petsc_vec, &petsc_vec_array);

  VecDestroy(&petsc_vec);
  PetscRandomDestroy(&rctx);
}
