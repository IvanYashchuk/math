#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <petsc.h>

TEST(MathStanPetscInterface, mpi_conversions) {
    // initialize random Eigen array
    Eigen::VectorXd eigen_vec1 = Eigen::VectorXd::Random(5);
    Eigen::VectorXd eigen_vec2 = Eigen::VectorXd::Random(6);

    // now transform to petsc
    using stan::math::petsc::EigenVectorToPetscVecMPI;
    Vec petsc_vec1 = EigenVectorToPetscVecMPI(eigen_vec1);
    Vec petsc_vec2 = EigenVectorToPetscVecMPI(eigen_vec2);

    // now transform back to Eigen
    Eigen::VectorXd new_eigen_vec1;
    using stan::math::petsc::PetscVecToEigenVectorMPI;
    PetscVecToEigenVectorMPI(petsc_vec1, new_eigen_vec1);
    ASSERT_TRUE( eigen_vec1.isApprox(new_eigen_vec1) );

    Eigen::VectorXd new_eigen_vec2;
    PetscVecToEigenVectorMPI(petsc_vec2, new_eigen_vec2);
    ASSERT_TRUE( eigen_vec2.isApprox(new_eigen_vec2) );

    // Try to modify vec2, only local portion of Eigen array changes
    PetscRandom rctx;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    VecSetRandom(petsc_vec2,rctx);
    PetscRandomDestroy(&rctx);

    Eigen::VectorXd new_eigen_vec3;
    PetscVecToEigenVectorMPI(petsc_vec2, new_eigen_vec3);
    PetscInt size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size > 1) {
        ASSERT_FALSE( eigen_vec2.isApprox(new_eigen_vec3) );
    }
    else {
        // when run with one process, the local portion is the whole array
        ASSERT_TRUE( eigen_vec2.isApprox(new_eigen_vec3) );
    }

    VecDestroy(&petsc_vec1);
    VecDestroy(&petsc_vec2);
}
