#ifndef STAN_PETSC_INTERFACE_HPP
#define STAN_PETSC_INTERFACE_HPP

#define PETSC_CLANGUAGE_CXX 1

#include <petscvec.h>
#include <petscerror.h>

#include <stan/math/rev/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <tuple>

namespace stan {
namespace math {

namespace petsc {

void PetscVecToEigenVector(const Vec& pvec, Eigen::VectorXd& evec)
{
    PetscErrorCode ierr;
    PetscScalar *pdata;
    // Returns a pointer to a contiguous array containing this processor's portion
    // of the vector data. For standard vectors this doesn't use any copies.
    // If the the petsc vector is not in a contiguous array then it will copy
    // it to a contiguous array.
    ierr = VecGetArray(pvec, &pdata);CHKERRXX(ierr);

    // Make the Eigen type a map to the data. Need to be mindful of anything that
    // changes the underlying data location like re-allocations.
    PetscInt size;
    ierr = VecGetSize(pvec, &size);CHKERRXX(ierr);
    evec = Eigen::Map<Eigen::VectorXd>(pdata, size);
    ierr = VecRestoreArray(pvec, &pdata);CHKERRXX(ierr);
}

Vec EigenVectorToPetscVec(const Eigen::Ref<const Eigen::VectorXd>& evec)
{
    PetscErrorCode ierr;
    Vec pvec;
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, evec.size(), evec.data(), &pvec);CHKERRXX(ierr);
    return pvec;
}

template <class ExternalSolver>
class petsc_functor {
    int N_;
    double* x_mem_;  // Holds the input vector
    ExternalSolver solver_;

public:
    petsc_functor() : N_(0), x_mem_(nullptr), solver_(PETSC_COMM_WORLD) {}

    /**
     * Call the PETSc function for the input vector
     *
     * @param x input vector.
     * @return Solution.
     */
    template <std::size_t size>
    Eigen::VectorXd operator()(const std::array<bool, size>& /* needs_adj */,
                                const Eigen::VectorXd& x) {
        // Save the input vector for multiply_adjoint_jacobian
        N_ = x.size();
        x_mem_ = ChainableStack::instance_->memalloc_.alloc_array<double>(N_);
        for (int n = 0; n < N_; ++n) {
            x_mem_[n] = x(n);
        }

        // Convert Eigen input to PETSc Vec
        Vec petsc_x = EigenVectorToPetscVec(x);

        // Initialize PETSc Real to hold the results
        PetscReal petsc_out;

        // petsc_out = forward_function(petsc_x)
        solver_.solve_forward(petsc_x, &petsc_out);

        // Convert PETSc output to Eigen
        Eigen::VectorXd out(1);
        out(0) = petsc_out;
        PetscErrorCode ierr = VecDestroy(&petsc_x);CHKERRXX(ierr);

        return out;
    }

    /**
     * Compute the result of multiply the transpose of the adjoint vector times
     * the Jacobian of the PETSc forward function.
     *
     * @param adj Eigen::VectorXd of adjoints
     * @return Eigen::VectorXd adj*Jacobian
     */
    template <std::size_t size>
    std::tuple<Eigen::VectorXd> multiply_adjoint_jacobian(
        const std::array<bool, size>& /* needs_adj */,
        const Eigen::VectorXd& adj) const {

        // Restore input Eigen Vector
        Eigen::Map<vector_d> x(x_mem_, N_);

        // Convert Eigen input to PETSc Vec
        Vec petsc_grad;
        Vec petsc_x = EigenVectorToPetscVec(x);
        PetscErrorCode ierr;
        ierr = VecDuplicate(petsc_x, &petsc_grad);CHKERRXX(ierr);

        // Calculate petsc_grad = adj * Jacobian(petsc_x)
        solver_.solve_adjoint(petsc_x, petsc_grad, adj(0));
        ierr = VecDestroy(&petsc_x);CHKERRXX(ierr);

        // Convert PETSc Vec to Eigen
        Eigen::VectorXd out(N_);
        PetscVecToEigenVector(petsc_grad, out);
        ierr = VecDestroy(&petsc_grad);CHKERRXX(ierr);

        return std::make_tuple(out);
    }
};
}  // namespace petsc

}  // namespace math
}  // namespace stan

#endif
