Arpaca - ARPACk Adaptor for real symmetric eigenproblem in C++ with Eigen
=========================================================================

Arpaca is a thin wrapper of ARnoldi PACKage (ARPACK) in C++ using Eigen.

Requirement
-----------

Arpaca uses [ARPACK](http://www.caam.rice.edu/software/ARPACK/ "arpack").
You can install ARPACK in many package systems.
If you want to install it from source code,
gfortran, BLAS (ATLAS is better as its implementation) and LAPACK are
required.

arpaca_performance_test also depends on
[pficommon](http://github.com/pfi/pficommon "pficommon").

Installation
------------

Just copy arpaca.hpp to any path in include paths list.

Typical Usage
-------------

You can use arpaca by including arpaca.hpp and linking to the dependent libraries
listed above.

In order to compute the top ten eigenvalues and corresponding eigenvectors of
large sparse symmetric matrix A, write as follows:

```c++
Eigen::SparseMatrix<double, Eigen::RowMajor> A;
//...

const int num_eigenvalues = 10;
const arpaca::EigenvalueType type = arpaca::ALGEBRAIC_LARGEST;

arpaca::SymmetricEigenSolver<double> solver =
    arpaca::Solve(A, num_eigenvalues, type);

const Eigen::MatrixXd& eigenvectors = solver.eigenvectors();
const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
```

__EigenvalueType__ indicates which side of eigenvalues to compute.
You can compute large or small side of eigenvalues in the sense of signed or
absolute value.

Thanks to the flexibility of ARPACK, you can use arbitrary formulation of
operator A * x, where x is a real vector.

```c++
template<typename MatrixA, typename MatrixB>
class TimesAB {
 public:
  explicit TimesAB(MatrixA& A, typename MatrixB)
      : A_(A),
        B_(B)
  {}

  template<typename X, typename Y>
  void operator(X x, Y y) const
  {
    y = A_ * (B_ * x);
  }

 private:
  MatrixA& A_;
  MatrixB& B_;
};

template<typename MatrixA, typename MatrixB>
TimesAB<MatrixA, MatrixB> MakeTimesAB(MatrixA& A, MatrixB& B) {
  return TimesAB<MatrixA, MatrixB>(A, B);
}

Eigen::SparseMatrix<double, Eigen::RowMajor> A;
Eigen::SparseMatrix<double, Eigen::ColMajor> B;
//...

// Solve eigenproblem of AB'
arpaca::SymmetricEigenSolver<double> solver;
solver.Solve(A.rows(), 10, MakeTimesAB(A, B.transpose()));
```

License
--------

Arpaca is distributed under MIT License, which is available in LICENSE file.


Enjoy!

Copyright (c) 2011 Seiya Tokui <beam.web@gmail.com>. All Rights Reserved.
