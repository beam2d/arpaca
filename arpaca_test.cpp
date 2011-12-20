#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <gtest/gtest.h>
#include "arpaca.hpp"

namespace arpaca {

template<typename T> struct PrecisionThreshold;

template<> struct PrecisionThreshold<float> { static const float value; };
template<> struct PrecisionThreshold<double> { static const double value; };

const float PrecisionThreshold<float>::value = 1e-3f;
const double PrecisionThreshold<double>::value = 1e-10;


std::vector<int> MakeRandomSequence(int n, int k)
{
  std::vector<int> seq;
  seq.reserve(n);
  for (int i = 0; i < n; ++i) seq.push_back(i);
  std::random_shuffle(seq.begin(), seq.end());
  seq.resize(k);
  return seq;
}

double GetRandomValue(double range)
{
  return 2 * range * std::rand() / RAND_MAX - range;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
MakeSymmetrixRandomMatrix(int n, int k, Scalar range)
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X(n, n);
  X.setZero();

  std::vector<int> idx = MakeRandomSequence(n*n, 2*k);
  for (size_t i = 0; i < idx.size(); ++i)
    X(idx[i]/n, idx[i]%n) = GetRandomValue(range);

  return (X + X.transpose()) / 2;
}

template<typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor>
MakeSparse(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X)
{
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> S(X.rows(), X.cols());

  for (int i = 0; i < S.rows(); ++i) {
    S.startVec(i);
    for (int j = 0; j < S.cols(); ++j)
      if (X(i, j) != 0)
        S.insertBack(i, j) = X(i, j);
  }

  S.finalize();
  return S;
}

struct RandomMatrixTestParameter {
  int n;
  int k;
  int num_eigenvalues;
  int num_lanczos_vectors;
};

RandomMatrixTestParameter MakeParameter(int n, int k, int ne, int nlv)
{
  RandomMatrixTestParameter p = { n, k, ne, nlv };
  return p;
}

class RandomMatrixTest
    : public testing::TestWithParam<RandomMatrixTestParameter> {
 protected:
  template<typename Scalar>
  void TestOfScalar()
  {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    RandomMatrixTestParameter p = GetParam();

    Matrix X = MakeSymmetrixRandomMatrix(p.n, p.k, Scalar(10.0));
    SparseMatrix S = MakeSparse(X);

    Eigen::SelfAdjointEigenSolver<Matrix> X_eigen(S);
    SymmetricEigenSolver<Scalar> S_eigen =
        Solve(S, p.num_eigenvalues, ALGEBRAIC_LARGEST, p.num_lanczos_vectors);

    Vector X_eigenvalues = X_eigen.eigenvalues().bottomRows(p.num_eigenvalues);
    Matrix X_eigenvectors =
        X_eigen.eigenvectors().rightCols(p.num_eigenvalues);
    Vector S_eigenvalues = S_eigen.MoveEigenvalues();
    Matrix S_eigenvectors = S_eigen.MoveEigenvectors();

    {
      const Scalar diff_mean =
          (X_eigenvalues - S_eigenvalues).cwiseAbs().mean();
      EXPECT_TRUE(diff_mean < PrecisionThreshold<Scalar>::value)
          << "eigenvalues diff_mean: " << diff_mean;
    }

    {
      Scalar diff_sum = 0.0;
      for (int i = 0; i < p.num_eigenvalues; ++i) {
        // Since each eigenvector given by X and S may have different signs,
        // we have to check either same or opposite.
        const Scalar diff1 =
            (X_eigenvectors.col(i) - S_eigenvectors.col(i)).cwiseAbs().mean();
        const Scalar diff2 =
            (X_eigenvectors.col(i) + S_eigenvectors.col(i)).cwiseAbs().mean();
        diff_sum += std::min(diff1, diff2);
      }
      const Scalar diff_mean = diff_sum / p.num_eigenvalues;

      EXPECT_TRUE(diff_mean < PrecisionThreshold<Scalar>::value)
          << "eigenvectors diff_mean: " << diff_mean;
    }

    std::cout << "info: " << S_eigen.GetInfo() << std::endl;
    std::cout << "# actual iterations: "
              << S_eigen.num_actual_iterations() << std::endl;
    std::cout << "# converged eigenvalues: "
              << S_eigen.num_converged_eigenvalues() << std::endl;
  }
};

TEST_P(RandomMatrixTest, Double)
{
  TestOfScalar<double>();
}

TEST_P(RandomMatrixTest, Float)
{
  TestOfScalar<float>();
}

INSTANTIATE_TEST_CASE_P(
    SizeTest,
    RandomMatrixTest,
    testing::Values(MakeParameter(100, 1000, 10, 50),
                    MakeParameter(1000, 10000, 100, 500)));

}  // namespace arpaca
