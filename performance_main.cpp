#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <tr1/cstdint>
#include <tr1/unordered_map>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <pficommon/math/random.h>
#include <pficommon/system/time_util.h>
#include "arpaca.hpp"

struct EqualFirst {
  template<typename Pair>
  bool operator()(const Pair& l, const Pair& r) const
  {
    return l.first == r.first;
  }
};

template<typename RandomNumberGenerator>
Eigen::SparseMatrix<double, Eigen::RowMajor>
MakeSparseSymmetricRandomMatrix(int n, int k, RandomNumberGenerator& rnd)
{
  std::vector<std::pair<std::pair<int, int>, double> > elems;
  elems.reserve(2 * n * k);

  std::vector<int> seq;
  for (int i = 0; i < n; ++i) {
    pfi::math::random::sample_with_replacement(rnd, n, k, seq);
    for (int j = 0; j < k; ++j)
      elems.push_back(
          std::make_pair(std::make_pair(i, seq[j]), rnd.next_gaussian()));
  }

  // make it symmetric
  for (size_t i = 0, orig_size = elems.size(); i < orig_size; ++i) {
    const std::pair<int, int>& pos = elems[i].first;
    elems.push_back(
        std::make_pair(std::make_pair(pos.second, pos.first), elems[i].second));
  }

  std::sort(elems.begin(), elems.end());
  elems.erase(std::unique(elems.begin(), elems.end(), EqualFirst()),
              elems.end());

  Eigen::SparseMatrix<double, Eigen::RowMajor> mat(n, n);

  int cur_row = 0;
  for (size_t i = 0; i < elems.size(); ++i) {
    const int row = elems[i].first.first;
    const int col = elems[i].first.second;

    if (cur_row < row) {
      cur_row = row;
      mat.startVec(row);
    }
    mat.insertBack(row, col) = elems[i].second;
  }

  mat.finalize();
  return mat;
}

arpaca::EigenvalueType
GetEigenvalueType(const std::string& name)
{
  if (name == "LA")
    return arpaca::ALGEBRAIC_LARGEST;
  else if (name == "SA")
    return arpaca::ALGEBRAIC_SMALLEST;
  else if (name == "BE")
    return arpaca::ALGEBRAIC_BOTH_END;
  else if (name == "LM")
    return arpaca::MAGNITUDE_LARGEST;
  else if (name == "SM")
    return arpaca::MAGNITUDE_SMALLEST;
  throw std::invalid_argument("invalid eigenvalue type");
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " <dimension>"
              << " <# of non-zero values in each row>"
              << " <# of eigenvectors>"
              << " <type of eigenvalues>"
              << std::endl;
    std::cerr << "\ttype of eigenvalues: LA SA BE LM SM" << std::endl;
    return 1;
  }

  const int n = std::atoi(argv[1]),
      k = std::atoi(argv[2]),
      r = std::atoi(argv[3]);
  const arpaca::EigenvalueType type = GetEigenvalueType(argv[4]);

  std::cerr << "Making matrix" << std::endl;
  pfi::math::random::mtrand rnd;
  Eigen::SparseMatrix<double, Eigen::RowMajor> X =
      MakeSparseSymmetricRandomMatrix(n, k, rnd);

  std::cerr << "Start performance test" << std::endl;
  pfi::system::time::clock_time begin = pfi::system::time::get_clock_time();

  arpaca::SymmetricEigenSolver<double> solver = arpaca::Solve(X, r, type);

  const double duration = pfi::system::time::get_clock_time() - begin;

  std::cout << "        DIMENSION: " << X.rows() << std::endl;
  std::cout << "         NONZEROS: " << X.nonZeros() << std::endl;
  std::cout << "         DURATION: "
            << duration << " SEC." << std::endl;
  std::cout << "             ITER: "
            << solver.num_actual_iterations() << std::endl;
  std::cout << "CONVERGED EIGVALS: "
            << solver.num_converged_eigenvalues() << std::endl;
  std::cout << "             INFO: " << solver.GetInfo() << std::endl;
}
