
// test autodiff for jacobian calculation

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends("RcppEigen")]]

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
// #define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// autodiff can only be used in vector functions

// #include <Rcpp.h>
#include <RcppEigen.h>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// // The vector function with parameters for which the Jacobian is needed
// VectorXreal f(const VectorXreal& x, const VectorXreal& p, const real& q)
// {
//   return x * p.sum() * exp(q);
// }

VectorXreal f(const VectorXreal& A, const VectorXreal& B) {
  return A.array()*B.array();
  // return A * B.transpose();
}


// [[Rcpp::export]]
int test_autodiff(Eigen::Map<Eigen::VectorXd> A, Eigen::Map<Eigen::VectorXd> B) {
  using Eigen::MatrixXd;
  
  VectorXreal Ar = A.cast<real>();
  VectorXreal Br = B.cast<real>();
  
  // Rcpp::Rcout << Ar << std::endl;
  // Rcpp::Rcout << Br << std::endl;
  Rcpp::Rcout << A * B << std::endl;
  Rcpp::Rcout << Ar * Br << std::endl;
  
  
  VectorXreal F;
  
  MatrixXd J = jacobian(f, wrt(Ar), at(Ar, Br), F);
  Rcpp::Rcout << J << std::endl;
  Rcpp::Rcout << F << std::endl;
  
  // Rcpp::Rcout << f(Ar,Br) << std::endl;
  
  return 0;
}


