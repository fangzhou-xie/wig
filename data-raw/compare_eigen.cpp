
#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd test_eigen(Eigen::MatrixXd &A, Eigen::MatrixXd &B) {
  return A*B;
}
