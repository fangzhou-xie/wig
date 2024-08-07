
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat test_arma(arma::mat &A, arma::mat &B) {
  return A*B;
}
