
#include <RcppArmadillo.h>

// arma::mat commutation(int m, int n) {
//   arma::uvec onetomtimesn = arma::linspace<arma::uvec>(0, m*n-1, m*n);
//   arma::mat A = arma::reshape(arma::conv_to<arma::vec>::from(onetomtimesn), m, n);
//   // arma::mat A = Avec.reshape(m, n);
//   arma::uvec v = arma::conv_to<arma::uvec>::from(arma::vectorise(A.t()));
//   arma::mat P(m*n, m*n, arma::fill::eye);
//   P = P.submat(v, onetomtimesn);
//   return P;
// }


// void kron1(arma::mat &C) {
//   arma::mat M = arma::kron(arma::vec(100, arma::fill::ones).t(), C);
// }
// 
// void kron2(arma::mat &C) {
//   arma::mat M = arma::kron(arma::vec(100, arma::fill::ones).t(), C);
// }

// [[Rcpp::export]]
void f1(arma::mat &M) {
  arma::vec S = M * arma::vec(M.n_cols, arma::fill::ones);
}

// [[Rcpp::export]]
void f2(arma::mat &M) {
  arma::vec S = arma::sum(M, 1);
}
