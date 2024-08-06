
#include <RcppArmadillo.h>

// // [[Rcpp::export]]
// arma::mat dim3(arma::mat a1, arma::mat a2, arma::mat b) {
//   arma::cube a(a1.n_rows, a1.n_cols, 2);
//   
//   a.slice(0) = a1;
//   a.slice(1) = a2;
//   
//   Rcpp::Rcout << a << std::endl;
//   // Rcpp::Rcout << a.each_slice() * b << std::endl;
//   // Rcpp::Rcout << sum(a.each_slice() * b, 2) << std::endl;
//   arma::cube aa = a.each_slice( [](arma::mat& x) {x.t();});
//   Rcpp::Rcout << a.each_slice( [](arma::mat& x) {x.t();}) << std::endl;
//   
//   return b;
// }

// [[Rcpp::export]]
arma::cube dim3(arma::mat A, arma::cube B) {
  arma::cube C = A * B.each_slice();
  
  Rcpp::Rcout << C << std::endl;
  C.each_slice([](arma::mat& X) {X = X.t();});
  Rcpp::Rcout << C << std::endl;
  return C;
}
