
#include <type_traits>
#include <RcppArmadillo.h>

template<typename T>
void print_type() {
  if constexpr (std::is_name_v<T, arma::mat>) {
    
  }
}


// [[Rcpp::export]]
int test_size(arma::mat m) {
  auto s = size(m);
  // m.n_slices();
  // Rcpp::Rcout << size(m) << std::endl;
  return 0;
}
