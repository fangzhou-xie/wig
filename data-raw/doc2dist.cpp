
#include <algorithm>
#include <RcppArmadillo.h>

// create the document density vector/matrix in C++ (faster)

// [[Rcpp::export]]
arma::mat doc2dist(std::vector<std::vector<std::string>> docs, std::vector<std::string> dict) {
  arma::mat docmat(dict.size(),docs.size(),arma::fill::zeros);
  
  for (size_t j = 0; j < docs.size(); ++j) {
    // auto doc = docs[j];
    for (std::string &tk : docs[j]) {
      auto it = std::find(dict.begin(), dict.end(), tk);
      // if the token is OOV, then cast it into "</s>"
      it = (it != dict.end()) ? it : dict.end() - 1; 
      const int i = it - dict.begin();
      docmat(i,j) += 1;
    }
    // scale the matrix so that each col sum to 1
    docmat.col(j) /= accu(docmat.col(j));
  }
  return docmat;
}
