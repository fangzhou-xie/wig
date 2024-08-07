
// utils
// row-wise euclidean distance and doc dist conversion

#include <algorithm>
#include <RcppArmadillo.h>

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export("euclidean_cpp")]]
arma::mat euclidean(const arma::mat &a) {
  // for each two rows, calculate the euclidean distance
  arma::mat euc(a.n_rows, a.n_rows, arma::fill::zeros);
  for (size_t i = 0; i < a.n_rows; ++i) {
    for (size_t j = 0; j < a.n_rows; ++j) {
      if (i < j) {
        // Rcpp::Rcout << a.row(i)
        double c = accu(pow(a.row(i) - a.row(j), 2));
        // Rcpp::Rcout << c << std::endl;
        euc(i,j) = c;
        euc(j,i) = c;
      }
    }
  }
  // check if the matrix is symmetric
  if (!euc.is_symmetric()) {
    Rcpp::stop("euc matrix is not symmetric!");
  }
  return euc;
}

// [[Rcpp::export("doc2dist_cpp")]]
arma::mat doc2dist(std::vector<std::vector<std::string>> docs, std::vector<std::string> dict) {
  arma::mat docmat(dict.size(),docs.size(),arma::fill::zeros);
  
  for (size_t j = 0; j < docs.size(); ++j) {
    // auto doc = docs[j];
    for (std::string &tk : docs[j]) {
      auto it = std::find(dict.begin(), dict.end(), tk);
      it = (it != dict.end()) ? it : dict.end();
      const int i = it - dict.begin();
      docmat(i,j) += 1;
    }
    // scale the matrix so that each col sum to 1
    docmat.col(j) /= accu(docmat.col(j));
  }
  return docmat;
}

// [x] checked
double softmin(arma::vec z, const double eps) {
  // double zmin = z.min();
  return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
}

arma::cube update_S(arma::mat &C, arma::mat &f, arma::mat &g, double &eps) {
  // C: N*N
  // f, g: N*K
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::cube S(C.n_rows, C.n_cols, f.n_cols, arma::fill::zeros);
  for (size_t k = 0; k < f.n_cols; ++k) {
    S.slice(k) = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
  }
  return S;
}

arma::mat softmin_col(arma::cube &S, const double &eps) {
  arma::mat S_softmax_col(S.n_cols, S.n_slices, arma::fill::zeros);
  for (size_t k = 0; k < S.n_slices; ++k) {
    for (size_t j = 0; j < S.n_cols; ++j) {
      S_softmax_col(j,k) = softmin(S.slice(k).col(j), eps);
    }
  }
  return S_softmax_col;
}

arma::mat softmin_row(arma::cube &S, const double &eps) {
  arma::mat S_softmax_row(S.n_rows, S.n_slices, arma::fill::zeros);
  // Rcpp::Rcout << S_softmax_row << std::endl;
  for (size_t k = 0; k < S.n_slices; ++k) {
    for (size_t i = 0; i < S.n_rows; ++i) {
      arma::vec S_ik = arma::conv_to<arma::colvec>::from(S.slice(k).row(i));
      S_softmax_row(i,k) = softmin(S_ik, eps);
    }
  }
  return S_softmax_row;
}

arma::vec softmin_row(arma::mat &S, const double &eps) {
  arma::vec S_row(S.n_rows, arma::fill::zeros);
  for (size_t i = 0; i < S.n_rows; ++i) {
    // arma::vec S_i = arma::conv_to<arma::colvec>::from()
    S_row.row(i) = softmin(S.row(i), eps);
  }
  return S_row;
}

arma::vec update_r(arma::mat &C, arma::vec &log_d, const double &eps) {
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::mat S_r = C - ones_N * log_d.t();
  return softmin_row(S_r, eps);
}

// // [[Rcpp::export()]]
// arma::mat doc2dist_cpp(Rcpp::List docs, std::vector<std::string> dict) {
//   arma::mat docdist()
// }

// arma::vec testslice(arma::cube &m) {
//   return m.tube(arma::span(),arma::span());
// }
