
// debiased sinkhorn barycenter in log domain (numeric stablization)
// TODO: 
// [ ]: first implement DSB

// [[Rcpp::plugins(cpp17)]]

#include <RcppArmadillo.h>

// #include "utils.h" // softmin_col, softmin_row
// [x] checked
// double softmin(arma::vec z, double eps) {
//   return z.min() - eps * log(accu(exp(-(z-z.min())/eps)));
// }

arma::mat softmin_col(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  // compute S first: S = C - f * 1^T - 1 * g^T
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::mat S_col(C.n_cols, f.n_cols, arma::fill::zeros);
  arma::vec S_j(C.n_rows, arma::fill::zeros);
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t j = 0; j < C.n_cols; ++j) {
      S_j = S.col(j);
      S_col(j,k) = S_j.min() - eps * log(accu(exp(-(S_j-S_j.min())/eps)));
      // S_col(j,k) = softmin(S.col(j), eps);
    }
  }
  return S_col;
} 

arma::mat softmin_row(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S_row(C.n_rows, f.n_cols, arma::fill::zeros);
  arma::vec S_i(C.n_rows, arma::fill::zeros);
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t i = 0; i < C.n_rows; ++i) {
      S_i = S.row(i).t();
      S_row(i,k) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
      // S_row(i,k) = softmin(S.row(i).t(), eps);
    } 
  }
  return S_row;
} 

arma::vec softmin_row(arma::mat &C, arma::vec log_d, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S = C - ones_N * log_d.t();
  arma::vec r_row(C.n_rows, arma::fill::zeros);
  arma::vec S_i(C.n_rows, arma::fill::zeros);
  for (size_t i = 0; i < C.n_rows; ++i) {
    S_i = S.row(i).t();
    r_row(i) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
    // r_row(i) = softmin(S.row(i).t(), eps);
  }
  return r_row;
}

// [[Rcpp::export]]
Rcpp::List dsink_log(arma::mat &A, arma::mat &C, arma::vec &w, const double reg,
               const int maxIter = 10000, const double zeroTol = 1e-6) {
  arma::mat F(arma::size(A), arma::fill::zeros);
  arma::mat G(arma::size(A), arma::fill::zeros);
  
  // log vars in matrix form: for broadcasting purpose
  arma::mat log_A = log(A);
  
  arma::vec log_b(A.n_rows, arma::fill::value(log(1/(double)A.n_rows)));
  arma::vec log_d(A.n_rows, arma::fill::zeros);
  arma::vec log_b_prev;
  
  // init some temp vars
  arma::mat S(arma::size(A), arma::fill::zeros);
  arma::vec r(A.n_rows, arma::fill::zeros);
  
  // init iter and err
  int iter = 0; 
  double err = 1000.;
  
  while ((iter < maxIter) & (err > zeroTol)) {
    log_b_prev = log_b;
    
    // update F
    // Rcpp::Rcout << "F: " << std::endl;
    // Rcpp::Rcout << F << std::endl;
    // Rcpp::Rcout << softmin_row(C, F, G, reg) << std::endl;
    F += softmin_row(C, F, G, reg) + reg*log_A;
    S = softmin_col(C, F, G, reg);
    
    // Rcpp::Rcout << "f:" << std::endl << F << std::endl;
    
    // update b (log_b)
    log_b = log_d - (S + G)*w / reg;
    // log_b = - (S + G)*w / reg;
    // Rcpp::Rcout << "log_b:" << std::endl << log_b << std::endl;
    
    // update G
    // Rcpp::Rcout << "G: " << std::endl;
    // Rcpp::Rcout << G << std::endl;
    G += S.each_col() + reg*log_b;
    // Rcpp::Rcout <<  S.each_col() + reg*log_b << std::endl;
    // Rcpp::Rcout << "g:" << std::endl << G << std::endl;
    
    // update d
    r = softmin_row(C, reg*log_d, reg);
    log_d = (log_d + log_b + r/reg)/2;
    // Rcpp::Rcout << "log_d:" << std::endl << log_d << std::endl;
    
    if (iter % 30 == 0) {
      Rcpp::Rcout << "A:" << std::endl << A << exp(-softmin_row(C,F,G,reg)/reg);
    }
    
    // terminal condition
    ++iter;
    // err = norm(exp(log_b) - exp(log_b_prev));
    err = norm(A - exp(-softmin_row(C,F,G,reg)/reg));
  }
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("A") = exp(-softmin_row(C,F,G,reg)/reg),
    Rcpp::Named("b") = exp(log_b),
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}

// [[Rcpp::export]]
Rcpp::List dsink(arma::mat &A, arma::mat &C, arma::vec &w, const double reg,
          const int maxIter = 1000, const double zeroTol = 1e-6) {
  arma::mat u(arma::size(A), arma::fill::ones);
  arma::mat v(arma::size(A), arma::fill::ones);
  arma::mat K = exp(-C/reg);

  arma::vec b;
  arma::vec d(A.n_rows, arma::fill::ones);

  // init iter and err
  int iter = 0;
  double err = 1000.;

  while ((iter < maxIter) & (err > zeroTol)) {
    arma::mat Kv = K*v;
    u = A / Kv;

    arma::mat KTu = K.t() * u;
    arma::mat KTu_lbd = pow(KTu.each_row(), w.t());
    arma::vec Pi = prod(KTu_lbd, 1);
    b = d % Pi;

    v = b / KTu.each_col();
    d = sqrt((d % b) / (K*d));

    ++iter;
  }
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("A") = u % (K*v),
    Rcpp::Named("b") = v % (K.t()*u),
    Rcpp::Named("iter") = iter,
    Rcpp::Named("err") = err
  );
  return res;
}



// // [[Rcpp::export]]
// Rcpp::List dsink_log(arma::mat &A, arma::mat &C, arma::vec &w, double &reg,
//                const int maxIter=1000, const double zeroTol=1e-6) {
//   // init: f, g: N*K, same size as A
//   // f, g: zeros <==> u, v: ones
//   arma::mat f(arma::size(A), arma::fill::zeros);
//   arma::mat g(arma::size(A), arma::fill::zeros);
//   
//   arma::mat f_prev = f;
//   arma::mat g_prev = g;
//   arma::cube S(C.n_rows, C.n_cols, A.n_cols, arma::fill::zeros);
//   arma::mat log_A = log(A);
//   
//   // other log domain vars
//   arma::vec log_b(A.n_rows, arma::fill::value(1/A.n_rows));
//   arma::vec log_d(A.n_rows, arma::fill::zeros);
//   
//   // transport map P
//   arma::cube P(C.n_rows, C.n_cols, A.n_cols, arma::fill::zeros);
//   
//   // init iter and err
//   int iter = 0;
//   double err = 1000.;
//   
//   while ((iter < maxIter) & (err > zeroTol)) {
//     f_prev = f;
//     g_prev = g;
//     
//     // update f (equi to update u)
//     S = update_S(C, f, g, reg);
//     f = softmin_row(S, reg) + f + reg*log_A;
//     
//     // Rcpp::Rcout << f << std::endl;
//     // Rcpp::Rcout << exp(-softmin_row(S, reg)/reg) << std::endl;
//     
//     // update b (estimated output)
//     S = update_S(C, f, g, reg);
//     arma::mat S_b = softmin_col(S, reg);
//     log_b = log_d - (S_b * w) / reg;
//     
//     Rcpp::Rcout << exp(log_b) << std::endl;
//     // Rcpp::Rcout << S_b << std::endl;
//     
//     // update g (equi to update v)
//     S = update_S(C, f, g, reg);
//     arma::mat g1 = softmin_col(S, reg) + g;
//     g = g1.each_col() + reg*log_b;
//     
//     // update d (aux var)
//     arma::vec r = update_r(C, log_d, reg);
//     log_d = log_d + (S_b * w) / (2*reg);
//     
//     // update err (conv criterion)
//     err = norm(f - f_prev) + norm(g - g_prev);
//     // update counter
//     iter += 1;
//     
//     Rcpp::Rcout << "iter: " << iter << ", err: " << err << std::endl;
//   }
//   S = update_S(C, f, g, reg);
//   arma::mat Ahat = exp(-softmin_row(S, reg)/reg);
//   
//   Rcpp::List res = Rcpp::List::create(
//     Rcpp::Named("P") = P,
//     Rcpp::Named("iter") = iter,
//     Rcpp::Named("err") = err,
//     Rcpp::Named("A") = Ahat
//   );
//   return res;
// }


// arma::cube update_S(arma::mat &C, arma::mat &f, arma::mat &g, double &eps) {
//   // C: N*N
//   // f, g: N*K
//   arma::vec ones_N = arma::ones(C.n_rows);
//   arma::cube S(C.n_rows, C.n_cols, f.n_cols, arma::fill::zeros);
//   for (size_t k = 0; k < f.n_cols; ++k) {
//     S.slice(k) = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
//   }
//   return S;
// }
// 
// arma::mat softmin_col(arma::cube &S, double &eps) {
//   arma::mat S_softmax_col(S.n_cols, S.n_slices, arma::fill::zeros);
//   for (size_t k = 0; k < S.n_slices; ++k) {
//     for (size_t j = 0; j < S.n_cols; ++j) {
//       S_softmax_col(j,k) = softmin(S.slice(k).col(j), eps);
//     }
//   }
//   return S_softmax_col;
// }
// 
// arma::mat softmin_row(arma::cube &S, double &eps) {
//   arma::mat S_softmax_row(S.n_rows, S.n_slices, arma::fill::zeros);
//   // Rcpp::Rcout << S_softmax_row << std::endl;
//   for (size_t k = 0; k < S.n_slices; ++k) {
//     for (size_t i = 0; i < S.n_rows; ++i) {
//       arma::vec S_ik = arma::conv_to<arma::colvec>::from(S.slice(k).row(i));
//       S_softmax_row(i,k) = softmin(S_ik, eps);
//     }
//   }
//   return S_softmax_row;
// }
// 
// arma::vec softmin_row(arma::mat &S, double &eps) {
//   arma::vec S_row(S.n_rows, arma::fill::zeros);
//   for (size_t i = 0; i < S.n_rows; ++i) {
//     arma::vec S_i = arma::conv_to<arma::colvec>::from(S.row(i));
//     S_row.row(i) = softmin(S_i, eps);
//   }
//   return S_row;
// }
// 
// arma::vec update_r(arma::mat &C, arma::vec &log_d, double &eps) {
//   arma::vec ones_N = arma::ones(C.n_rows);
//   arma::mat S_r = C - ones_N * log_d.t();
//   return softmin_row(S_r, eps);
// }
// arma::vec softmin_row(arma::mat &C, arma::vec &log_d, const double eps) {
//   arma::vec ones_N = arma::ones(C.n_cols);
//   arma::vec r_row(C.n_rows, arma::fill::zeros);
// 
//   arma::mat S = C - ones_N * log_d.t();
//   for (size_t i = 0; i < C.n_rows; ++i) {
//     r_row(i) = softmin(S.row(i).t(), eps);
//   }
//   return r_row;
// }

// arma::mat softmin_row(arma::mat &C, arma::mat &log_D, const double eps) {
//   arma::vec log_d = log_D.col(0);
//   arma::vec ones_N = arma::ones(C.n_cols);
//   arma::mat S = C - ones_N * log_d.t();
//   arma::vec r_row(C.n_rows, arma::fill::zeros);
//   // Rcpp::Rcout << "S:" << std::endl << S << std::endl;
//   for (size_t i = 0; i < C.n_rows; ++i) {
//     r_row(i) = softmin(S.row(i).t(), eps);
//   }
//   // arma::mat R(arma::size(log_D), arma::fill::zeros);
//   // R.each_col() = r_row;
//   // return R;
//   return r_row;
// }
// arma::mat softmin_row(arma::mat &C, arma::mat &log_D, const double eps) {
//   arma::vec log_d = log_D.col(0);
//   arma::vec ones_N = arma::ones(C.n_cols);
//   arma::mat S = C - ones_N * log_d.t();
//   arma::vec r_row(C.n_rows, arma::fill::zeros);
//   for (size_t i = 0; i < C.n_rows; ++i) {
//     r_row(i) = softmin(S.row(i).t(), eps);
//   }
//   arma::mat R(arma::size(log_D), arma::fill::zeros);
//   R.each_col() = r_row;
//   return R;
// }
