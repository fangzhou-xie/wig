
// some helper functions
// 1. Euclidean distance for the word embedding matrix
// 2. standardize to 100 mean and 1 sd

#include <RcppArmadillo.h>

arma::mat init(int nrow, int ncol) {
  Rcpp::Function f("rnorm");
  auto v = f(nrow*ncol, Rcpp::Named("mean") = 0, Rcpp::Named("sd") = 1);
  std::vector<double> v_vec = Rcpp::as<std::vector<double>>(v);
  arma::mat m(v_vec);
  m.reshape(nrow, ncol);
  return m;
}


// #include <Rcpp.h>
// #include <RcppEigen.h>
// 
// 
// // draw random samples from standard normal distribution
// std::vector<double> rnorm(int n, double m = 0, double sd = 1) {
//   // probably not the fastest way to sample from normal distribution
//   // but can by controlled by `set.seed()` from R side
//   Rcpp::Function f("rnorm");
//   auto v = f(n, Rcpp::Named("mean") = m, Rcpp::Named("sd") = sd);
//   std::vector<double> v_vec = Rcpp::as<std::vector<double>>(v);
//   return v_vec;
// }
// 
// // initialze Eigen matrix from random samples of normal distribution
// Eigen::MatrixXd init(int nrow, int ncol) {
//   std::vector<double> rn_vec = rnorm(nrow * ncol);
//   Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(rn_vec.data(), nrow, ncol);
//   return m;
// }
// 
// // initialze Eigen vector from random samples of normal distribution
// Eigen::VectorXd init(int nrow) {
//   std::vector<double> rn_vec = rnorm(nrow);
//   Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(rn_vec.data(), nrow);
//   return v;
// }
// 
// // compute column-wise softmax of matrix/vector
// void softmax(Eigen::MatrixXd &a) {
//   a = a.array().exp();
//   for (int i = 0; i < a.cols(); ++i) {
//     a.col(i) = a.col(i) / a.col(i).array().sum();
//   }
// }
// 
// void softmax(Eigen::VectorXd &a) {
//   a = a.array().exp();
//   a = a / a.array().sum();
// }

// calculate Euclidean distance of two matrices by columns
// [[Rcpp::export("euclidean_cpp")]]
Eigen::MatrixXd euclidean(const Eigen::Map<Eigen::MatrixXd> &a) {
  Eigen::MatrixXd euc(a.rows(), a.rows());
  // for each two rows, calculate euclidean distance
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.rows(); ++j) {
      if (i < j) {
        double c = (a.row(i).array() - a.row(j).array()).square().sum();
        euc(i, j) = std::sqrt(c);
        euc(j, i) = std::sqrt(c);
      } else if (i == j) {
        euc(i, j ) = 0.;
      }
    }
  }
  return euc;
}


// standardize to mean 100 and sd 1
// [[Rcpp::export("standardize_cpp")]]
Eigen::VectorXd standardize(const Eigen::Map<Eigen::VectorXd> &v) {
  double m = v.array().mean();
  double s = std::sqrt((v.array() - m).pow(2).sum() / (v.size() - 1));
  return (v.array() - m)/s + 100;
}


// // [[Rcpp::export("docdict2dist_cpp")]]
// Eigen::MatrixXd docdict2dist(
//   const std::vector<std::string> &dict,
//   const Rcpp::List &docs_token
// ) {
//   // for each doc, create a vector (column) as the empirical distribution
//   // of the tokens
//   Eigen::MatrixXd docsmat(dict.size(), docs_token.size());
//   docsmat.setZero();
//   
//   for (int j = 0; j < docs_token.size(); ++j) {
//     std::vector<std::string> d = docs_token[j];
//     for (auto &token : d) {
//       auto d_itr = std::find(dict.begin(), dict.end(), d);
//       auto d_id = std::distance(dict.begin(), d_itr);
//       docsmat(d_id, j) += 1;
//     }
//     docsmat.col(j) = docsmat.col(j) / docsmat.col(j).array().sum();
//   }
//   return docsmat;
// }

// // [[Rcpp::export()]]
// Eigen::MatrixXd flip_sign_sklearn(const Eigen::Map<Eigen::VectorXd> &M) {
//   // for each column, take the maximum value and see if it is positive
//   Eigen::MatrixXd M2 = M;
//   for (int i = 0; i < M.cols(); ++i) {
//     double cmax = M.col(i).array().abs().maxCoeff();
//     if (cmax < 0) {
//       M2.col(i) = -M2.col(i);
//     }
//   }
//   return M2;
// }

// // [[Rcpp::export()]]
// Eigen::MatrixXd flip_sign_auto(
//   const Eigen::Map<Eigen::VectorXd> &X,
//   const Eigen::Map<Eigen::VectorXd> &U,
//   const Eigen::Map<Eigen::VectorXd> &S,
//   const Eigen::Map<Eigen::VectorXd> &V
// ) {
//   // Reference:
//   // Bro, R., Acar, E., & Kolda, T. G. (2008).
//   // Resolving the sign ambiguity in the singular value decomposition.
//   // Journal of Chemometrics, 22(2), 135–140. https://doi.org/10.1002/cem.1122
//   
//   Eigen::MatrixXd Y = X - U * S * V.transpose();
//   Eigen::VectorXd s_left(S.cols());
//   Eigen::VectorXd s_right(S.cols());
//   
//   for (int k = 0; k < S.cols(); ++k) {
//     double sk_left = 0.;
//     double sk_right = 0.;
//     
//     for (int j = 0; j < Y.cols(); ++j) {
//       auto uTy = (U.col(k).array() * Y.col(j).array()).sum();
//       int sign_left = uTy > 0 ? 1 : -1;
//       sk_left += sign_left * std::pow(uTy, 2);
//     }
//     
//     for (int i = 0; i < Y.rows(); ++i) {
//       auto vTy = (V.row(i).array() * Y.row(i).array()).sum();
//       int sign_right = vTy > 0 ? 1 : -1;
//       sk_right += sign_right * std::pow(vTy, 2);
//     }
//     
//     if ()
//   }
//   
// }


// // QR decomposition: get matrix Q 
// Eigen::MatrixXd qrq(const Eigen::MatrixXd &M) {
//   // QR decomposition and get Q
//   // Modified Gram-Schmidt
//   // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability
//   
//   int dimcol = M.cols() <= M.rows() ? M.cols() : M.rows();
//   
//   Eigen::MatrixXd Q(M.rows(), dimcol);
//   Eigen::VectorXd u_k(M.rows());
//   for (int i = 0; i < Q.cols(); ++i) {
//     if (i == 0) {
//       Q.col(i) = M.col(i);
//     } else {
//       u_k = M.col(i);
//       for (int j = 0; j < i; ++j) {
//         u_k = u_k - 
//           (u_k.array() * Q.col(j).array()).sum()/(Q.col(j).array().pow(2).sum()) * Q.col(j);
//       }
//       Q.col(i) = u_k;
//     }
//   }
//   
//   // normalize columns
//   for (int i = 0; i < Q.cols(); ++i) {
//     Q.col(i).normalize();
//   }
//   
//   // get matrix R (upper triangular)
//   // Eigen::MatrixXd R(Q.cols(), M.cols());
//   // R.setZero();
//   // for (int i = 0; i < R.rows(); ++i) {
//   //   for (int j = 0; j < R.cols(); ++j) {
//   //     if (j >= i) {
//   //       R(i, j) = (Q.col(i).array() * M.col(j).array()).sum();
//   //     }
//   //   }
//   // }
//   
//   return Q;
// }
// 
// // customized SVD function (without using JacobiSVD)
// std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> svd(const Eigen::MatrixXd &M) {
//   int dim = std::min(M.cols(), M.rows());
//   
//   auto MTM = M.transpose() * M;
//   Eigen::EigenSolver<Eigen::MatrixXd> es;
//   es.compute(MTM, true);
//   
//   Eigen::MatrixXd V = es.eigenvectors().real().block(0,0,M.cols(),dim);
//   Eigen::VectorXd S = es.eigenvalues().real().array().sqrt().block(0,0,dim,1);
//   
//   Eigen::MatrixXd U = M * V;
//   for (int i = 0; i < U.cols(); ++i) {
//     U.col(i).normalize();
//   }
//   return {U, S, V};
// }

// // [[Rcpp::export("rsvd_cpp")]]
// Eigen::MatrixXd rsvd(const Eigen::Map<Eigen::MatrixXd> &M, int k) {
//   // Randomized SVD: <doi:10.1137/090771806>
//   Eigen::MatrixXd Omega = init(M.cols(), k * 2);
//   Eigen::MatrixXd Q = qrq(M * Omega);
//   
//   for (int i = 0; i < 2; ++i) {
//     Q = qrq(M * M.transpose() * Q);
//   }
//   Eigen::MatrixXd B = Q.transpose() * M;
//   auto [U, S, V] = svd(B);
//   
//   Eigen::MatrixXd U_sub = U.block(0,0,M.rows(),k);
//   Eigen::MatrixXd S_sub = S.block(0,0,k,k);
//   
//   // TODO: sign ambiguity
//   return U_sub * S_sub;
// }
// 
// int tt(SEXP x) {
//   Rcpp::Rcout << x << std::endl;
//   Rcpp::Rcout << typeid(x).name() << std::endl;
//   return 0;
// }
