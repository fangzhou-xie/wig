
#include <RcppArmadillo.h>

// arma::mat init(int nrow, int ncol);
// 
// void softmax(arma::mat &a);

// #include <Rcpp.h>
// #include <RcppEigen.h>
// 
// 
// std::vector<double> rnorm(int n, double m = 0, double sd = 1);
// Eigen::MatrixXd init(int nrow, int ncol);
// Eigen::VectorXd init(int nrow);
// 
// void softmax(Eigen::MatrixXd &a);
// void softmax(Eigen::VectorXd &a);

// Euclidean distance of input matrix
// Eigen::MatrixXd euclidean(const Eigen::Map<Eigen::MatrixXd> &a);
// Eigen::VectorXd standardize(const Eigen::Map<Eigen::VectorXd> &v);


// QR and SVD for randomized SVD (not used)
// Eigen::MatrixXd qrq(const Eigen::MatrixXd &M);
// std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> svd(const Eigen::MatrixXd &M);
// std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> rsvd(Eigen::MatrixXd M);


// identity matrix
arma::mat I(const int n) {
  arma::mat In = arma::mat(n, n, arma::fill::eye);
  return In;
}

// ones vector
arma::mat ones(const int n) {
  arma::vec onesn = arma::vec(n, arma::fill::ones);
  return onesn;
}

// commutation matrix
arma::mat commutation(int m, int n) {
  arma::uvec onetomtimesn = arma::linspace<arma::uvec>(0, m*n-1, m*n);
  arma::mat A = arma::reshape(arma::conv_to<arma::vec>::from(onetomtimesn), m, n);
  // arma::mat A = Avec.reshape(m, n);
  arma::uvec v = arma::conv_to<arma::uvec>::from(arma::vectorise(A.t()));
  arma::mat P(m*n, m*n, arma::fill::eye);
  P = P.submat(v, onetomtimesn);
  return P;
}
