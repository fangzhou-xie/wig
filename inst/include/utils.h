
#include <RcppArmadillo.h>

arma::mat init(int nrow, int ncol);

void softmax(arma::mat &a);

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
