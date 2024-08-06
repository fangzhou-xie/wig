
# test log-domain stablization



softmin <- function(z, eps) {
  # browser()
  min(z)-eps*log(sum(exp(-(z-min(z))/eps)))
}
# softmin
minrow <- function(C, eps) {
  # Cgt <- C - rep(1, nrow(C)) %*% t(g)
  f1 <- rep(0, nrow(C))
  for (i in 1:nrow(C)) {
    f1[i] <- softmin(C[i,], eps)
  }
  f1
}
mincol <- function(C, eps) {
  # Cft <- C - f %*% t(rep(1, ncol(C)))
  g1 <- rep(0, ncol(C))
  for (j in 1:ncol(C)) {
    g1[j] <- softmin(C[,j], eps)
  }
  g1
}
update_S <- function(C, f, g) {
  C - f %*% t(rep(1, ncol(C))) - rep(1, nrow(C)) %*% t(g)
}
update_f <- function(C, f, g, a, eps) {
  S <- update_S(C, f, g)
  f <- minrow(S, eps) + f + eps * log(a)
  f
}
update_g <- function(C, f, g, b, eps) {
  S <- update_S(C, f, g)
  g <- mincol(S, eps) + g + eps * log(b)
  g
} 

sinkhorn <- function(a, C, b, w, reg, maxIter = 10, zeroTol = 1e-6) {
  K <- exp(-C/reg)
  u <- rep(1, length(a))
  # u <- matrix(1, nrow = nrow(a), ncol = ncol(a))
  v <- rep(1, length(b))
  
  iter <- 1
  err <- 1000
  
  while (iter <= maxIter & err >= zeroTol) {
    u <- a / (K %*% v)
    # for (j in 1:ncol(a)) {
    #   u[,j] <- a[,j] / (K %*% v)
    # }
    v <- b / (t(K) %*% u)
    
    # Pi in u and v
    PiKTu <- c(t(K) %*% u) ^ w
    print(PiKTu)
    print(sum(PiKTu))
    
    iter <- iter + 1
    err <- norm(a - u * (K %*% v), "2") + 
      norm(b - v * (t(K) %*% u), type = "2")
  }
  # print(iter)
  # diag(c(u)) %*% K %*% diag(c(v))
  # list(a = u * (K %*% v), b = v * (t(K) %*% u), iter = iter)
}

sinkhorn_stable <- function(a, C, b, w, reg, maxIter = 1000, zeroTol = 1e-8) {
  iter <- 1
  err <- 1000
  
  # init dual: f, g
  # f <- matrix(0, nrow = nrow(a), ncol = ncol(a))
  # f <- log(rep(1, nrow(C)))*reg
  # g <- log(rep(1, ncol(C)))*reg
  f <- matrix(0, nrow = nrow(a), ncol = ncol(aa))
  
  
  while (iter <= maxIter & err >= zeroTol) {
    # browser()
    # for (j in 1:ncol(a)) {
    #   f[,j] <- update_f(C, f[,j], g, a[,j], reg)
    # }
    f <- update_f(C, f, g, a, reg)
    g <- update_g(C, f, g, b, reg)
    
    print(f)
    
    # Pi in f and g
    S <- update_S(C, f, g)
    PiKTu <- exp(-(mincol(S, reg) + g)*w/reg)
    print(PiKTu)
    print(sum(PiKTu))
    
    P <- exp(-update_S(C, g, f)/reg)
    P1 <- P %*% rep(1, ncol(C))
    PT1 <- t(P) %*% rep(1, nrow(C))
    
    iter <- iter + 1
    err <- norm(a - P1, "2") + norm(b - PT1, "2")
  }
  # list(P = P, a = P1, b = PT1, iter = iter)
}

# a <- cbind(c(.1, .2 ,.7), c(.2,.3,.5))
a <- c(.1,.2,.7)
b <- c(.6, .2, .2)
C <- matrix(c(100,200,300,400,100,200,300,400,100), nrow = 3, ncol = 3)
sinkhorn(a, C/1000, b, .1, .1, maxIter = 4)
sinkhorn_stable(a, C/1000, b, .1, .1, maxIter = 4)



minus1 <- function(C, f, g) {
  for (i in 1:nrow(C)) {
    for (j in 1:ncol(C)) {
      
    }
  }
}
minus2 <- function(C, f, g) {
  t(C) - g %*% t(rep(1, nrow(C))) - rep(1, ncol(C)) %*% t(f)
}



Rcpp::sourceCpp(here::here("data-raw", "compare_eigen.cpp"))
Rcpp::sourceCpp(here::here("data-raw", "compare_arma.cpp"))

A <- matrix(rnorm(1000000), 1000, 1000)
B <- matrix(rnorm(1000000), 1000, 1000)

A <- matrix(rnorm(10000), 100, 100)
B <- matrix(rnorm(10000), 100, 100)

A2 <- A
B2 <- B
A2[1:30,1:30] <- NA
B2[30:60,30:60] <- NA

A <- matrix(NA, 100, 100)
B <- matrix(NA, 100, 100)

bench::mark(test_arma(A,B), test_eigen(A,B))
bench::mark(test_arma(A2,B2), test_eigen(A2,B2))

# BLIS is way faster than OpenBLAS

Rcpp::cppFunction("double softmin_cpp(arma::vec &z, double &eps) {
  double zmin = z.min();
  return zmin - eps * log(accu(exp(-(z-zmin)/eps)));
}", depends = "RcppArmadillo")

a <- rnorm(10)
softmin(a*1000, .1)
softmin_cpp(a*1000, .1)


Rcpp::cppFunction("arma::vec testslice(arma::cube &m) {
  // return m.tube(arma::span(0),arma::span(0));
  // Rcpp::Rcout << m.each_slice().each_col() << std::endl;
  return m.col_as_mat(0);
}
", depends = "RcppArmadillo")

arr <- array(dim = c(2,2,2))
arr[,,1] <- matrix(c(1,2,3,4), 2, 2)
arr[,,2] <- matrix(c(5,6,7,8), 2, 2)

testslice(arr)
arr[1,,]


Rcpp::sourceCpp(here::here("src", "utils.cpp"))

arr[,,1]

softmin_col(arr, 1)
mincol(arr[,,1], 1)
mincol(arr[,,2], 1)

softmin_row(arr, 1)
minrow(arr[,,1], 1)
minrow(arr[,,2], 1)





sen <- c("this is a sentence", "this is another sentence")
toks <- tokenizers::tokenize_words(sen)
model <- word2vec::word2vec(toks, min_count = 1)
dict <- rownames(as.matrix(model))

Rcpp::sourceCpp(here::here("data-raw", "doc2dist.cpp"))
docmat <- doc2dist(toks, dict)
rownames(docmat) <- dict
docmat

append(toks, list(c("this", "is", "jslf")))

doc2dist(append(toks, list(c("this", "is", "jslf"))), dict)

# docdict2dist(toks, dict)
# 
# bench::mark(
#   doc2dist(toks, dict), 
#   docdict2dist(toks, dict)
# )




a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
w <- c(.7, .2, .1)
C <- matrix(c(.1, .6, .8, .3), ncol = 2)

Rcpp::sourceCpp(here::here("src", "dsink_log.cpp"))

sol <- dsink_log(a, C, w, .1, maxIter = 2)
sol






date_list <- vector("list", 10)
date_list[[1]] <- 1:4
date_list

d1 <- as.Date("2013-02-12")

seq.Date(d1, d1+30, "1 day")
seq(d1, d1+30, "1 day")

bench::mark(
  seq.Date(d1, d1+30, "1 day"),
  seq(d1, d1+30, "1 day"),
  min_time = 10
)





# test col/row loop performance
Rcpp::sourceCpp(here::here("data-raw", "test_arma_rowcol.cpp"))
Rcpp::sourceCpp(here::here("data-raw", "test_eigen_rowcol.cpp"))

m <- matrix(rnorm(1000000), 1000, 1000)

bench::mark(
  test_col(m),
  test_row(m),
  test_row_t(m),
  test_eigen_col(m),
  test_eigen_row(m)
)

bench::mark(
  test_arma_load(m),
  test_eigen_load1(m),
  test_eigen_load2(m)
)



Rcpp::sourceCpp(here::here("data-raw", "test_softmin.cpp"))

softmin1_col(m, .1)[1:8]
softmin2_col(m, .1)[1:8]

bench::mark(
  softmin1_col(m, .1),
  softmin2_col(m, .1),
  min_time = 10
)


Rcpp::sourceCpp(here::here("data-raw", "test_var.cpp"))

m <- matrix(rnorm(100000000), 10000, 10000)

bench::mark(
  test_var1(m), 
  test_var2(m),
  test_var3(m),
  min_time = 5
)


Rcpp::sourceCpp(here::here("src", "dsink_log.cpp"))

dsink_log(a, C, w, .1, maxIter = 1)

a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
w <- c(.6, .2, .2)
C <- matrix(c(.1, .6, .5, .2), ncol = 2)

sol <- dsink_log(a, C, w, .1, zeroTol = 1e-6)
dsink_log(a, C, w, .1, zeroTol = 1e-6)
dsink_log(a, C, w, .01, zeroTol = 1e-6)
dsink_log(a, C, w, .001, zeroTol = 1e-8, maxIter = 10000)
colSums(sol$A)
dsink(a, C, w, .001)

# FIXME: when `eps` is too small the solution is WRONG!
# WHY????


Rcpp::cppFunction("arma::mat softmin_row(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  arma::vec ones_N = arma::ones(C.n_cols);
  arma::mat S_row(C.n_rows, f.n_cols, arma::fill::zeros);
  
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t i = 0; i < C.n_rows; ++i) {
      arma::vec S_i = S.row(i).t();
      S_row(i,k) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
    }
  }
  return S_row;
}", depends = "RcppArmadillo")

Rcpp::cppFunction("arma::mat softmin_col(arma::mat &C, arma::mat &f, arma::mat &g, const double eps) {
  // compute S first: S = C - f * 1^T - 1 * g^T
  arma::vec ones_N = arma::ones(C.n_rows);
  arma::mat S_col(C.n_cols, f.n_cols, arma::fill::zeros);
  
  for (size_t k = 0; k < f.n_cols; ++k) {
    arma::mat S = C - f.col(k) * ones_N.t() - ones_N * g.col(k).t();
    for (size_t j = 0; j < C.n_cols; ++j) {
      arma::vec S_j = S.col(j);
      S_col(j,k) = S_j.min() - eps * log(accu(exp(-(S_j-S_j.min())/eps)));
    }
  }
  return S_col;
}", depends = "RcppArmadillo")

# Rcpp::cppFunction("arma::vec softmin_row_d(arma::mat &C, arma::vec &log_d, const double eps) {
#   arma::vec ones_N = arma::ones(C.n_cols);
#   arma::vec r_row(C.n_rows, arma::fill::zeros);
# 
#   arma::mat S = C - ones_N * log_d.t();
#   for (size_t i = 0; i < C.n_rows; ++i) {
#     auto S_i = S.row(i).t();
#     r_row(i) = S_i.min() - eps * log(accu(exp(-(S_i-S_i.min())/eps)));
#   }
#   return r_row;
# }", depends = "RcppArmadillo")

f <- matrix(0, nrow = nrow(a), ncol = ncol(a))
g <- matrix(0, nrow = nrow(a), ncol = ncol(a))


a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
w <- c(.7, .2, .1)
C <- matrix(c(.1, .8, .6, .3), ncol = 2)

softmin_row(C, f, g, .1)
softmin_col(C, f, g, .1)

v <- matrix(1, nrow = nrow(a), ncol = ncol(a))
K <- exp(-C/.1)



d <- rep(1, nrow(C))
log_d <- log(d)

# update f
f <- softmin_row(C,f,g,.1) + f + .1*log(a)
# update u
u <- a / (K %*% v)
exp(f/.1)
u

# update b
b <- d * matrixStats::rowProds((function(m, v) {
  mout <- matrix(0, nrow = nrow(m), ncol = ncol(m))
  for (i in 1:nrow(m)) {
    mout[i,] <- m[i,] ^ w
  }
  mout
})(t(K) %*% u, w))
# update log_b
log_b <- log_d - c(((softmin_col(C,f,g,.1)+g) %*% w)/.1)
b
exp(log_b)


# update g
g <- softmin_col(C,f,g,.1) + g + .1*log(b)
# update v
v <- b / (t(K) %*% u)
exp(g/.1)
v

# update d
d <- sqrt(d * b / (K %*% d))
# update log_d
log_d <- (log_d + log_b + c(softmin_row(C, matrix(0, nrow = nrow(C), ncol = 1), matrix(log_d), .1))/.1)/2
# log_d <- c(softmin_row(C, matrix(0, nrow = nrow(C), ncol = 1), matrix(log_d), .1))
d
exp(log_d)


log(K %*% d)
-softmin_row(C, matrix(0, nrow = nrow(C), ncol = 1), matrix(log_d), .1)/.1



 b <- c(.2,.8)

f <- softmin_row(C,f,g,.1) + f + .1*log(a)
g <- softmin_col(C,f,g,.1) + g + .1*log(b)

softmin_col(C, u, g, .1)
softmin_row(C, u, g, .1)
softmin_row_r(C, u, g, .1)

softmin_col(C, f, g, .1)

exp((softmin_row(C,f,g,.1) + f + .1*log(a))/.1)

u <- matrix(1, nrow = nrow(a), ncol = ncol(a))
v <- matrix(1, nrow = nrow(a), ncol = ncol(a))
K <- exp(-C/.1)

u = a / (K %*% v)

exp(-(softmin_row(C,f,g,.1))/.1)


softmin_row(C,f,g,.1)
softmin_col(t(C),g,f,.1)
bench::mark(
  softmin_row(C,f,g,.1),
  softmin_col(C,f,g,.1),
  check = FALSE
)




Rcpp::sourceCpp(here::here("data-raw", "test_logKT.cpp"))

test_logKT(a, C, w, .1)





a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
w <- c(.5, .2, .3)
C <- matrix(c(.1, .6, .5, .2), ncol = 2)

Rcpp::sourceCpp(here::here("src", "test_dsink_log.cpp"))

test_dsink(a, C, w, .001)





a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
w <- c(.5, .2, .3)
C <- matrix(c(.1, .6, .5, .2), ncol = 2)

f <- matrix(0, nrow(a), ncol(a))
g <- matrix(0, nrow(a), ncol(a))

K <- exp(-C/.1)
v <- matrix(1, nrow(a), ncol(a))

u <- a / (K %*% v)

exp((softmin_row(C, f, g, .1) + f + .1*log(a))/.1)




Rcpp::sourceCpp(here::here("src", "softmin.cpp"))
softmin_col(C, .1)
softmin_col(C, .01)
softmin_col(C, .001)
softmin_row(C, .1)
softmin_row(C, .01)
softmin_row(C, .001)



Rcpp::sourceCpp(
  here::here("data-raw", "test_autodiff.cpp"), 
  verbose = FALSE,
  showOutput = FALSE,
  echo = FALSE
)

a <- c(1,2)
b <- c(2,3)
test_autodiff(a,b)




Rcpp::cppFunction("arma::mat diagmat(arma::mat A) {
  Rcpp::Rcout << diagmat(A) << std::endl;
  Rcpp::Rcout << diagmat(A, 2) << std::endl;
  return diagmat(A, 2);
}
", depends = "RcppArmadillo")

A <- matrix(c(1,2,3,4), 2, 2)
diagmat(A)
