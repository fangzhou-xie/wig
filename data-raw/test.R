
a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
b <- c(.8, .2)
M <- matrix(c(.1, .5, .5, .1), ncol = 2)

a <- matrix(
  c(.4, .3, .2, .5, .2, .2, .1, .5, .6),
  byrow = TRUE, nrow = 3
)
b <- c(.3, .3, .4)
M <- matrix(
  c(.1, .3, .2, .5, .1, .5, .4, .2, .1),,
  byrow = TRUE, nrow = 3
)
lbd <- c(.2, .2, .6)

sol <- sinkhorn(a, M, b, lbd, .1, 1000)

u <- sol$u
v <- sol$v
K <- sol$K

u * (K %*% v)
v * (t(K) %*% u)

sol$gradient_a
sol$gradient_alpha

bench::mark(
  sinkhorn(a[,c(2,1,3)], M, b, .1, 2)
)


sinkhorn(a[,1], M, b, .1, 1000)$gradient
sinkhorn(a[,2], M, b, .1, 1000)$gradient
sinkhorn(a[,3], M, b, .1, 1000)$gradient

sol <- sinkhorn(a[,1], M, b, .1, 1000)
u <- sol$u
v <- sol$v
K <- sol$K
u * (K %*% v)
v * (t(K) %*% u)
sol$iter
sol$gradient


sinkhorn(a, M, b, .1, 1000)$gradient


Rcpp::sourceCpp(here::here("src", "t.cpp"))

testfunc(M, b)

f <- function(a) sinkhorn(a, M, b, .5, 10)
pracma::fderiv(
  Vectorize(f),
  a
)

pracma::numderiv(f, t(a[,1]))


Rcpp::cppFunction(
"
Eigen::MatrixXd softmax(Eigen::MatrixXd &a) {
  Eigen::VectorXd acol(a.rows());
  Eigen::MatrixXd out(a.rows(), a.cols());
  for (int i = 0; i < a.cols(); ++i) {
    acol = a.col(i).array().exp();
    out.col(i) = acol / acol.array().sum();
  }
  return out;
}
",
depends = "RcppEigen"
)

softmax(M)

Rcpp::cppFunction(
"
Eigen::VectorXd softmax(Eigen::VectorXd a) {
  auto aexp = a.array().exp();
  Eigen::VectorXd out = aexp / aexp.array().sum();
  return out;
}
",
depends = "RcppEigen"
)

softmax(b)

Rcpp::cppFunction(
  "
std::vector<double> rcpp_rnorm(int n, double m, double sd) {
  Rcpp::Function f(\"rnorm\");
  auto v = f(n, Rcpp::Named(\"mean\") = m, Rcpp::Named(\"sd\") = sd);
  std::vector<double> v_vec = Rcpp::as<std::vector<double>>(v);
  return v_vec;
}
",
depends = "RcppEigen"
)

set.seed(1)
rnorm(5)
set.seed(1)
rcpp_rnorm(5, 0, 1)


Rcpp::cppFunction(
  "
  Eigen::MatrixXd slice(Eigen::MatrixXd docs, int batch_size, int batch_id) {
  // given batch_size and batch_id, calculate indices to be used for block API
  int batch_start_id = batch_size * batch_id;
  int batch_size_adj = docs.cols() - batch_size - batch_start_id;
  batch_size = batch_size_adj >= 0 ? batch_size : batch_size + batch_size_adj;
  return docs.block(0, batch_start_id, docs.rows(), batch_size);
}
  ",
depends = "RcppEigen"
)

slice(M, 2, 0)
slice(M, 2, 1)

Rcpp::cppFunction(
  "
  Eigen::MatrixXd euclidean(Eigen::Map<Eigen::MatrixXd> a) {
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
",
depends = "RcppEigen"
)



data(brussels_reviews, package = "udpipe")

x <- subset(brussels_reviews, language == "nl")
x <- tolower(x$feedback)
head(x)

model <- word2vec::word2vec(x, type = "cbow", dim = 15, iter = 20)
embedding <- as.matrix(model)

dim(embedding)
embedding[1:10, 1:5]
dim(cov(embedding))

euclidean(embedding)[1:10, 1:10]

euclidean(embedding[1:10,])



# test standardize
Rcpp::cppFunction(
  "
  Eigen::VectorXd standardize_cpp(Eigen::VectorXd v) {
  double m = v.array().mean();
  double s = std::sqrt((v.array() - m).pow(2).sum() / (v.size() - 1));
  return (v.array() - m)/s + 100;
}
",
depends = "RcppEigen"
)

v <- (1:100)/2
sv <- standardize_cpp(v)
mean(sv)
sd(sv)

do.call(word2vec::word2vec, args = list(x = x, type = "cbow", dim = 15, iter = 20))

l <- list(type = "cbow", dim = 15, iter = 20)
append(list(x = x), l)

model <- do.call(word2vec::word2vec, args = append(list(x = x), l))


M <- rbind(c(20, 10, 15, 0, 2),
           c(10, 5, 8, 1, 0),
           c(0, 1, 2, 6, 3),
           c(1, 0, 0, 10, 5))

M_svd <- svd(M)
M_svd$u[,1:k] %*% diag(M_svd$d[1:k], k, k)

colSums(M_svd$u)
norm(M_svd$u[,1])


Ms <- Matrix::Matrix(M, sparse=TRUE)
M_svd <- sparsesvd::sparsesvd(Ms, rank = 2L)

dim(M_svd$u)
dim(M_svd$v)
M_svd$d
diag(M_svd$d)
diag(1)

M_svd <- sparsesvd::sparsesvd(Ms, rank = 1L)
str(M_svd$d)
diag(M_svd$d)

M2 <- M_svd$u %*% diag(M_svd$d) %*% t(M_svd$v)
# diag(M_svd$d) %*% t(M_svd$v)
M_svd$u %*% diag(M_svd$d)

M %*% M_svd$v

M_svd <- svd(M)

M_svd$u %*% matrix(M_svd$d[1]) %*% matrix(M_svd$v[,1], ncol = 1)
M_svd$u[,1] %*% matrix(M_svd$d[1])


# benchmark several Truncated SVD methods

svd1 <- function(M, k = 1) {
  M_svd <- svd(M)
  M_svd$u[,1:k] %*% diag(M_svd$d[1:k], k, k)
}

svd2 <- function(M, k = 1) {
  Ms <- Matrix::Matrix(M, sparse=TRUE)
  M_svd <- sparsesvd::sparsesvd(Ms, rank = as.integer(k))
  M_svd$u %*% diag(M_svd$d[1:k], k, k)
}

# Rcpp::cppFunction(
#   "
#   Eigen::MatrixXd svd3(Eigen::MatrixXd M, int k = 1) {
#   using Eigen::MatrixXd;
#   using Eigen::ComputeThinU;
#   using Eigen::ComputeThinV;
#   Eigen::JacobiSVD<MatrixXd> svd;
#   svd.compute(M, ComputeThinU | ComputeThinV);
# 
#   MatrixXd U_new = svd.matrixU().block(0,0,M.rows(),k);
#   MatrixXd S = svd.singularValues().asDiagonal();
#   MatrixXd S_new = S.block(0,0,k,k);
#   MatrixXd M_svd = U_new * S_new;
#   // test signs
#   int s = (M_svd.array() > 0).count();
#   return (static_cast<double>(s) / M.rows() > .5) ? M_svd : -M_svd;
# }", depends = "RcppEigen"
# )



svd4 <- function(M, k = 1) {
  fsvd <- bootSVD::fastSVD(M)
  fsvd$u[,1:k] %*% diag(fsvd$d[1:k], k, k)
}

M <- rbind(c(20, 10, 15, 0, 2),
           c(10, 5, 8, 1, 0),
           c(0, 1, 2, 6, 3),
           c(1, 0, 0, 10, 5))

svd1(M)
svd2(M)
# svd3(M)
svd4(M)
rsvd(M)


M <- matrix(rnorm(5000000), ncol = 50, nrow = 100000)
M <- matrix(rnorm(50000), ncol = 5)

# profvis::profvis(wig::rsvd(M))

bench::mark(
  svd1(M),
  svd2(M),
  # svd3(M),
  svd4(M),
  wig::rsvd(M),
  check = FALSE
)

svd1(M)[1:10,]
svd2(M)[1:10,]
svd3(M)[1:10,]

# svd3 using Eigen::JacobiSVD is way faster (but the sign might be wrong)

M <- rbind(
  c(1, 2, 3),
  c(4, 5, 6)
)

svd(M)
q <- qr(M)$qr
t(q) %*% q
q %*% t(q)
t(q)
solve(q)

qr(M, tol = 1e-7, LAPACK = FALSE)

QR <- qr(M)
q <- qr.Q(QR)
r <- qr.R(QR)

t(q) %*% q
q %*% t(q)

matlib::GramSchmidt(M)

rSVD <- function(M, k = 1) {
  # implement randomized SVD algorithm in R
  # Omega: ncol(M)
  Omega <- matrix(rnorm(ncol(M) * 2 * k), nrow = ncol(M), ncol = 2 * k)
  Q <- qr.Q(qr(M %*% Omega))
  for (i in 1:2) {
    # G <- qr.Q(qr(t(M) %*% Q))
    # Q <- qr.Q(qr(M %*% G))
    Q <- qr.Q(qr(M %*% t(M) %*% Q))
  }
  B <- t(Q) %*% M
  B_svd <- svd(B)
  U <- B_svd$u
  S <- diag(B_svd$d, k, k)
  # V <- B_svd$v
  U <- Q %*% U
  U[,1:k] %*% S
}



svd1(M, k = 1)[1:10,]
rSVD(M, k = 1)[1:10,]

bench::mark(
  svd1(M, k = 4),
  rSVD(M, k = 4),
  check = FALSE
)


Rcpp::cppFunction(
  "
  void qrexample(Eigen::MatrixXd M) {
  Eigen::HouseholderQR<Eigen::MatrixXd> QR(M.rows(), M.cols());
  QR.compute(M);
  // auto QR = M.householderQr();
  Eigen::MatrixXd Q = QR.householderQ();
  // Rcpp::Rcout << QR.matrixQR() << std::endl;
  Rcpp::Rcout << Q << std::endl;
  // QR.householderQ()
}
  ",
depends = "RcppEigen"
)

Rcpp::cppFunction(
  "
  Eigen::MatrixXd rsvd(Eigen::MatrixXd M, int k) {
  // Randomized SVD: <doi:10.1137/090771806>
  Eigen::MatrixXd Omega = init(M.cols(), k * 2);
  Eigen::MatrixXd Q = (M * Omega).householderQr().householderQ();
  
  for (int i = 0; i < 2; ++i) {
    Q = (M * M.transpose() * Q).householderQr().householderQ();
  }
  Eigen::MatrixXd B = Q.transpose() * M;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd;
  svd.compute(B, Eigen::ComputeThinU);
  Eigen::MatrixXd U = Q * svd.matrixU();
  Eigen::MatrixXd S = svd.singularValues().asDiagonal();
  
  Eigen::MatrixXd U_sub = U.block(0,0,M.rows(),k);
  Eigen::MatrixXd S_sub = S.block(0,0,k,k);
  return U_sub * S_sub;
}
", depends = "RcppEigen"
)

M <- rbind(
  c(1, 2, 3),
  c(4, 5, 6)
)
qrexample(M)
qr.Q(qr(M))

rsvd(M, 5)

svd(M)


# test performance of JacobiSVD in Eigen vs R `svd` using LAPACK
Rcpp::cppFunction(
  "
  Rcpp::List svd_eigen(Eigen::Map<Eigen::MatrixXd> M) {
  using Eigen::MatrixXd;
  using Eigen::ComputeThinU;
  using Eigen::ComputeThinV;
  
  Eigen::JacobiSVD<MatrixXd> svd;
  svd.compute(M, ComputeThinU | ComputeThinV);
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named(\"d\") = svd.singularValues(),
    Rcpp::Named(\"u\") = svd.matrixU(),
    Rcpp::Named(\"v\") = svd.matrixV()
  );
  return res;
}
  ", depends = "RcppEigen"
)

Rcpp::cppFunction(
  "
  Rcpp::List svd_r1(Rcpp::NumericMatrix M) {
  Rcpp::Function svd(\"svd\");
  auto res = svd(M);
  return res;
}
  "
)

Rcpp::cppFunction(
  "
  Rcpp::List svd_r2(Eigen::Map<Eigen::MatrixXd> M) {
  Rcpp::NumericMatrix M2 = Rcpp::wrap(M);
  Rcpp::Function svd(\"svd\");
  auto res = svd(M2);
  return res;
}
  ", depends = "RcppEigen"
)



M <- rbind(
  c(1, 2, 3),
  c(4, 5, 6)
)
svd_eigen(M)
svd_r1(M)
svd_r2(M)
svd(M)

M <- matrix(rnorm(500000), ncol = 50)

svd_eigen(M)$u[1:5,]
svd_r(M)$u[1:5,]
svd(M)$u[1:5,]

bench::mark(
  svd_eigen(M),
  svd_r1(M),
  svd_r2(M),
  svd(M),
  check = FALSE
)

QR <- qr(M)
qr(M)
qr(M, LAPACK = TRUE)
Matrix::qr(M)


Rcpp::cppFunction(
  "
  void test_call(SEXP cl) {
  Rcpp::Rcout << cl << std::endl << std::endl;
}
  "
)

test_call(quote(1 + x))



Rcpp::sourceCpp(here::here("src", "test_rcall.cpp"))

# qr_r4 <- function(M, f) f(M)

qr_r1(M, function(x) qr.Q(qr(x)))
qr_r2(M, function(x) qr.Q(qr(x)))
qr_r3(M, function(x) qr.Q(qr(x)))
qr_r4(M, function(x) qr.Q(qr(x)))

bench::mark(
  qr_r1(M, function(x) qr.Q(qr(x))),
  qr_r2(M, function(x) qr.Q(qr(x))),
  qr_r3(M, function(x) qr.Q(qr(x))),
  qr_r4(M, function(x) qr.Q(qr(x)))
)


Rcpp::cppFunction(
  "
  Rcpp::List svd_arma(arma::mat M) {
  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, M);
  
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named(\"d\") = s,
    Rcpp::Named(\"u\") = U,
    Rcpp::Named(\"v\") = V
  );
  return res;
}
  ", depends = "RcppArmadillo"
)

svd_arma(M)
svd(M)

bench::mark(
  svd_arma(M),
  svd(M),
  check = FALSE
)


# svd speed plots



time_df <- tidyr::expand_grid(
  cols = seq(10, 100, 20),
  rows = seq(1000, 10000, 2000)
) %>% 
  purrr::pmap(
    function(cols, rows, k) {
      M <- matrix(rnorm(cols * rows), ncol = cols, nrow = rows)
      bench::mark(svd1 = svd1(M, 5), svd2 = svd2(M, 5), svd4 = svd4(M, 5), check = FALSE) %>% 
        select(expression, median) %>% 
        mutate(cols = !!cols, rows = !!rows, k = !!k)
    }
  )

M <- rbind(c(20, 10, 15, 0, 2),
           c(10, 5, 8, 1, 0),
           c(0, 1, 2, 6, 3),
           c(1, 0, 0, 10, 5))
M <- matrix(rnorm(500000), ncol = 50)

rsvd(M, 6)



Rcpp::sourceCpp(here::here("src", "test_eigen.cpp"))
# eigenvv(t(M) %*% M)
# eigen(t(M) %*% M)$vectors
# svd(M)$d ^2


svd1(M)
svd2(M)
svd(M)

bench::mark(
  svd1(M),
  svd2(M),
  svd(M),
  check = FALSE
)

qrq(M)
qr.R(qr(M))
qr.Q(qr(M))

qr.Q(qr(M)) %*% qr.R(qr(M))

dim(M)
dim(qr.Q(qr(M)))

dim(t(M))
dim(qr.Q(qr(t(M))))


tsvd(M, 3, "sklearn")
tsvd(M, 3, "auto")
tsvd(M, 3, "none")

Rcpp::cppFunction(
  "
  int tt(SEXP x) {
  Rcpp::Rcout << x << std::endl;
  Rcpp::Rcout << typeid(x).name() << std::endl;
  return 0;
}
  "
)
tt(quote(1+x))




data(brussels_reviews, package = "udpipe")

x <- subset(brussels_reviews, language == "nl")
x <- tolower(x$feedback)
head(x)

model <- word2vec::word2vec(x, type = "cbow", dim = 15, iter = 20)
embedding <- as.matrix(model)

embedding[1:10,1:5]

distmat <- euclidean_cpp(embedding)
dim(distmat)
distmat[1:10,1:5]

model$data %>% class()

class(model$data)
class(model$data[[1]])
model$data[[1]]

con <- file(model$data[[1]], "r")
readLines(con, n = 2)


# word2vec::txt_clean_word2vec(x[1])
summary(model, type = "vocabulary")


james <- paste0(
  "The question thus becomes a verbal one\n",
  "again; and our knowledge of all these early stages of thought and feeling\n",
  "is in any case so conjectural and imperfect that farther discussion would\n",
  "not be worth while.\n",
  "\n",
  "Religion, therefore, as I now ask you arbitrarily to take it, shall mean\n",
  "for us _the feelings, acts, and experiences of individual men in their\n",
  "solitude, so far as they apprehend themselves to stand in relation to\n",
  "whatever they may consider the divine_. Since the relation may be either\n",
  "moral, physical, or ritual, it is evident that out of religion in the\n",
  "sense in which we take it, theologies, philosophies, and ecclesiastical\n",
  "organizations may secondarily grow.\n"
)

tok <- tokenizers::tokenize_words(x, stopwords = stopwords::stopwords("en"))
model <- word2vec::word2vec(tok, dim = 5)
embedding <- as.matrix(model)
dim(embedding)
embedding[1:5,1:5]

dict <- rownames(embedding)
dict[1:10]

dist_mat <- euclidean(embedding)
dist_mat[1:10,1:5]

tok[[1]]


Rcpp::cppFunction(
  "
  Eigen::MatrixXd docdict2dist(
  const std::vector<std::string> &dict,
  const Rcpp::List &docs_token
) {
  // for each doc, create a vector (column) as the empirical distribution
  // of the tokens
  Eigen::MatrixXd docsmat(dict.size(), docs_token.size());
  docsmat.setZero();
  
  for (int j = 0; j < docs_token.size(); ++j) {
    std::vector<std::string> d = docs_token[j];
    for (auto &token : d) {
      auto d_itr = std::find(dict.begin(), dict.end(), d);
      auto d_id = std::distance(dict.begin(), d_itr);
      docsmat(d_id, j) += 1;
    }
    docsmat.col(j) = docsmat.col(j) / docsmat.col(j).array().sum();
  }
  return docsmat;
}
  ", depends = "RcppEigen"
)

dict <- dict[1:10]
# doc <- tok[[1]]
doc <- c("bank", "best", "6", "best")

docmat <- matrix(0, 10, 3)

doc1_inds <- unlist(Map(function(x) which(dict == x), doc), FALSE, FALSE)
docmat[doc1_inds, 1] <- 1

dictdf <- data.frame(dict)

table(doc, deparse.level = 0) %>% 
  as.data.frame()

doc_freq_df <- merge(dictdf, as.data.frame(table(doc)), 
                     by.x = "dict", by.y = "doc", all.x = TRUE)
# doc_freq_df[is.na(doc_freq_df$Freq)]
doc_freq_df2 <- transform(doc_freq_df, Freq = ifelse(is.na(Freq), 0, Freq))

doc_freq_df2$Freq

docs <- list(c("bank", "best", "6", "best"), c("balkon", "best", "cafe"))
docdict2dist(docs, dict)
docdict2dist2(docs, dict)

bench::mark(
  docdict2dist(docs, dict),
  docdict2dist2(docs, dict)
)

doc_dist <- docdict2dist(tok, dict)
dim(doc_dist)
doc_dist[1:20, 1:10]

colSums(doc_dist)[1:100]


t(tsvd(t(M)))
tsvd(t(M), 3, "auto")
tsvd(t(M), 3, "sklearn")
tsvd(t(M), 3, "none")

xts::split.xts(as.Date("1970-01-01") + 1:10, "weeks")
dts <- as.Date("1970-01-01") + 1:100
dts_end <- xts::endpoints(dts, "months")

dts[30]

for (i in 2:length(dts_end)) {
  print(dts[dts_end[i-1]+1])
  print(dts[dts_end[i]])
}

ss <- function(v) {
  (v - mean(v))/sd(v) + 100
}

v <- (1:1000)/2
ss(v)

standardize_cpp(v)

bench::mark(
  ss(v),
  standardize_cpp(v)
)


sqlite_path <- "/home/fangzhou/Dropbox/Rutgers/projects/wig-r-package/headlines.db"
con <- DBI::dbConnect(RSQLite::SQLite(), sqlite_path)
DBI::dbListTables(con)

headlines <- tbl(con, "headlines_reduce")

headlines %>% 
  select(Date = date, headlines = title) %>% 
  collect() %>% 
  write_tsv(
    here::here("data", "nyt_epu_headlines.tsv")
  )

DBI::dbDisconnect(con, shutdown = TRUE)

headlines <- read_tsv(here::here("data", "nyt_epu_headlines.tsv")) %>% 
  as.data.frame()
head(headlines)
usethis::use_data(headlines, overwrite = TRUE)


head(wig::headlines)

# headlines <- as.data.frame(headlines)
# head(headlines)

wig_index <- wig::wig(wig::headlines$Date, wig::headlines$headlines)





Rcpp::sourceCpp(here::here("data-raw", "arma_test.cpp"))

m <- matrix(1:12, 3, 4)
v <- c(1,2,3)

tt(m, v)
tt2(m, v)

bench::mark(
  tt(m,v), tt2(m,v)
)

tt3(12, 0, 1)




softmin <- function(x, eps) {
  - eps * log(sum(exp( -x/eps)))
}

x <- c(1,2,3,4,5)
min(x)
softmin(x, .01)



Rcpp::sourceCpp(here::here("data-raw", "sinkhorn2.cpp"))

# a <- matrix(c(.4, .6, .3, .7, .2, .8), byrow = FALSE, nrow = 2)
a <- c(.4, .6)
b <- c(.8, .2)
M <- matrix(c(.1, .5, .5, .1), ncol = 2)

sol <- sinkhorn_scaling(a, M, b, .0001, maxIter = 20000)
sol$iter
sol$u * (sol$K %*% sol$v)
sol$v * (t(sol$K) %*% sol$u)



K <- matrix(1:4, 2, 2)
A <- matrix(1:2/5, 2, 1)
B <- matrix(6:7/2, 2, 1)

(K %*% A) * B
K %*% (A * B)

K %*% A * B


D <- cbind(
  c(.2, .8),
  c(.7, .3)
)
M <- matrix(c(.1, .5, .5, .1), ncol = 2)

sol <- sinkhorn_barycenter(D, M, .1)

sol$v * (sol$K %*% sol$u)
sol$p


sum(D[,1] * A)
sum(D[,2] * A)

diag(c(A)) %*% D


Rcpp::sourceCpp(here::here("src", "dsink.cpp"))
D <- cbind(
  c(.2, .8),
  c(.7, .3)
)
M <- matrix(c(.1, .5, .5, .1), ncol = 2)

sol <- dsink(D, M, c(.55, .45), .1)

D
sol$u * (sol$K %*% sol$v)

sol$v * (t(sol$K) %*% sol$u)
sol$alpha

sol$K %*% diag(c(sol$d))


Rcpp::sourceCpp(here::here("src", "testcube.cpp"))

a <- cbind(c(.2, .8), c(.4, .6))
b <- matrix(1:4, 2, 2)

dim3(a,2*a,b)


arr <- array(dim = c(2,2,2))
arr[,,1] <- a
arr[,,2] <- 2*a

arr[,,1] %*% b
arr[,,2] %*% b


einsum::einsum("it,tjk->ijk", b, arr)
dim3(b, arr)


b <- array(dim = c(3,4,2))
b[,,1] <- matrix(1:12, 3, 4)
b[,,2] <- matrix(1:12, 3, 4)/2
a <- cbind(c(.1, .2, .3), c(.2,.3,.4))

einsum::einsum("ik,ijk->ij", 1/a, b)

G <- matrix(nrow = 3, ncol = 4)

for (i in 1:nrow(G)) {
  for (j in 1:ncol(G)) {
    G_ij <- 0
    for (k in 1:dim(b)[3]) {
      G_ij <- G_ij + b[i,j,k] / a[i,k]
    }
    G[i,j] <- G_ij
  }
}
G




# test jacobian and gradient
Rcpp::sourceCpp(here::here("src", "dsinkgrad.cpp"))

v <- c(.2,.9)
softmax(v)
v

docs <- cbind(
  c(.2,.2,.6),
  c(.2,.1,.7),
  c(.6,.2,.2)
)
A <- rbind(c(.4,.3,.2),c(.5,.2,.2),c(.1,.5,.6))
C <- rbind(c(.1,.3,.2),c(.5,.1,.5),c(.4,.2,.1))
lbd <- cbind(
  c(.2,.2,.6),
  c(.2,.5,.3),
  c(.6,.2,.2)
)

docs <- matrix(c(.2,.2,.6), ncol = 1)
lbd <- matrix(c(.2,.2,.6), ncol = 1)

dsinkgrad(docs, 0L, A, C, lbd, .1)
dsinkgrad(docs, 1L, A, C, lbd, .1)



a <- c(.4,.3,.2)
b <- c(.2,.2,.6)
C <- rbind(c(.1,.3,.2),c(.5,.1,.5),c(.4,.2,.1))
reg <- .1

sinkgrad(a,C,b,reg,100)

A <- rbind(c(.4,.3,.2),c(.5,.2,.2),c(.1,.5,.6))
C <- rbind(c(.1,.3,.2),c(.5,.1,.5),c(.4,.2,.1))
lbd <- c(.2,.2,.6)
reg <- .1

sol <- dsinkgrad(A, C, lbd, reg, maxIter = 1)
sol$Jalpha_lbd
sol$Jalpha_A

dsinkgrad(A, C, lbd, reg, maxIter = 3)$Jalpha_lbd
dsinkgrad(A, C, lbd, reg, maxIter = 3)$Jalpha_A

dsinkgrad(A, C, lbd, reg)$Jalpha_lbd
dsinkgrad(A, C, lbd, reg)$Jalpha_A


Rcpp::sourceCpp(here::here("src", "armasize.cpp"))

test_size(A)



# test optimize
opt_sgd <- function(p, g, lr=.001) {
  p - rowSums(g) * lr / ncol(g)
}

p <- c(.1,.2,.3)
g <- cbind(c(.4,.5,.6), c(.1,.3,.2))

opt_sgd(p, g)
test_opt_sgd(p, g)


