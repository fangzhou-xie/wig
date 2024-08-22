
Rcpp::sourceCpp(here::here("src", "test.cpp"))

M <- matrix(rnorm(10000), 100, 100)


bench::mark(
  f1(M),
  f2(M)
)

Rcpp::sourceCpp(here::here("src", "sinkhorn.cpp"))

Rcpp::sourceCpp(here::here("src", "barycenter.cpp"))

A <- cbind(
  c(.2, .2, .5, .1), c(.3, .4, .2, .1), c(.5, .4, .05, .05)
)
C <- cbind(c(1,2,3,4), c(2,3,4,5), c(3,4,5,6), c(4,5,6,7))

barycenter(A, C, c(.1,.1,.8), 2, FALSE)
barycenter_log(A, C, rep(1/3, 3), 2, FALSE, maxIter = 10)
