
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

sol <- wigr::sinkhorn(a, M, b, lbd, .1, 1000)

u <- sol$u
v <- sol$v
K <- sol$K

u * (K %*% v)
v * (t(K) %*% u)

sol$gradient_a
sol$gradient_alpha



seq(
  ymd("2015-03-01"),
  ymd("2015-03-01"),
  "1 month"
) %>% 
  days_in_month() %>% 
  unname()

mday(c(d1, d2)) <- 2

d1 <- ymd("2015-03-01")
d2 <- ymd("2015-05-01")
d_seq <- seq(d1, d2, "1 month")
d_days <- days_in_month(d_seq)

bench::mark(
  data.frame(d = d_seq, dd = d_days),
  tibble(d = d_seq, dd = d_days),
  check = FALSE
)
