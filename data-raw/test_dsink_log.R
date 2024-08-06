


A <- cbind(c(.4,.6), c(.3,.7), c(.2,.8))
C <- cbind(c(.1,.6), c(.8,.3))
w <- c(.7,.2,.1)


Rcpp::sourceCpp(here::here("src", "dsink_log.cpp"))

dsink_log(A,C,w,.1)
# bench::mark(dsink_log(A,C,w,.1))












library(data.table)
library(magrittr)
# library(dplyr)

# compare performance between data.table vs dplyr
bench::mark(
  data.table(mtcars)[, .(mpg)],
  mtcars %>% dplyr::select(mpg),
  dplyr::select(mtcars, mpg),
  check = FALSE
)


x <- data.table(Id  = c("A", "B", "C", "C"),
                X1  = c(1L, 3L, 5L, 7L),
                XY  = c("x2", "x4", "x6", "x8"),
                key = "Id")

y <- data.table(Id  = c("A", "B", "B", "D"),
                Y1  = c(1L, 3L, 5L, 7L),
                XY  = c("y1", "y3", "y5", "y7"),
                key = "Id")

y[x, on = "Id"]
dplyr::left_join(x, y, by = "Id")

bench::mark(
  y[x, on = "Id"],
  dplyr::left_join(x, y, by = "Id"),
  check = FALSE
)
