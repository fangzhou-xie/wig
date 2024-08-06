

test_that("randomized SVD", {
  M <- rbind(c(20, 10, 15, 0, 2),
             c(10, 5, 8, 1, 0),
             c(0, 1, 2, 6, 3),
             c(1, 0, 0, 10, 5))
  rsvd_scikit <- c(26.97223175, 13.73834416, 1.98003873, 1.5472032)
  rsvd_wig <- c(wig::tsvd(M, 1))
  
  expect_equal(rsvd_scikit, rsvd_wig, tolerance = 1e-8)
})
