context("preprocess_timeseries")

dl<- data.frame(id_1=rep(c("A", "B"), each=2), id_2=rep(c("A", "B"), 2), static=rep(c(-1, -2), each=2), time=rep(1L, 4), x=1:4, y=10:13)
dl<- rbind(dl, dl)
dl[5:8, c("time", "x", "y")]<- list(10, dl$x[5:8] + 1, dl$y[5:8] + 1)
dw<- as.data.frame(data.table::dcast(data.table::as.data.table(dl), id_1 + id_2 + static ~ time, value.var=c("x", "y")))
id.vars<- c("id_1", "id_2", "static")
vars<- setdiff(names(dl), c(id.vars, "time"))
vars.ts<- setdiff(names(dw), c(id.vars, "time"))


test_that("preprocess time series works with data.frame", {
  dl_w<- longToWide.ts(d=dl, timevar="time" , idCols=id.vars)
  dw_l<- wideToLong.ts(d=dw, timevar="time", vars=vars, idCols=id.vars)

  expect_equal(dw, dl_w)
  expect_equal(dl, dw_l)
  expect_s3_class(dl_w, "data.frame", exact=TRUE)
  expect_s3_class(dw_l, "data.frame", exact=TRUE)

  al<- longTo3Darray.ts(d=dl, timevar="time", idCols=id.vars)
  aw<- wideTo3Darray.ts(d=dw, vars=vars, idCols=id.vars)

  expect_equal(al, aw)
  expect_equal(dim(al), c(4, 2, 2))
  expect_equal(names(dimnames(al)), c("case", "t", "var"))
})


test_that("preprocess time series works with data.table", {
  dl<- data.table::as.data.table(dl)
  dw<- data.table::as.data.table(dw)
  dl_w<- longToWide.ts(d=dl, timevar="time", idCols=id.vars)
  dw_l<- wideToLong.ts(d=dw, timevar="time", vars=vars, idCols=id.vars)

  expect_equivalent(dw, dl_w) # Datasets have different keys. 'target': has no key. 'current': [id_1, id_2, static].
  expect_equal(dl, dw_l)
  expect_s3_class(dl_w, "data.table", exact=FALSE)
  expect_s3_class(dw_l, "data.table", exact=FALSE)

  al<- longTo3Darray.ts(d=dl, timevar="time", idCols=id.vars)
  aw<- wideTo3Darray.ts(d=dw, vars=vars, idCols=id.vars)

  expect_equal(al, aw)
  expect_equal(dim(al), c(4, 2, 2))
  expect_equal(names(dimnames(al)), c("case", "t", "var"))
})


test_that("preprocess time series works with matrix", {
  dl<- as.matrix(dl)
  dw<- as.matrix(dw)
  dl_w<- longToWide.ts(d=dl, timevar="time" , idCols=id.vars)
  dw_l<- wideToLong.ts(dw, timevar="time", vars=vars, idCols=id.vars)

  expect_equivalent(dw, dl_w) # Datasets have different keys. 'target': has no key. 'current': [id_1, id_2, static].
  expect_equal(dl, dw_l)
  expect_equal(dim(dl_w), c(4, 7))
  expect_equal(dim(dw_l), c(8, 6))

  al<- longTo3Darray.ts(d=dl, timevar="time", idCols=id.vars)
  aw<- wideTo3Darray.ts(d=dw, vars=vars, idCols=id.vars)
  dimnames(al)$t<- gsub(" ", "", dimnames(al)$t)

  expect_equal(al, aw)
  expect_equal(dim(al), c(4, 2, 2))
  expect_equal(names(dimnames(al)), c("case", "t", "var"))
})
