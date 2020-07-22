varScale<- c(-10, 1, 10)
names(varScale)<- paste0("scale", varScale)
tempdirRaster<- tempdir()
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
predInputR<- raster::raster(nrows=10, ncols=5)

## Input maps with patterns
predInputR.r0<- raster::setValues(predInputR, runif(raster::ncell(predInputR)))
predInputR.r<- raster::stack(lapply(varScale, function(i){
  predInputR.r0 * i
}))
# raster::plot(raster::stack(predInputR.r0, predInputR.r))

# Put some NAs to detect rotations
NAs<- expand.grid(col=1:ncol(predInputR), row=1:nrow(predInputR))
NAs<- NAs[NAs$row < NAs$col, ]
predInputR.r[NAs$row, NAs$col]<- NA


predInputR.seq<- raster::stack(lapply(varScale, function(i){
  raster::setValues(predInputR, 1:raster::ncell(predInputR) * i)
}))


i<- 1:10
j<- 1:5
v<- tcrossprod(i, j) # i %*% t(j)
predInputR.crossij<- raster::raster(predInputR)
predInputR.crossij[]<- v
predInputR.prodij<- raster::stack(
  lapply(varScale, function(i){
       predInputR.crossij * i
  })
)
# plot(stack(predInputR.crossij, predInputR.prodij))

predInputR<- raster::stack(predInputR.r, predInputR.seq, predInputR.prodij)
names(predInputR)<- paste0(rep(c("r", "seq", "prodij"), each=length(varScale)), "_", names(varScale))

# plot(predInputR)


## Identity model
model<- list(output_shape=list(NA, raster::nlayers(predInputR)))
class(model)<- "x1"

predict.x1<- function(object, v, ...) v


test_that("predict.Raster_keras with identity model produce identical maps", {
  predOutputR<- predict.Raster_keras(predInputR, model, fun=predict.x1, filename=filenameRasterPred, overwrite=TRUE)
  names(predOutputR)<- names(predInputR)

  expect_equal(predInputR[], predOutputR[], tolerance=1e-4)
  expect_equivalent(predInputR, predOutputR)


  predOutputR<- predict.Raster_keras(predInputR, model, fun=predict.x1, )
  names(predOutputR)<- names(predInputR)

  expect_equal(predInputR[], predOutputR[], tolerance=1e-4)
  expect_equivalent(predInputR, predOutputR)

  # raster::plot(predInputR)
  # raster::plot(predOutputR)

  # raster::compareRaster(predInputR, predOutputR, values=TRUE)
  # raster::plot(predInputR == predOutputR)
  # raster::plot(predInputR - predOutputR)
  # raster::summary(predInputR == predOutputR)
  # (predInputR == predOutputR)[]

})
