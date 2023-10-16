get_shap <- function(model, test_data, bg_data, bg_weight=NULL, verbose=TRUE) {
  shap <- kernelshap::kernelshap(object = model, X = test_data, bg_X = bg_data, bg_w = bg_weight, verbose = verbose)
  sv <- shapviz::shapviz(shap)

  return(sv)
}
