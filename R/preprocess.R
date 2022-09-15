
#' Time series format transformations
#'
#' For data in wide format, the name of time varying columns follow the pattern \code{var_time}
#'
#' @param d a data.frame or matrix.
#' @param timevar column name for the time.
#' @param vars time varying variables in the long format (ie. without the _time part in wide format).
#' @param idCols ids or no time varying columns. Works with column names and indexes.
#' @param regex_time a regular expression for the time part of the variable names for data in wide format.
#'
#' @return the same data in a wide [var_time columns], long [time + vars columns] or a 3D array [samples, time, vars] format.
#' @name transformTS
# @importFrom data.table .N .I ':=' .SD
# @examples
NULL

## TODO: rename idCols -> idVars / static vars ----

#' @rdname transformTS
#' @export
longToWide.ts<- function(d, timevar, idCols=NULL){
  fun.aggregate<- mean
  if (inherits(d, "matrix")){
    fun.aggregate<- function(x){
      if (length(x) == 1) return(x)
      else {
        return(mean(as.numeric(x), na.rm=TRUE))
      }
    }
  }

  if (!inherits(d, "data.table")){
    classOri<- class(d)
    d<- data.table::as.data.table(d)
  } else {
    classOri<- "data.table"
  }
  if (is.numeric(timevar)){
    timevar<- colnames(d)[timevar]
  }
  if (is.numeric(idCols)){
    idCols<- colnames(d)[idCols]
  }

  vars<- setdiff(colnames(d), c(idCols, timevar))
  LHS<- setdiff(idCols, timevar)
  if (is.null(LHS) | length(LHS) == 0){
    LHS<- "."
  }
  form<- paste0("`", paste(LHS, collapse="` + `"), "` ~ `", timevar, "`")
  d<- data.table::dcast(d, formula=stats::formula(form), fun.aggregate=fun.aggregate, value.var=vars)  # To wide format (var_time columns)
  if (!"data.table" %in% classOri & "data.frame" %in% classOri){
    d<- as.data.frame(d)
  } else if ("matrix" %in% classOri){
    d<- as.matrix(d)
  }

  return(d)
}


#' @rdname transformTS
#' @export
wideToLong.ts<- function(d, timevar, vars, idCols=NULL, regex_time=".+"){
  if (!inherits(d, "data.table")){
    classOri<- class(d)
    d<- data.table::as.data.table(d)
  } else {
    classOri<- "data.table"
  }
  if (is.numeric(timevar)){
    timevar<- colnames(d)[timevar]
  }
  if (is.numeric(idCols)){
    idCols<- colnames(d)[idCols]
  }

  if (missing(vars)){  # try to guess
    vars<- unique(gsub("^(.+)_.+$", "\\1", setdiff(colnames(d), idCols)))
  }

  idCols<- setdiff(idCols, timevar)
  pattern<- pattern<- paste0("^(", paste(vars, collapse="|"), ")_(", regex_time, ")$")
  cmd<- paste0("data.table::melt(d, id.vars=idCols, variable.factor=FALSE, ",
                           "measure.vars=data.table:::measure(value.name, ", timevar, ", pattern=pattern))")
  d<- eval(parse(text=cmd))

  rmRows<- apply(data.table::`[.data.table`(x=d, , j=vars, with=FALSE, drop=FALSE), 1, function(x) all(is.na(x))) # missing timesteps
  d<- d[!rmRows, ]

  ## WARNING: timevar in results is character instead of the original, columns in different order (timevar between idCols and vars)
  # Cast timevar to numeric if no info is lost
  timevals<- tryCatch(as.integer(d[[timevar]]), warning=function(w) d[[timevar]])
  if (is.character(timevals)){
    timevals<- tryCatch(as.numeric(d[[timevar]]), warning=function(w) d[[timevar]])
  }
  if (is.numeric(timevals)){
    d[[timevar]]<- timevals
  }

  if (!"data.table" %in% classOri & "data.frame" %in% classOri){
    d<- as.data.frame(d)
  } else if ("matrix" %in% classOri){
    d<- as.matrix(d)
  }

  return(d)
}

#' @rdname transformTS
#' @export
wideTo3Darray.ts<- function(d, vars, idCols=NULL){
  d<- as.data.frame(d)
  if (is.numeric(vars)){
    vars<- colnames(d)[vars]
  }
  if (is.numeric(idCols)){
    idCols<- colnames(d)[idCols]
  }
  if (missing(vars)){  # try to guess
    vars<- unique(gsub("^(.+)_.+$", "\\1", setdiff(colnames(d), idCols)))
  }

  batch_size<- 500 # Error in gsub for lenght(vars) >920 (perhaps nchar is more relevant?): assertion 'tree->num_tags == num_tags' failed in executing regexp: file 'tre-compile.c', line 634
  suppressWarnings(batch<- split(vars, rep(1:(length(vars) %/% batch_size + 1), each=batch_size)))
  timevals<- setdiff(colnames(d), idCols)
  for (i in seq_along(batch)){
    regexVars<- paste(batch[[i]], collapse="|")
    timevals<- unique(gsub(paste0("^(", regexVars, ")_"), "", timevals))
  }
  # timevals<- unique(gsub(paste0("^(", paste(vars, collapse="|"), ")_"), "", setdiff(colnames(d), idCols)))

  # Reshape to a 3D array [samples, timesteps, features] Format for LSTM layers in NN
  a<- lapply(vars, function(x){
    varTS<- d[, grep(paste0("^", x, "_(", paste(timevals, collapse="|"), ")$"), colnames(d))]
    a<- array(as.matrix(varTS), dim=c(nrow(varTS), ncol(varTS), 1), dimnames=list(case=NULL, t=gsub(paste0("^", x, "_"), "", colnames(varTS)), var=x))
  })
  names(a)<- vars
  a<- abind::abind(a)
  names(dimnames(a))<- c("case", "t", "var")
  if (!is.null(idCols)){
    dimnames(a)$case<- do.call(paste, c(d[, idCols, drop=FALSE], list(sep="_")))
  }

  return(a)
}


#' @rdname transformTS
#' @export
longTo3Darray.ts<- function(d, timevar, idCols=NULL){
  if (is.numeric(timevar)){
    timevar<- colnames(d)[timevar]
  }
  if (is.numeric(idCols)){
    idCols<- colnames(d)[idCols]
  }
  vars<- setdiff(colnames(d), c(idCols, timevar))
  d<- longToWide.ts(d=d, timevar=timevar, idCols=idCols)

  wideTo3Darray.ts(d=d, vars=vars, idCols=idCols)
}
