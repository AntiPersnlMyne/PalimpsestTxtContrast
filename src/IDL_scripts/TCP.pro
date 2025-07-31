;+
; :Arguments:
;   raster: bidirectional, required, any
;     Placeholder docs for argument, keyword, or property
;   target_matrix: bidirectional, required, any
;     Placeholder docs for argument, keyword, or property
;
; :Keywords:
;   class_images: bidirectional, optional, any
;     Placeholder docs for argument, keyword, or property
;
; :Project:
;   MISHA RIT
;
; :Copyright:
;   MIT
;
; :Date:
;   July 29 2025
;
;    PURPOSE:
;      Implements the Target Classification Process (TCP) from the ATDCA
;      using Orthogonal Subspace Projection (OSP) per target.
;
;    CALLING SEQUENCE:
;      TCP, raster, target_matrix, CLASS_IMAGES=score_cube
;
;    INPUTS:
;      raster        - ENVI raster object with expanded bands
;      target_matrix - [n_bands, n_targets] matrix of target vectors (from TGP)
;
;    KEYWORDS:
;      CLASS_IMAGES - Hold the classification score results for all targets: [rows, cols, n_targets]
;                     For each target t_k, the TCP procedure computes a score image of size [rows, cols]
;                     indicating how well each pixel matches that target.
;
; :Author:
;   Gian-Mateo (GM) Tifone
;
;-
pro TCP, raster, target_matrix, class_images = score_cube
  compile_opt idl2

  ; -------------------------------------------------------------------
  ; Input preparation
  ; -------------------------------------------------------------------
  img = raster.getData()
  dims = size(img, /dimensions)
  n_bands = dims[2]

  dims = size(target_matrix, /dimensions)
  n_targets = dims[1]

  ; Image shape for reshaping output
  rows = raster.nRows
  cols = raster.nColumns

  ; Allocate output cube
  score_cube = fltarr(rows, cols, n_targets)

  ; Loop through targets
  for k = 0, n_targets - 1 do begin
    ; -----------------------------------------
    ; Construct undesired targets' matrix U_k
    ; -----------------------------------------
    undesired = target_matrix[*, where(indgen(n_targets) ne k)]

    ; Ensure 2D form even for 1 target left
    if size(undesired, /n_dimensions) eq 1 then $
      undesired = reform(undesired, n_bands, 1)

    ; -----------------------------------------
    ; Compute orthogonal projector P_U_k
    ; -----------------------------------------
    ; Compute Moore-Penrose Pseudoinverse
    pinv_U = invert(matrix_multiply(undesired, undesired, /atranspose)) ; (AT * A)^-1
    pinv_U = matrix_multiply(pinv_U, undesired, /btranspose) ; (AT * A)^-1 * AT

    PU = identity(n_bands) - matrix_multiply(pinv_U, undesired)

    ; -----------------------------------------
    ; Apply projection to all pixels
    ; y_k = P_Uk * x
    ; score = |y_k|^2
    ; -----------------------------------------
    ; Convert image to float and reshape to [bands, pixels]
    img_data = reform(float(img), n_bands, rows * cols)

    ; Project pixels into orthogonal subspace
    projected = matrix_multiply(PU, img_data)
    scores = total(projected ^ 2, 1)

    ; -----------------------------------------
    ; Reshape score vector into image
    ; -----------------------------------------
    score_img = reform(scores, cols, rows) ; ENVI uses column-major
    score_cube[*, *, k] = transpose(score_img)
  endfor
end
