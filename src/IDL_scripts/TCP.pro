;+
; :Author: Gian-Mateo (GM) Tifone
; :Project: MISHA RIT
; :Copyright: MIT
; :Date: July 29 2025
;
; PURPOSE:
;   Implements the Target Classification Process (TCP) from the ATDCA
;   using Orthogonal Subspace Projection (OSP) per target.
;
; CALLING SEQUENCE:
;   TCP, raster, target_matrix, CLASS_IMAGES=score_cube
;
; INPUTS:
;   raster        - ENVI raster object with expanded bands
;   target_matrix - [n_bands, n_targets] matrix of target vectors (from TGP)
;
; KEYWORDS:
;   CLASS_IMAGES - Hold the classification score results for all targets: [rows, cols, n_targets]
;                  For each target t_k, the TCP procedure computes a score image of size [rows, cols]
;                  indicating how well each pixel matches that target.
;
;-

pro TCP, raster, target_matrix, class_images = score_cube
  compile_opt idl2

  ; -------------------------------------------------------------------
  ; Input preparation
  ; -------------------------------------------------------------------
  img = raster.getData()
  dims = size(img, /dimensions)
  n_bands = dims[0]
  ; n_pixels = dims[1]
  dims = size(target_matrix, /dimensions)
  n_targets = dims[1]

  ; Image shape for reshaping output
  rows = raster.nRows
  cols = raster.nCols

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
    pinv_U = PINV(undesired)
    PU = identity(n_bands) - undesired ## pinv_U

    ; -----------------------------------------
    ; Apply projection to all pixels
    ; y_k = P_Uk * x
    ; score = |y_k|^2
    ; -----------------------------------------
    projected = PU ## img
    scores = total(projected ^ 2, 1)

    ; -----------------------------------------
    ; Reshape score vector into image
    ; -----------------------------------------
    score_img = reform(scores, cols, rows) ; ENVI uses column-major
    score_cube[*, *, k] = transpose(score_img)
  endfor
end
