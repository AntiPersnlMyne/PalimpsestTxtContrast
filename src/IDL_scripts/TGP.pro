;+
; :Author: Gian-Mateo (GM) Tifone
; :Project: MISHA RIT
; :Copyright: MIT
; :Date: July 29 2025
;
; PURPOSE:
;   Implements the Target Generation Process (TGP) for ATDCA,
;   which detects potential target vectors in an unsupervised fashion.
;
; CALLING SEQUENCE:
;   targets = TGP(raster, OPCI_THRESHOLD=opci_thresh, MAX_TARGETS=max_targets)
;
; INPUTS:
;   raster - ENVI raster object (expanded bands via BGP)
;
; KEYWORDS:
;   OPCI_THRESHOLD - (optional) Orthogonal Projection Correlation Index, stopping threshold for residual projection.
;                               Higher (e.g. 0.1)   -> greater quantity, lower purity targets. May add reduntant classes.
;                               Lower (e.g. 0.001) -> fewer quantity, higher purity targets.
;                               [default=0.01]
;
;   MAX_TARGETS    - (optional) maximum number of targets to generate. [default=10]
;
; RETURNS:
;   Array of target vectors: [n_bands, n_targets]
;-

function TGP, raster, opci_threshold = opci_thresh, max_targets = max_targets
  compile_opt idl2

  ; -------------------------
  ; Parameter defaults
  ; -------------------------
  if n_elements(opci_thresh) eq 0 then opci_thresh = 0.01
  if n_elements(max_targets) eq 0 then max_targets = 10

  ; -------------------------
  ; Prepare image data
  ; -------------------------
  img_data = raster.getData()
  dims = size(img_data, /dimensions)
  samples = dims[0]
  lines = dims[1]
  n_bands = dims[2]
  n_pixels = samples * lines

  img_data = transpose(img_data, [2, 0, 1]) ; [n_bands, samples, lines]
  img_data = reform(img_data, n_bands, n_pixels) ; [n_bands, n_pixels]

  ; Convert to float to avoid integer overflow
  img_data = float(img_data)

  ; Now look, I'm ASSUMING the Chang & Ren paper wants you to normalize...
  norms = sqrt(total(img_data ^ 2, 1)) ; [n_pixels]
  norms = norms > 1e-10
  norms2d = replicate(1.0, n_bands, 1) # norms ; broadcast to [n_bands, n_pixels]
  normalized = img_data / norms2d ; [n_bands, n_pixels]

  ; -------------------------------------------------------------------------
  ; Select initial target with "most extreme" attribute (i.e. magnitude)
  ; -------------------------------------------------------------------------
  ; Index of the max value, choose chosen as first target
  _ = max(norms, idx_max, /nan)
  T0 = normalized[*, idx_max]
  target_list = reform(T0, n_bands, 1)
  target_num = 1
  done = 0

  ; Mathematical proof read Cheng and Ren, IEEE, 2000
  while ~done and (target_num lt max_targets) do begin
    ; -----------------------------------------------------------------------
    ; Project all pixels onto orthogonal subspace of previous targets
    ; -----------------------------------------------------------------------
    T_mat = transpose(reform(target_list, n_bands, target_num)) ; [n_bands, k]

    ; Compute Moore-Penrose Pseudoinverse - bcs ENVI doesn't have that builtin??
    pinv_T = invert(matrix_multiply(T_mat, T_mat, /atranspose)) ; (AT * A)^-1
    pinv_T = matrix_multiply(pinv_T, T_mat, /btranspose) ; (AT * A)^-1 * AT
    PU = identity(n_bands) - matrix_multiply(pinv_T, T_mat)

    ; -----------------------------------------------------------------------
    ; Apply P_U to all pixel vectors and find max projection magnitude
    ; -----------------------------------------------------------------------
    projected = matrix_multiply(PU, img_data)
    mags = total(projected ^ 2, 1)

    ; Fix: get index of max value in mags
    _ = max(mags, idx_next, /nan)
    t_next = normalized[*, idx_next]

    ; -----------------------------------------------------------------------
    ; Compute OPCI between new target and current subspace
    ; OPCI = || (I - PU) * t_next ||^2
    ; -----------------------------------------------------------------------
    residual = matrix_multiply(t_next, identity(n_bands) - PU)
    opci = total(residual ^ 2)

    ; -----------------------------------------------------------------------
    ; Check stopping condition
    ; -----------------------------------------------------------------------
    if opci lt opci_thresh then begin
      done = 1
    endif else begin
      target_list = [[target_list], [reform(t_next, n_bands, 1)]]
      target_num += 1
    endelse
  endwhile

  ; -------------------------
  ; Return target matrix
  ; -------------------------
  RETURN, target_list
end
