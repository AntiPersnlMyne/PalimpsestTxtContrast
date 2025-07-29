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
  img = raster.getData()
  dims = size(img, /dimensions)
  n_bands = dims[0]

  ; Normalize pixel vectors
  norms = sqrt(total(img ^ 2, 1))
  normalized = img / replicate(1.0, n_bands) # norms

  ; -------------------------
  ; Step 1: Select initial target with max norm
  ; -------------------------
  idx_max = max(norms, /nan)
  t0 = normalized[*, idx_max]

  ; Initialize target list
  target_list = [t0]
  done = 0
  k = 1 ; Target counter

  while ~done and (k lt max_targets) do begin
    ; -------------------------
    ; Step 2: Project all pixels onto orthogonal subspace of previous targets
    ; -------------------------
    T_mat = transpose(reform(target_list, n_bands, k)) ; [n_bands, k]
    pinv_T = PINV(T_mat)
    PU = identity(n_bands) - T_mat ## pinv_T

    ; -------------------------
    ; Step 3: Apply PU to all pixel vectors and find max projection magnitude
    ; -------------------------
    projected = PU ## img
    mags = total(projected ^ 2, 1)

    idx_next = max(mags, /nan)
    t_next = normalized[*, idx_next]

    ; -------------------------
    ; Step 4: Compute OPCI between new target and current subspace
    ; OPCI = || (I - PU) * t_next ||^2
    ; -------------------------
    residual = (identity(n_bands) - PU) ## t_next
    opci = total(residual ^ 2)

    ; -------------------------
    ; Check stopping condition
    ; -------------------------
    if opci lt opci_thresh then begin
      done = 1
    endif else begin
      target_list = [target_list, t_next]
      k += 1
    endelse
  endwhile

  ; -------------------------
  ; Return target matrix
  ; -------------------------
  targets = reform(target_list, n_bands, k)
  RETURN, targets
end
