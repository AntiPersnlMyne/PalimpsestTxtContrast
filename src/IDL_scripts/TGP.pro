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
  n_bands = dims[0]

  ; Now look, I'm ASSUMING the Chang,Ren paper wants you to normalize the results
  ; I'm doing it because it feels like the right thing to do, but no better reason
  ; Also the normalizing function is in the ML library, and I'm not downloading it
  norms = sqrt(total(img_data ^ 2, 1))
  normalized = img_data / replicate(1.0, n_bands) # norms

  ; -------------------------
  ; Select initial target with "most extreme" attribute (i.e. magnitude)
  ; -------------------------
  idx_max = max(norms, /nan)
  T0 = normalized[*, idx_max]

  ; Initialize target list
  target_list = [T0]
  done = 0
  k = 1 ; Target counter

  ; Author of everything in the while loop = ChatGPT (OpenAI)
  ; I couldn't translate the math notation on my own
  while ~done and (k lt max_targets) do begin
    ; -----------------------------------------------------------------------
    ; Project all pixels onto orthogonal subspace of previous targets
    ; -----------------------------------------------------------------------
    T_mat = transpose(reform(target_list, n_bands, k)) ; [n_bands, k]
    pinv_T = PINV(T_mat)
    PU = identity(n_bands) - T_mat ## pinv_T

    ; -----------------------------------------------------------------------
    ; Apply P_U to all pixel vectors and find max projection magnitude
    ; -----------------------------------------------------------------------
    projected = PU ## img_data
    mags = total(projected ^ 2, 1)

    idx_next = max(mags, /nan)
    t_next = normalized[*, idx_next]

    ; -----------------------------------------------------------------------
    ; Compute OPCI between new target and current subspace
    ; OPCI = || (I - PU) * t_next ||^2
    ; -----------------------------------------------------------------------
    residual = (identity(n_bands) - PU) ## t_next
    opci = total(residual ^ 2)

    ; -----------------------------------------------------------------------
    ; Check stopping condition
    ; -----------------------------------------------------------------------
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
