PRO open_file
    COMPILE_OPT IDL3
    e = ENVI(/HEADLESS)
    file = FILEPATH('qb_boulder_msi', ROOT_DIR=e.ROOT_DIR, SUBDIRECTORY=['data'])
    PRINT, 'Found file at: ', file
    MESSAGE, '', /INFORMATIONAL
END

