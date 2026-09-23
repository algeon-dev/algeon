# Resolve cuML only from its canonical installation prefix.
set(cuml_ROOT "$ENV{CUML_INSTALL_DIR}" CACHE PATH "Canonical cuML install prefix")
set(CUDF_INSTALL_DIR "$ENV{CUDF_INSTALL_DIR}" CACHE PATH "Canonical cuDF install prefix")
