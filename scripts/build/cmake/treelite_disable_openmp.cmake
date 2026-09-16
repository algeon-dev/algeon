# Injected after project(treelite) through CMAKE_PROJECT_treelite_INCLUDE.
# cuML's get_treelite.cmake passes USE_OPENMP ON as a CPM option, which CPM
# sets as a normal variable in the calling scope, so a -DUSE_OPENMP=OFF cache
# entry is ignored. A normal variable in Treelite's own scope wins because its
# option(USE_OPENMP) follows project() under CMP0077 NEW.
set(USE_OPENMP OFF)
