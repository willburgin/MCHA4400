# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-src")
  file(MAKE_DIRECTORY "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-src")
endif()
file(MAKE_DIRECTORY
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-build"
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix"
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/tmp"
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src"
  "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/willburgin/MCHA4400/lab2/build/_deps/nanobench-subbuild/nanobench-populate-prefix/src/nanobench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
