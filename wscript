APPNAME = 'arpaca'
VERSION = '0.1.0'

top = '.'
out = 'build'

def options(opt):
  opt.load('compiler_cxx unittest_gtest')

def configure(cnf):
  cnf.check_tool('compiler_cxx unittest_gtest')
  cnf.env.append_unique('CXXFLAGS', ['-g', '-W', '-Wall', '-O3'])

  cnf.check_cxx(lib = 'arpack')
  cnf.check_cxx(lib = 'lapack', mandatory = False)
  cnf.check_cxx(lib = 'f77blas', mandatory = False)
  cnf.check_cxx(lib = 'gfortran', mandatory = False)
  cnf.check_cxx(lib = 'atlas', mandatory = False)

  # for test and performance
  cnf.check_cfg(package = 'pficommon', args = '--cflags --libs')

def build(bld):
  bld.program(target = 'arpaca_test',
              features = 'gtest',
              source = 'arpaca_test.cpp',
              uselib = 'ARPACK LAPACK F77BLAS ATLAS GFORTRAN')

  bld.program(target = 'arpaca_performance_test',
              source = 'performance_main.cpp',
              uselib = 'PFICOMMON ARPACK LAPACK F77BLAS ATLAS GFORTRAN',
              install_path = None)

  bld.install_files('${PREFIX}/include', 'arpaca.hpp')
