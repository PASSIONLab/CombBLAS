import sys, os, platform, re
from distutils import sysconfig
from distutils.util  import convert_path
from distutils.util  import split_quoted
from distutils.spawn import find_executable
from distutils import log


from distutils.command import config  as cmd_config
from distutils.command import build   as cmd_build
from distutils.command import install as cmd_install
from distutils.command import sdist   as cmd_sdist
from distutils.command import clean   as cmd_clean

from distutils.command import build_ext    as cmd_build_ext


from mpiconfig import Config

from mpidistutils import cmd_mpi_opts, cmd_initialize_mpi_options, configuration, configure_compiler

class build_ext(cmd_build_ext.build_ext):

    user_options = cmd_build_ext.build_ext.user_options + cmd_mpi_opts

    def initialize_options(self):
        cmd_build_ext.build_ext.initialize_options(self)
        cmd_initialize_mpi_options(self)


    def build_extensions(self):
        # parse configuration file and configure compiler
        config = configuration(self, verbose=True)
        configure_compiler(self.compiler, config)
        if self.compiler.compiler_type == "unix":
            so_ext = sysconfig.get_config_var('SO')
            self.compiler.shared_lib_extension = so_ext
        self.config = config # XXX
        # extra configuration, check for all MPI symbols
#        if self.configure:

        # build extensions
        for ext in self.extensions:
            try:
                self.build_extension(ext)
            except (DistutilsError, CCompilerError):
                if not ext.optional: raise
                e = sys.exc_info()[1]
                self.warn('building extension "%s" failed' % ext.name)
                self.warn('%s' % e)

    def config_extension (self, ext):
        configure = getattr(ext, 'configure', None)
        if configure:
            config_cmd = self.get_finalized_command('config')
            config_cmd.compiler = self.compiler # fix compiler
            configure(ext, config_cmd)


