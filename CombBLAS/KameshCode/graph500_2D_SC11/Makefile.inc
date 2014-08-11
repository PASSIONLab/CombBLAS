all-targets: all-recursive all-local

ifeq ($(ARCH),)
  ARCH := generic
endif

include $(top_srcdir)/ARCH/Makefile.$(ARCH)

CFLAGS_STD = -I. -I$(top_srcdir)

%.o: %.c 
	$(CC) $(CFLAGS_STD) $(CDEPFLAGS) $(CFLAGS) $(CFLAGSOMP) $(COPTFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CFLAGS_STD) $(CDEPFLAGS) $(CXXFLAGS) $(CFLAGSOMP) $(COPTFLAGS) -o $@ -c $<


all-local: all

all-recursive:
	@echo "  "
	@echo "Starting build process ..."
	@echo "ARCH set to $(ARCH)."
	@echo "  "
	@$(MAKE) recursive RECTGT=all

clean: clean-local clean-recursive

clean-recursive:
	@$(MAKE) recursive RECTGT=clean

clean-local:
	rm -rf core *~ *.d $(TARGETS) $(CLEANFILES)

recursive:
	@test -n "$(RECTGT)"
	@for subdir in $(SUBDIRS) ; do \
	  echo "=== $(RECTGT): Entering subdirectory, `pwd`/$$subdir ... ===" ; \
	  cd $$subdir || exit 1 ; \
	  $(MAKE) $(RECTGT) || exit 1 ; \
	  cd - ; \
	done
