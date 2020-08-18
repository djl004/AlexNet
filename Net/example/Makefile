ifndef ARCHLAB_ROOT
$(error ARCHLAB_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif
ifndef PIN_ROOT
$(error PIN_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif
ifndef CANELA_ROOT
$(error CANELA_ROOT is not set.  Have you sourced config.sh?  Maybe do it again.)
endif

OPTIMIZE=-O3 $(EXTRA) #-ftree-vectorize -fopt-info-vec-all=vec.log #-march=native -mtune=native 

default:stabilize.exe

OPENMP=yes
include $(ARCHLAB_ROOT)/cse141.make




