# host/c/Makefile: libphi512-c.so, the AVX-512 intercept preload in C,
# linked with the assembly modules not yet ported (README.md, "The hybrid
# build"). Everything builds on the rack (gcc, GNU as).
#
#   make                    out/libphi512-c.so (C_MODULES in C, the rest assembly)
#   make asm-only           out/libphi512-c-asm.so: the same link, every module assembly (the A/B reference)
#   make check              the library, then: no AVX-512 instruction in the C objects, the imports
#                           against imports.want, clang-format, clang-tidy
#   make compdb             ../../compile_commands.json for clangd and clang-tidy
#   make format             clang-format the C sources in place
#   make clean

# Relative paths: make cannot take the spaces in the checkout's name.
ROOT := ../..
ASMDIR := ../asm/phi512
COMMON := ../asm/common
PROTO := ../../card/vpu
OUT := out

# The modules in C. Every other module of ASM_ALL is assembled from
# host/asm/phi512; a port moves a name from the second list to the first.
C_MODULES ?= main
ASM_ALL := main sigill card elf seeds decode fnscan mvex xlate blob xlfn region pool place
COMMON_ALL := text env lock mem table window fp
ASM_MODULES := $(filter-out $(C_MODULES),$(ASM_ALL))
LIB ?= libphi512-c.so

CC := gcc
AS := as
# Never AVX-512 in the preload itself (the host has none); gcc-only flags
# are kept apart so compile_commands.json suits clang.
CFLAGS_COMMON := -std=gnu17 -O2 -g -fPIC -fvisibility=hidden -fno-plt -march=x86-64-v2 -mno-avx512f \
	-Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wformat=2 -Werror \
	-Iphi512 -I$(PROTO)
CFLAGS_GCC := -mincoming-stack-boundary=3
CFLAGS := $(CFLAGS_COMMON) $(CFLAGS_GCC)
ASFLAGS := --64 -I $(ASMDIR) -I $(COMMON) -I $(PROTO)
LDFLAGS := -shared -Wl,-soname,$(LIB) -Wl,-z,now -Wl,-z,relro -Wl,-z,noexecstack -Wl,-z,defs \
	-Wl,-z,initfirst -Wl,-Bsymbolic -Wl,--version-script=phi512.map

C_SRCS := $(foreach m,$(C_MODULES),phi512/$(m).c)
C_OBJS := $(foreach m,$(C_MODULES),$(OUT)/c_$(m).o)
ASM_OBJS := $(foreach m,$(ASM_MODULES),$(OUT)/asm_$(m).o) $(foreach m,$(COMMON_ALL),$(OUT)/common_$(m).o)
ASM_INCS := $(wildcard $(ASMDIR)/*.inc $(ASMDIR)/*.h $(COMMON)/*.inc $(PROTO)/*.inc)

.PHONY: all asm-only check noavx512 imports format-check tidy compdb format clean
all: $(OUT)/$(LIB)

$(OUT):
	mkdir -p $@

$(OUT)/c_%.o: phi512/%.c | $(OUT)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(OUT)/asm_%.o: $(ASMDIR)/%.S $(ASM_INCS) | $(OUT)
	$(AS) $(ASFLAGS) -o $@ $<

$(OUT)/common_%.o: $(COMMON)/%.S $(ASM_INCS) | $(OUT)
	$(AS) $(ASFLAGS) -o $@ $<

$(OUT)/$(LIB): $(C_OBJS) $(ASM_OBJS) phi512.map
	$(CC) $(LDFLAGS) -o $@.new $(C_OBJS) $(ASM_OBJS)
	mv $@.new $@
	@echo "$(LIB): C [$(C_MODULES)] + assembly [$(ASM_MODULES)], $$(wc -c < $@) bytes"

asm-only:
	$(MAKE) C_MODULES= LIB=libphi512-c-asm.so

# No AVX-512 instruction compiled from C (the assembly modules carry MVEX
# and EVEX bytes as data for the card, so only the C objects are read).
noavx512: $(C_OBJS)
	@n=$$(objdump -d --no-show-raw-insn $(C_OBJS) | grep -c -E '%zmm|%k[0-7]\b|\{1to'); \
	if [ "$$n" -ne 0 ]; then objdump -d --no-show-raw-insn $(C_OBJS) | grep -E '%zmm|%k[0-7]\b|\{1to' | head; \
		echo "noavx512: $$n AVX-512 instruction(s) in the C objects"; exit 1; fi; \
	echo "noavx512: ok ($(words $(C_OBJS)) C object(s))"

# The library's undefined dynamic symbols against imports.want: a new
# import is a decision, recorded by updating the list.
imports: $(OUT)/$(LIB)
	@nm -D --undefined-only $(OUT)/$(LIB) | awk '{print $$NF}' | sed 's/@.*//' | LC_ALL=C sort -u > $(OUT)/imports.have
	@if diff -u imports.want $(OUT)/imports.have; then echo "imports: ok ($$(wc -l < $(OUT)/imports.have) symbols)"; \
	else echo "imports: the library's imports differ from imports.want"; exit 1; fi

format-check:
	@clang-format --dry-run -Werror $(wildcard phi512/*.c phi512/*.h) && echo "format: ok"

tidy: compdb
	@clang-tidy -p $(ROOT) --quiet --warnings-as-errors='*' $(C_SRCS) 2>&1 | grep -v -E '^[0-9]+ warnings? generated' ; \
	test $${PIPESTATUS[0]} -eq 0 && echo "tidy: ok"
tidy: SHELL := /bin/bash

check: all noavx512 imports format-check tidy

# compile_commands.json for clangd and clang-tidy, the gcc-only flags left out.
compdb:
	@{ echo '['; sep=''; for m in $(C_MODULES); do \
		printf '%s{"directory": "%s", "file": "%s", "arguments": ["clang", %s, "-c", "%s", "-o", "%s"]}\n' "$$sep" \
		"$(CURDIR)" "$(CURDIR)/phi512/$$m.c" \
		"$$(printf '"%s", ' $(CFLAGS_COMMON) | sed 's/, $$//')" "$(CURDIR)/phi512/$$m.c" "$(CURDIR)/$(OUT)/c_$$m.o"; sep=','; \
	done; echo ']'; } > $(ROOT)/compile_commands.json
	@echo "compdb: $(ROOT)/compile_commands.json ($(words $(C_MODULES)) file(s))"

format:
	clang-format -i $(wildcard phi512/*.c phi512/*.h)

clean:
	rm -rf $(OUT)

-include $(C_OBJS:.o=.d)
