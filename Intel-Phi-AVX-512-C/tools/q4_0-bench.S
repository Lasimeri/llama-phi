# q4_0-bench.S: cycles per call of the Q4_0 one-row kernels and of their parts,
# one thread, every operand in the first-level cache (the stride argument
# is 0, so the kernels' prefetches hit the lines they are working on).
# Static, no libc; prints "name cycles-per-call" lines. Runs on the card.
	.include "mvex.inc"

	.set N, 200000

	.bss
	.balign 64
wblk:	.skip 4096			# a weight superblock (and what follows it)
xs:	.skip 4096			# activations / a prepared superblock
acc:	.skip 512
scratch:	.skip 512
line:	.skip 64

	.data
	.balign 64
consts:	.float 16.0, 4.0, 2.0
	.float 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625
	.float 8.0
	.fill 4, 4, 0

	.text
# say(rsi name, rax value): "name value\n" to standard output.
say:
	pushq %rbx
	pushq %r12
	movq %rax, %r12
	leaq line(%rip), %rbx
	xorl %ecx, %ecx
1:	movb (%rsi,%rcx), %al
	testb %al, %al
	jz 2f
	movb %al, (%rbx,%rcx)
	incq %rcx
	jmp 1b
2:	movb $' ', (%rbx,%rcx)
	incq %rcx
	leaq 20(%rbx,%rcx), %rdi		# digits written backwards from here
	movq %rdi, %r8
	movq %r12, %rax
	movl $10, %r9d
3:	xorl %edx, %edx
	divq %r9
	addb $'0', %dl
	decq %rdi
	movb %dl, (%rdi)
	testq %rax, %rax
	jnz 3b
4:	cmpq %r8, %rdi
	jae 5f
	movb (%rdi), %al
	movb %al, (%rbx,%rcx)
	incq %rcx
	incq %rdi
	jmp 4b
5:	movb $'\n', (%rbx,%rcx)
	incq %rcx
	movl $1, %eax
	movl $1, %edi
	movq %rbx, %rsi
	movq %rcx, %rdx
	syscall
	popq %r12
	popq %rbx
	ret

# TIME name, body: the body N times between two rdtsc, cycles per pass said.
	.macro TIME name
	.section .rodata
7:	.asciz "\name"
	.text
	leaq 7b(%rip), %r15
	movl $N, %r14d
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	movq %rax, %r13
	.endm
	.macro TIME_END
	decl %r14d
	jnz 8b
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	subq %r13, %rax
	xorl %edx, %edx
	movl $N, %ecx
	divq %rcx
	movq %r15, %rsi
	call say
	.endm
	.macro ARGS
	leaq wblk(%rip), %rdi
	leaq xs(%rip), %rsi
	leaq scratch(%rip), %rcx
	leaq acc(%rip), %r8
	leaq consts(%rip), %r9
	.endm

	.globl _start
_start:
	andq $-64, %rsp
	# weights: bytes 0x53 (lo 3, hi 5), scales a float16 1.0 (0x3c00) per block
	leaq wblk(%rip), %rdi
	movl $4096, %ecx
	movb $0x53, %al
	rep stosb
	leaq wblk(%rip), %rdi
	xorl %ecx, %ecx
1:	movw $0x3c00, (%rdi,%rcx)
	addq $18, %rcx
	cmpq $1152, %rcx
	jb 1b

	TIME "phi_q4_0_1_(old,_114_ops)"
8:	ARGS
	pushq $0
	call phi_q4_0_1
	addq $8, %rsp
	TIME_END


	TIME "phi_q4_0_row1p_one_superblock_(6_prefetch)"
8:	ARGS
	movl $1, %edx
	xorl %ecx, %ecx				# no jump: the prefetch stays on these lines
	call phi_q4_0_row1p
	TIME_END

	TIME "empty_call_and_args"
8:	ARGS
	call nothing
	TIME_END

	TIME "16_unpack_loads_uint8_(4_lo_then_4_hi)"
8:	ARGS
	.irp g, 0, 72
	VLOADUNPACKLPS zmm8, rdi, (\g+2), conv=CV_U8
	VLOADUNPACKLPS zmm9, rdi, (\g+20), conv=CV_U8
	VLOADUNPACKLPS zmm10, rdi, (\g+38), conv=CV_U8
	VLOADUNPACKLPS zmm11, rdi, (\g+56), conv=CV_U8
	VLOADUNPACKHPS zmm8, rdi, (\g+66), conv=CV_U8
	VLOADUNPACKHPS zmm9, rdi, (\g+84), conv=CV_U8
	VLOADUNPACKHPS zmm10, rdi, (\g+102), conv=CV_U8
	VLOADUNPACKHPS zmm11, rdi, (\g+120), conv=CV_U8
	.endr
	TIME_END

	TIME "8_unpack_lo_only_uint8"
8:	ARGS
	.irp g, 0, 72
	VLOADUNPACKLPS zmm8, rdi, (\g+2), conv=CV_U8
	VLOADUNPACKLPS zmm9, rdi, (\g+20), conv=CV_U8
	VLOADUNPACKLPS zmm10, rdi, (\g+38), conv=CV_U8
	VLOADUNPACKLPS zmm11, rdi, (\g+56), conv=CV_U8
	.endr
	TIME_END

	TIME "8_aligned_loads_uint8_(vmovaps)"
8:	ARGS
	.irp g, 0, 64
	VMOVAPS zmm8, rdi, (\g), conv=CV_U8
	VMOVAPS zmm9, rdi, (\g+16), conv=CV_U8
	VMOVAPS zmm10, rdi, (\g+32), conv=CV_U8
	VMOVAPS zmm11, rdi, (\g+48), conv=CV_U8
	.endr
	TIME_END

	TIME "8_vbroadcastss_float16"
8:	ARGS
	.irp g, 0, 72
	VBROADCASTSS zmm16, rdi, (\g), conv=CV_F16
	VBROADCASTSS zmm17, rdi, (\g+18), conv=CV_F16
	VBROADCASTSS zmm18, rdi, (\g+36), conv=CV_F16
	VBROADCASTSS zmm19, rdi, (\g+54), conv=CV_F16
	.endr
	TIME_END

	TIME "8_vmulps_1to16_and_8_vrndfxpntps"
8:	ARGS
	.rept 2
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16
	VRNDFXPNTPS zmm12, zmm12, 1
	VRNDFXPNTPS zmm13, zmm13, 1
	VRNDFXPNTPS zmm14, zmm14, 1
	VRNDFXPNTPS zmm15, zmm15, 1
	.endr
	TIME_END

	TIME "8_vmulps_1to16_only"
8:	ARGS
	.rept 2
	VMULPS zmm12, zmm8, r9, 24, conv=CV_1TO16
	VMULPS zmm13, zmm9, r9, 24, conv=CV_1TO16
	VMULPS zmm14, zmm10, r9, 24, conv=CV_1TO16
	VMULPS zmm15, zmm11, r9, 24, conv=CV_1TO16
	.endr
	TIME_END

	TIME "16_fma_with_memory_operands"
8:	ARGS
	.irp g, 0, 512
	VMULPS zmm8, zmm8, rsi, (\g)
	VMULPS zmm9, zmm9, rsi, (\g+128)
	VMULPS zmm10, zmm10, rsi, (\g+256)
	VMULPS zmm11, zmm11, rsi, (\g+384)
	VFMADD231PS zmm8, zmm12, rsi, (\g+64)
	VFMADD231PS zmm9, zmm13, rsi, (\g+192)
	VFMADD231PS zmm10, zmm14, rsi, (\g+320)
	VFMADD231PS zmm11, zmm15, rsi, (\g+448)
	.endr
	TIME_END

	TIME "8_vsubps_4to16"
8:	ARGS
	.irp g, 0, 64
	VSUBPS zmm8, zmm8, rsi, (1024+\g), conv=CV_4TO16
	VSUBPS zmm9, zmm9, rsi, (1040+\g), conv=CV_4TO16
	VSUBPS zmm10, zmm10, rsi, (1056+\g), conv=CV_4TO16
	VSUBPS zmm11, zmm11, rsi, (1072+\g), conv=CV_4TO16
	.endr
	TIME_END

	TIME "8_fma_register"
8:	ARGS
	.rept 2
	VFMADD231PS zmm0, zmm8, zmm16
	VFMADD231PS zmm1, zmm9, zmm17
	VFMADD231PS zmm2, zmm10, zmm18
	VFMADD231PS zmm3, zmm11, zmm19
	.endr
	TIME_END

	TIME "8_prefetches_of_lines_in_L1"
8:	ARGS
	VPREFETCH0 rdi, 0
	VPREFETCH0 rdi, 64
	VPREFETCH0 rdi, 128
	VPREFETCH0 rdi, 192
	VPREFETCH1 rdi, 0
	VPREFETCH1 rdi, 64
	VPREFETCH1 rdi, 128
	VPREFETCH1 rdi, 192
	TIME_END

	movl $60, %eax
	xorl %edi, %edi
	syscall

nothing:
	ret

	.include "q4_0.S"		# tools/q4_0-kernels.c's output (q4_0-bench.sh makes it)
