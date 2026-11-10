# q4_0-stream.S: the whole-row Q4_0 kernel streaming a weight buffer (BIG
# bytes, 64 MiB unless the assembler is given --defsym BIG=) on
# one thread, rows of 20 superblocks (2880 bytes), against one prepared
# activation row in the first-level cache; cycles per superblock for each
# prefetch variant, rows consecutive and every other row.
	.include "mvex.inc"
	.ifndef BIG
	.set BIG, 67108864
	.endif
	.set ROWB, 2880
	.bss
	.balign 4096
big:	.skip BIG + 65536
	.balign 64
prep:	.skip 23040
acc:	.skip 64
line:	.skip 64
	.data
	.balign 64
consts:	.float 16.0, 4.0, 2.0
	.float 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625
	.float 8.0
	.fill 4, 4, 0
	.text
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
	leaq 20(%rbx,%rcx), %rdi
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

# STREAM name, kern, step: the rows of big, every step-th, through kern;
# cycles per superblock, in tenths.
	.macro STREAM name, kern, step
	.section .rodata
7:	.asciz "\name"
	.text
	leaq big(%rip), %r12
	leaq big + BIG(%rip), %r14
	xorl %r15d, %r15d
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	movq %rax, %r13
8:	movq %r12, %rdi
	leaq prep(%rip), %rsi
	movl $20, %edx
	leaq acc(%rip), %r8
	leaq consts(%rip), %r9
	movl $ROWB * (\step - 1), %ecx		# the jump to the next row taken
	call \kern
	incq %r15
	addq $ROWB * \step, %r12
	cmpq %r14, %r12
	jb 8b
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	subq %r13, %rax
	imulq $10, %rax, %rax
	imulq $20, %r15, %rcx
	xorl %edx, %edx
	divq %rcx
	leaq 7b(%rip), %rsi
	call say
	.endm

	.globl _start
_start:
	andq $-64, %rsp
	leaq big(%rip), %rdi
	movl $BIG + 65536, %ecx
	movb $0x53, %al
	rep stosb
	leaq prep(%rip), %rdi			# written, so it is 23040 bytes of its own and not the shared zero page
	movl $23040 / 4, %ecx
	movl $0x3f800000, %eax
	rep stosl
	STREAM "none_step1", kv_none, 1
	STREAM "pf_576_2304_step1", kv_576_2304, 1
	STREAM "pf_288_1152_step1", kv_288_1152, 1
	STREAM "pf_1152_4608_step1", kv_1152_4608, 1
	STREAM "pf_1152_8640_step1", kv_1152_8640, 1
	STREAM "pf_2304_11520_step1", kv_2304_11520, 1
	STREAM "pf_L2only_2304_step1", kv_0_2304, 1
	STREAM "pf_L1only_576_step1", kv_576_0, 1
	STREAM "none_step2", kv_none, 2
	STREAM "pf_576_2304_step2", kv_576_2304, 2
	STREAM "pf_1152_8640_step2", kv_1152_8640, 2
	movl $60, %eax
	xorl %edi, %edi
	syscall
	.include "q4_0-variants.S"	# q4_0-bench.sh: the row kernel at each prefetch distance
