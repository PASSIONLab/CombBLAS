# Copyright (C) 2010-2013 The Trustees of Indiana University.            
#                                                                        
# Use, modification and distribution is subject to the Boost Software    
# License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)                                  
#                                                                        
#  Authors: Jeremiah Willcock                                            
#           Andrew Lumsdaine                                             
#
# Modified from GCC 4.8.1-generated code for some functions in kernels.c.

	.file	"kernels.c"
	.text
	.p2align 5,,31
	.globl	bitmap_matvec_and_mark_impl
	.type	bitmap_matvec_and_mark_impl, @function
bitmap_matvec_and_mark_impl:
.LFB40:
	pushq	%r15
.LCFI0:
	shrq	$16, %r9
	pushq	%r14
.LCFI1:
	pushq	%r13
.LCFI2:
	pushq	%r12
.LCFI3:
	pushq	%rbp
.LCFI4:
	pushq	%rbx
.LCFI5:
	movq	80(%rsp), %r12
	movq	%rdi, -32(%rsp)
	movq	%rsi, -24(%rsp)
	movq	%r9, -8(%rsp)
	je	.L3
	movq	72(%rsp), %rax
	movq	$0, -16(%rsp)
	movq	%rcx, %r13
	movq	%r8, %r9
	xorl	%ebx, %ebx
	xorl	%r10d, %r10d
	movl	$1, %r14d
	shrq	$16, %rax
	movq	%rax, -40(%rsp)
	.p2align 5,,24
	.p2align 3
.L15:
	movq	56(%rsp), %rdi
	movq	-16(%rsp), %rdx
	cmpb	$0, (%rdi,%rdx)
	jne	.L17
	cmpq	$0, -40(%rsp)
	je	.L6
	movq	64(%rsp), %r11
	xorl	%r15d, %r15d
	.p2align 5,,24
	.p2align 3
.L10:
	movzbl	0(%r13,%rbx), %edx
	incq	%rbx
	cmpl	$255, %edx
	je	.L28
.L7:
	movq	-32(%rsp), %r8
	leaq	(%r10,%r10), %rbp
	xorl	%eax, %eax
	addq	%rbp, %r8
	addq	-24(%rsp), %rbp
	testq	%rdx, %rdx
	je	.L12
	movq	%rbx, -48(%rsp)
	.p2align 5,,24
	.p2align 3
.L27:
	movzwl	(%r8,%rax,2), %ecx
	movl	%ecx, %esi
	shrl	$6, %esi
	movq	(%r9,%rsi,8), %rsi
	btq	%rcx, %rsi
	jnc	.L11
	movzwl	0(%rbp,%rax,2), %ecx
	movl	%ecx, %esi
	shrl	$6, %esi
	movq	(%r11,%rsi,8), %rdi
	btq	%rcx, %rdi
	jc	.L11
	movq	%r14, %rbx
	salq	%cl, %rbx
	movq	%rbx, %rcx
	orq	%rdi, %rcx
	movq	%r14, %rdi
	movq	%rcx, (%r11,%rsi,8)
	leaq	(%rax,%r10), %rcx
	movq	%rcx, %rsi
	shrq	$6, %rsi
	salq	%cl, %rdi
	orq	%rdi, (%r12,%rsi,8)
.L11:
	incq	%rax
	cmpq	%rdx, %rax
	jne	.L27
	movq	-48(%rsp), %rbx
.L12:
	addq	%rdx, %r10
	incq	%r15
	addq	$8192, %r11
	cmpq	-40(%rsp), %r15
	jne	.L10
.L6:
	incq	-16(%rsp)
	movq	-8(%rsp), %rax
	addq	$8192, %r9
	cmpq	%rax, -16(%rsp)
	jne	.L15
.L3:
	popq	%rbx
.LCFI6:
	popq	%rbp
.LCFI7:
	popq	%r12
.LCFI8:
	popq	%r13
.LCFI9:
	popq	%r14
.LCFI10:
	popq	%r15
.LCFI11:
	ret
.L17:
.LCFI12:
	movq	-8(%rsp), %rcx
	xorl	%eax, %eax
	.p2align 5,,24
	.p2align 3
.L5:
	movzbl	0(%r13,%rbx), %edx
	incq	%rbx
	cmpl	$255, %edx
	je	.L29
.L13:
	incq	%rax
	addq	%rdx, %r10
	cmpq	%rcx, %rax
	jne	.L5
	jmp	.L6
.L28:
	movzbl	0(%r13,%rbx), %eax
	incq	%rbx
	movzbl	%al, %ecx
	addq	%rcx, %rdx
	cmpb	$-1, %al
	je	.L28
	jmp	.L7
	.p2align 5,,7
	.p2align 3
.L29:
	movzbl	0(%r13,%rbx), %esi
	incq	%rbx
	movzbl	%sil, %edi
	addq	%rdi, %rdx
	cmpb	$-1, %sil
	je	.L29
	jmp	.L13
.LFE40:
	.size	bitmap_matvec_and_mark_impl, .-bitmap_matvec_and_mark_impl
	.p2align 5,,31
	.globl	bitmap_matvec_trans_and_mark_impl
	.type	bitmap_matvec_trans_and_mark_impl, @function
bitmap_matvec_trans_and_mark_impl:
.LFB41:
	pushq	%r15
.LCFI13:
	pushq	%r14
.LCFI14:
	pushq	%r13
.LCFI15:
	pushq	%r12
.LCFI16:
	pushq	%rbp
.LCFI17:
	pushq	%rbx
.LCFI18:
	movq	72(%rsp), %rax
	movq	%rdi, -48(%rsp)
	movq	%rsi, -40(%rsp)
	movq	%r8, -8(%rsp)
	movq	80(%rsp), %r14
	shrq	$16, %rax
	movq	%rax, -24(%rsp)
	je	.L43
	movq	56(%rsp), %r8
	shrq	$16, %r9
	movq	64(%rsp), %r12
	movq	%r9, -16(%rsp)
	movq	$0, -32(%rsp)
	movq	%rcx, %r13
	xorl	%ebx, %ebx
	xorl	%r11d, %r11d
	movl	$1, %r15d
	addq	%r9, %r8
	movq	%r8, -56(%rsp)
.L53:
	cmpq	$0, -16(%rsp)
	je	.L45
	movq	56(%rsp), %r10
	movq	-8(%rsp), %rsi
	.p2align 5,,24
	.p2align 3
.L49:
	movzbl	0(%r13,%rbx), %edx
	incq	%rbx
	cmpq	$255, %rdx
	je	.L63
.L46:
	cmpb	$0, (%r10)
	jne	.L50
	movq	-48(%rsp), %rbp
	leaq	(%r11,%r11), %r9
	addq	%r9, %rbp
	addq	-40(%rsp), %r9
	testq	%rdx, %rdx
	je	.L50
	movq	%rbx, -64(%rsp)
	xorl	%eax, %eax
	.p2align 5,,24
	.p2align 3
.L52:
	movzwl	(%r9,%rax,2), %ecx
	movl	%ecx, %edi
	shrl	$6, %edi
	movq	(%rsi,%rdi,8), %rdi
	btq	%rcx, %rdi
	jnc	.L51
	movzwl	0(%rbp,%rax,2), %ecx
	movl	%ecx, %edi
	shrl	$6, %edi
	movq	(%r12,%rdi,8), %r8
	btq	%rcx, %r8
	jc	.L51
	movq	%r15, %rbx
	salq	%cl, %rbx
	movq	%rbx, %rcx
	orq	%r8, %rcx
	movq	%r15, %r8
	movq	%rcx, (%r12,%rdi,8)
	leaq	(%rax,%r11), %rcx
	movq	%rcx, %rdi
	shrq	$6, %rdi
	salq	%cl, %r8
	orq	%r8, (%r14,%rdi,8)
.L51:
	incq	%rax
	cmpq	%rdx, %rax
	jne	.L52
	movq	-64(%rsp), %rbx
.L50:
	addq	%rdx, %r11
	incq	%r10
	addq	$8192, %rsi
	cmpq	-56(%rsp), %r10
	jne	.L49
.L45:
	incq	-32(%rsp)
	movq	-24(%rsp), %rax
	addq	$8192, %r12
	cmpq	%rax, -32(%rsp)
	jne	.L53
.L43:
	popq	%rbx
.LCFI19:
	popq	%rbp
.LCFI20:
	popq	%r12
.LCFI21:
	popq	%r13
.LCFI22:
	popq	%r14
.LCFI23:
	popq	%r15
.LCFI24:
	ret
.L63:
.LCFI25:
	movzbl	0(%r13,%rbx), %eax
	incq	%rbx
	movzbl	%al, %ecx
	addq	%rcx, %rdx
	cmpb	$-1, %al
	je	.L63
	jmp	.L46
.LFE41:
	.size	bitmap_matvec_trans_and_mark_impl, .-bitmap_matvec_trans_and_mark_impl
	.align 8
.LEFDE43:
	.ident	"GCC: (GNU) 4.7.2 20120920 (Cray Inc.)"
	.section	.note.GNU-stack,"",@progbits
