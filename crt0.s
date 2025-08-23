    .section .text
    .globl _start
_start:
    la   sp, _stack_top      # stack
    call main                # calls main C
1:  j 1b                     # infinite loop
