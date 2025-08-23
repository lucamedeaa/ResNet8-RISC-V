.option norvc
.option nopic
.section .data
.balign 1

# INPUT 3x32x32
.global input
input:
    .rept 32*32*3
        .byte 1
    .endr

#WEIGHTS 32x3x3x3
.global weights
weights:
    .rept 32*3*3*3
        .byte 1
    .endr

#BIAS 32 
.global bias
bias:
    .rept 32
        .word 0
    .endr

# OUTPUT BUFFER 32x32x32 
.global output
output:
    .space 32*32*32
