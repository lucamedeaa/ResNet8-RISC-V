.option norvc
.option nopic
.section .text
.global _start

UART_TX = 0x10000000

_start:
    la   sp, _stack_top

    csrr s10, mcycle

    la   a0, input
    la   a1, output
    la   a2, weights
    la   a3, bias
    call conv0_v1

    csrr s11, mcycle
    sub  s9, s11, s10

    li   s0, 16
print_hex:
    srli s1, s9, 60
    andi s1, s1, 0xF
    la   s2, HEX_CHARS
    add  s1, s1, s2
    lb   s1, 0(s1)
    li   s3, UART_TX
    sb   s1, 0(s3)
    slli s9, s9, 4
    addi s0, s0, -1
    bnez s0, print_hex

    li   s1, '\n'
    sb   s1, 0(s3)
    j    .

.section .rodata
HEX_CHARS:
    .ascii "0123456789ABCDEF"

.globl conv0_v1
conv0_v1:
    li   s0, 0                  # out_ch
out_ch:
    li   s1, 0                  # out_h 
out_h:
    li   s2, 0                  # out_w  
out_w:
    li   s3, 0                  # acc = 0

slli t0, s0, 2              
add  t0, a3, t0
lw   s3, 0(t0)

    li   t0, 1024               # 32*32
    mul  t1, s0, t0             # oc*1024
    li   t2, 32
    mul  t3, s1, t2             # oh*32
    add  t3, t3, s2             # oh*32 + ow
    add  t3, t3, t1             # + oc*1024
    add  t3, a1, t3             # t3 = &output[oc][oh][ow]

    # acc on the 3 channels and 3x3 with PADDING 1
    li   s4, 0                  # in_ch (ic)
loop_in_ch:
    li   s5, 0                  # k_h (kh)
loop_kh:
    li   s6, 0                  # k_w (kw)
loop_kw:
    # in_h = oh + kh - 1  (PADDING=1)
    # in_w = ow + kw - 1
    add  t4, s1, s5             # oh + kh
    addi t4, t4, -1             # ih
    add  t5, s2, s6             # ow + kw
    addi t5, t5, -1             # iw

    #Border check: 0 <= ih,iw < 32 
    blt  t4, x0, skip_mac
    li   t6, 32
    bge  t4, t6, skip_mac
    blt  t5, x0, skip_mac
    li   t6, 32
    bge  t5, t6, skip_mac

    #input[in_ch][ih][iw] 
    li   t6, 1024               # 32*32
    mul  s7, s4, t6             
    li   t6, 32
    mul  t6, t4, t6             # ih*32
    add  t6, t6, t5             # ih*32 + iw
    add  t6, t6, s7             # + in_ch*1024
    add  t6, a0, t6             # &input[in_ch][ih][iw]
    lbu  t4, 0(t6)
    sext.b t4, t4               # int8 -> int32

    #weight[out_ch][in_ch][k_h][k_w] 
    li   t6, 27
    mul  s8, s0, t6             # s8 = oc*27
    li   t6, 9
    mul  t0, s4, t6             # in_ch*9
    add  s8, s8, t0
    li   t6, 3
    mul  t0, s5, t6             # k_h*3
    add  s8, s8, t0
    add  s8, s8, s6             # + k_w
    add  s8, a2, s8
    lbu  t5, 0(s8)
    sext.b t5, t5               # int8 -> int32

    # MAC
    mul  t5, t4, t5
    add  s3, s3, t5

skip_mac:
    addi s6, s6, 1
    li   t6, 3
    blt  s6, t6, loop_kw
    addi s5, s5, 1
    blt  s5, t6, loop_kh
    addi s4, s4, 1
    blt  s4, t6, loop_in_ch

    # Quantization >>8, then ReLU
    srai s3, s3, 8
    bge  s3, x0, 1f
    li   s3, 0
1:
    li   t0, 127
    ble  s3, t0, 2f
    mv   s3, t0
2:
    sb   s3, 0(t3)

    #Next ow/oh/oc
    addi s2, s2, 1
    li   t0, 32
    blt  s2, t0, out_w

    li   s2, 0
    addi s1, s1, 1
    blt  s1, t0, out_h

    li   s1, 0
    addi s0, s0, 1
    li   t0, 32
    blt  s0, t0, out_ch

    ret
