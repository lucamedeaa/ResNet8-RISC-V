.option norvc
.option nopic
.section .text
.global _start
.global conv0_v2
UART_TX = 0x10000000

# _start: create halo 34x34x3,cycle count, calls conv0_v2, HEX print
_start:
    la   sp, _stack_top

    csrr s10, mcycle

    li   t0, 1156               # 34*34  (stride halo channel, in bytes)
    li   t1, 1024               # 32*32  (stride input/output channel, in bytes)
    li   t2, 34                 # stride halo row
    li   t3, 32                 # stride input/output row

    la   s0, input              
    la   s1, input_halo         

    # Halo Reset
    li   t4, 1156*3
    mv   t5, s1
zero_loop:
    sb   x0, 0(t5)
    addi t5, t5, 1
    addi t4, t4, -1
    bnez t4, zero_loop

    li   s2, 0                  # in_ch = 0..2
copy_ic:
    # halo base = s1 + ic*1156 + 34 + 1
    mul  t4, s2, t0
    add  t4, t4, s1
    addi t4, t4, 34
    addi t4, t4, 1
    # input base = s0+ic*1024
    mul  t5, s2, t1
    add  t5, t5, s0

    li   s3, 32                 # rows
copy_row:
    mv   t6, t5                 # ptr input row
    mv   t0, t4                 # ptr halo row
    li   s4, 32                 # cols
copy_col:
    lbu  t1, 0(t6)             
    sb   t1, 0(t0)
    addi t6, t6, 1
    addi t0, t0, 1
    addi s4, s4, -1
    bnez s4, copy_col

    add  t4, t4, t2             # halo:+34
    add  t5, t5, t3             # in:+32
    addi s3, s3, -1
    bnez s3, copy_row

    addi s2, s2, 1
    li   t0, 3
    blt  s2, t0, copy_ic
 
    la   a0, input_halo         # padded input
    la   a1, output             # output[32][32][32] 
    la   a2, weights            # weights[32][3][3][3] 
    la   a3, bias               # bias [32]
    call conv0_v2

    # mcycle end & print 
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

conv0_v2:
    li   s0, 0                  # oc = 0..31
oc_loop:
    li   s1, 0                  # oh = 0..31
oh_loop:
    li   s2, 0                  # ow = 0..31
ow_loop:
    # acc = bias[oc] 
    slli t0, s0, 2              # oc*4
    add  t0, a3, t0
    lw   s3, 0(t0)              # s3 = acc

    # output ptr = a1 + oc*1024 + oh*32 + ow 
    li   t1, 1024
    mul  t2, s0, t1             # oc*1024
    li   t3, 32
    mul  t4, s1, t3             # oh*32
    add  t4, t4, s2             # oh*32 + ow
    add  t4, t4, t2
    add  t4, a1, t4             # t4 = &output[oc][oh][ow]

    #  Pre-calculated input base ptr 
    li   t5, 34
    mul  t6, s1, t5             # oh*34
    add  t6, t6, s2             # oh*34 + ow
    add  s7, a0, t6             # s7 = in0 = &halo[0][oh][ow]
    li   t0, 1156               
    add  s8, s7, t0             # s8 = in1 = &halo[1][oh][ow]
    add  s9, s8, t0             # s9 = in2 = &halo[2][oh][ow]

    # Precalcolo weight base ptr  
    li   t1, 27
    mul  s4, s0, t1             # s4 = oc*27 (byte)
    add  s4, s4, a2             # s4 = &W[oc][0][0][0]
    li   t2, 9
    add  s5, s4, t2             # s5 = &W[oc][1][0][0]
    add  s6, s5, t2             # s6 = &W[oc][2][0][0]


    # 27 MAC unrolled (3 channels × 3 kernel rows × 3 kernel columns)                      

    # Channel 0 (base: s7, weights: s4)
    # row 0
    lb t0, 0(s7);  lb  t1, 0(s4);  mul t1, t0, t1;  add s3, s3, t1  #loads input and weight, computes input*weight and adds the product to acc (s3)
    lb t0, 1(s7);  lb  t1, 1(s4);  mul t1, t0, t1;  add s3, s3, t1  
    lb t0, 2(s7);  lb  t1, 2(s4);  mul t1, t0, t1;  add s3, s3, t1

    # row 1
    addi s7, s7, 34
    addi s4, s4, 3
    lb t0, 0(s7);  lb  t1, 0(s4);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s7);  lb  t1, 1(s4);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s7);  lb  t1, 2(s4);  mul t1, t0, t1;  add s3, s3, t1

    # row 2
    addi s7, s7, 34
    addi s4, s4, 3
    lb t0, 0(s7);  lb  t1, 0(s4);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s7);  lb  t1, 1(s4);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s7);  lb  t1, 2(s4);  mul t1, t0, t1;  add s3, s3, t1

    # Channel 1 (base: s8, weights: s5)
    # row 0
    lb t0, 0(s8);  lb  t1, 0(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s8);  lb  t1, 1(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s8);  lb  t1, 2(s5);  mul t1, t0, t1;  add s3, s3, t1

    # row 1
    addi s8, s8, 34
    addi s5, s5, 3
    lb t0, 0(s8);  lb  t1, 0(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s8);  lb  t1, 1(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s8);  lb  t1, 2(s5);  mul t1, t0, t1;  add s3, s3, t1

    # row 2
    addi s8, s8, 34
    addi s5, s5, 3
    lb t0, 0(s8);  lb  t1, 0(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s8);  lb  t1, 1(s5);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s8);  lb  t1, 2(s5);  mul t1, t0, t1;  add s3, s3, t1

    # Channel 2 (base: s9, weights: s6)
    # row 0
    lb t0, 0(s9);  lb  t1, 0(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s9);  lb  t1, 1(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s9);  lb  t1, 2(s6);  mul t1, t0, t1;  add s3, s3, t1

    # row 1
    addi s9, s9, 34
    addi s6, s6, 3
    lb t0, 0(s9);  lb  t1, 0(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s9);  lb  t1, 1(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s9);  lb  t1, 2(s6);  mul t1, t0, t1;  add s3, s3, t1

    # row 2
    addi s9, s9, 34
    addi s6, s6, 3
    lb t0, 0(s9);  lb  t1, 0(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 1(s9);  lb  t1, 1(s6);  mul t1, t0, t1;  add s3, s3, t1
    lb t0, 2(s9);  lb  t1, 2(s6);  mul t1, t0, t1;  add s3, s3, t1

    # Quantization, ReLU and saturation
    srai s3, s3, 8              # >>8 (quant)
    bge  s3, x0, 1f             # ReLU
    li   s3, 0
1:
    li   t0, 127                # clamp high
    ble  s3, t0, 2f
    li   s3, 127
2:
    sb   s3, 0(t4)             

    #next ow/oh/oc
    addi s2, s2, 1
    li   t0, 32
    blt  s2, t0, ow_loop

    li   s2, 0
    addi s1, s1, 1
    blt  s1, t0, oh_loop

    li   s1, 0
    addi s0, s0, 1
    li   t0, 32
    blt  s0, t0, oc_loop
    ret

.section .bss
.balign 4
input_halo: .space 34*34*3       
# output e input/weights/bias are expected in data_v2.s

.section .rodata
HEX_CHARS: .ascii "0123456789ABCDEF"

.section .stack
.balign 16
_space_stack: .space 0x1000
_stack_top:
