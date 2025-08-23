#include <stdint.h>
#include <string.h>

#define IN_H 32
#define IN_W 32
#define IN_C 3
#define OUT_C 32
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1
#define OUT_H 32
#define OUT_W 32
#define K_PAD 32 
#define QSHIFT 8 
#define UART_TX 0x10000000UL


static inline void uart_putc(char c) { *(volatile uint8_t *)UART_TX = (uint8_t)c; }
static void uart_puts(const char *s)
{
    while (*s)
    {
        uart_putc(*s++);
    }
}
static void uart_puthex64(uint64_t x)
{
    static const char HEX[] = "0123456789ABCDEF";
    for (int i = 15; i >= 0; i--)
    {
        uart_putc(HEX[(x >> (i * 4)) & 0xF]);
    }
}
static inline uint64_t rdcycle()
{
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

static int8_t conv0_w[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
static int32_t conv0_b[OUT_C];

static inline int8_t relu(int32_t acc)
{
    int32_t q = acc >> QSHIFT;
    if (q < 0)
    {
        q = 0;
    }
    if (q > 127)
    {
        q = 127;
    }
    return (int8_t)q;
}

static void copy16_i8_to_i16(const int8_t *src, int src_ld, int16_t *dst, int dst_ld)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            dst[i * dst_ld + j] = (int16_t)src[i * src_ld + j];
        }
    }
}
static void add16_i16(const int16_t *A, const int16_t *B, int16_t *R, int ld)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            R[i * ld + j] = A[i * ld + j] + B[i * ld + j];
        }
    }
}
static void sub16_i16(const int16_t *A, const int16_t *B, int16_t *R, int ld)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            R[i * ld + j] = A[i * ld + j] - B[i * ld + j];
        }
    }
}
static void add8_i16(const int16_t *A, const int16_t *B, int16_t *R, int ld)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            R[i * ld + j] = A[i * ld + j] + B[i * ld + j];
        }
    }
}
static void sub8_i16(const int16_t *A, const int16_t *B, int16_t *R, int ld)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            R[i * ld + j] = A[i * ld + j] - B[i * ld + j];
        }
    }
}

static void mm8_i16_i32(const int16_t *A, const int16_t *B, int32_t *C, int ldA, int ldB, int ldC)
{
    int32_t acc;
    const int16_t *ar, *bc;

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            acc = 0;
            ar = A + i * ldA;
            bc = B + j;
            for (int k = 0; k < 8; k++)
            {
                acc += (int32_t)ar[k] * (int32_t)bc[k * ldB];
            }

            C[i * ldC + j] = acc;
        }
    }
}

//Strassen 16×16 used as "base multiplication” at 32 level
static void strassen16_level1(const int16_t A[16][16], const int16_t B[16][16], int32_t C[16][16])
{

    static int16_t A11[8][8], A12[8][8], A21[8][8], A22[8][8];
    static int16_t B11[8][8], B12[8][8], B21[8][8], B22[8][8];
    static int16_t T1[8][8], T2[8][8];
    static int32_t M1[8][8], M2[8][8], M3[8][8], M4[8][8], M5[8][8], M6[8][8], M7[8][8];

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][8 + j];
            A21[i][j] = A[8 + i][j];
            A22[i][j] = A[8 + i][8 + j];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][8 + j];
            B21[i][j] = B[8 + i][j];
            B22[i][j] = B[8 + i][8 + j];
        }
    }

    // M1..M7 su 8×8 with classic kernel
    add8_i16(&A11[0][0], &A22[0][0], &T1[0][0], 8);
    add8_i16(&B11[0][0], &B22[0][0], &T2[0][0], 8);
    mm8_i16_i32(&T1[0][0], &T2[0][0], &M1[0][0], 8, 8, 8);

    add8_i16(&A21[0][0], &A22[0][0], &T1[0][0], 8);
    mm8_i16_i32(&T1[0][0], &B11[0][0], &M2[0][0], 8, 8, 8);

    sub8_i16(&B12[0][0], &B22[0][0], &T2[0][0], 8);
    mm8_i16_i32(&A11[0][0], &T2[0][0], &M3[0][0], 8, 8, 8);

    sub8_i16(&B21[0][0], &B11[0][0], &T2[0][0], 8);
    mm8_i16_i32(&A22[0][0], &T2[0][0], &M4[0][0], 8, 8, 8);

    add8_i16(&A11[0][0], &A12[0][0], &T1[0][0], 8);
    mm8_i16_i32(&T1[0][0], &B22[0][0], &M5[0][0], 8, 8, 8);

    sub8_i16(&A21[0][0], &A11[0][0], &T1[0][0], 8);
    add8_i16(&B11[0][0], &B12[0][0], &T2[0][0], 8);
    mm8_i16_i32(&T1[0][0], &T2[0][0], &M6[0][0], 8, 8, 8);

    sub8_i16(&A12[0][0], &A22[0][0], &T1[0][0], 8);
    add8_i16(&B21[0][0], &B22[0][0], &T2[0][0], 8);
    mm8_i16_i32(&T1[0][0], &T2[0][0], &M7[0][0], 8, 8, 8);

    // Recomp in C(16×16) 
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];         // C11 
            C[i][8 + j] = M3[i][j] + M5[i][j];                           // C12 
            C[8 + i][j] = M2[i][j] + M4[i][j];                           // C21 
            C[8 + i][8 + j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]; // C22 
        }
    }
}

// Strassen 32×32 with 2 levels: 32→16 uses Strassen; 16 uses Strassen→8
static void strassen32_level2(const int8_t A[32][32], const int8_t B[32][32], int32_t C[32][32])
{
    static int16_t A11[16][16], A12[16][16], A21[16][16], A22[16][16];
    static int16_t B11[16][16], B12[16][16], B21[16][16], B22[16][16];
    static int16_t T1[16][16], T2[16][16];
    static int32_t M1[16][16], M2[16][16], M3[16][16], M4[16][16], M5[16][16], M6[16][16], M7[16][16];

    copy16_i8_to_i16(&A[0][0], 32, &A11[0][0], 16);
    copy16_i8_to_i16(&A[0][16], 32, &A12[0][0], 16);
    copy16_i8_to_i16(&A[16][0], 32, &A21[0][0], 16);
    copy16_i8_to_i16(&A[16][16], 32, &A22[0][0], 16);

    copy16_i8_to_i16(&B[0][0], 32, &B11[0][0], 16);
    copy16_i8_to_i16(&B[0][16], 32, &B12[0][0], 16);
    copy16_i8_to_i16(&B[16][0], 32, &B21[0][0], 16);
    copy16_i8_to_i16(&B[16][16], 32, &B22[0][0], 16);

    // M1..M7 at 32 level: every 16×16 product is Strassen16 (which uses an 8×8 base) 
    add16_i16(&A11[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B22[0][0], &T2[0][0], 16);
    strassen16_level1((const int16_t (*)[16])T1, (const int16_t (*)[16])T2, (int32_t (*)[16])M1);

    add16_i16(&A21[0][0], &A22[0][0], &T1[0][0], 16);
    strassen16_level1((const int16_t (*)[16])T1, (const int16_t (*)[16])B11, (int32_t (*)[16])M2);

    sub16_i16(&B12[0][0], &B22[0][0], &T2[0][0], 16);
    strassen16_level1((const int16_t (*)[16])A11, (const int16_t (*)[16])T2, (int32_t (*)[16])M3);

    sub16_i16(&B21[0][0], &B11[0][0], &T2[0][0], 16);
    strassen16_level1((const int16_t (*)[16])A22, (const int16_t (*)[16])T2, (int32_t (*)[16])M4);

    add16_i16(&A11[0][0], &A12[0][0], &T1[0][0], 16);
    strassen16_level1((const int16_t (*)[16])T1, (const int16_t (*)[16])B22, (int32_t (*)[16])M5);

    sub16_i16(&A21[0][0], &A11[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B12[0][0], &T2[0][0], 16);
    strassen16_level1((const int16_t (*)[16])T1, (const int16_t (*)[16])T2, (int32_t (*)[16])M6);

    sub16_i16(&A12[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B21[0][0], &B22[0][0], &T2[0][0], 16);
    strassen16_level1((const int16_t (*)[16])T1, (const int16_t (*)[16])T2, (int32_t (*)[16])M7);

    // Final recomp in C (32×32)
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];           // C11
            C[i][16 + j] = M3[i][j] + M5[i][j];                            // C12
            C[16 + i][j] = M2[i][j] + M4[i][j];                            // C21
            C[16 + i][16 + j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]; // C22
        }
    }
}

// Padded weights in A
static void buildA(int8_t A[32][32])
{
    int k;
    for (int oc = 0; oc < OUT_C; oc++)
    {
        k = 0;
        for (int ic = 0; ic < IN_C; ic++)
        {
            for (int kh = 0; kh < KERNEL_SIZE; kh++)
            {
                for (int kw = 0; kw < KERNEL_SIZE; kw++)
                {
                    A[oc][k++] = conv0_w[oc][ic][kh][kw];
                }
            }
        }

        for (; k < K_PAD; k++)
        {
            A[oc][k] = 0;
        }
    }
}

// Build B 
static void buildB(const int8_t input[IN_C][IN_H][IN_W], int oh, int ow, int8_t B[32][32])
{
    int ih, iw, idx = 0;
    int8_t v;

    for (int r = 0; r < 32; r++)
    {
        for (int c = 0; c < 32; c++)
        {
            B[r][c] = 0;
        }
    }

    // fill just column 0 with the patch
    for (int ic = 0; ic < IN_C; ic++)
    {
        for (int kh = 0; kh < KERNEL_SIZE; kh++)
        {
            ih = oh + kh - PADDING;
            for (int kw = 0; kw < KERNEL_SIZE; kw++)
            {
                iw = ow + kw - PADDING;
                v = 0;
                if ((unsigned)ih < IN_H && (unsigned)iw < IN_W)
                {
                    v = input[ic][ih][iw];
                }
                B[idx++][0] = v;
            }
        }
    }
}

void conv0_strassen(const int8_t input[IN_C][IN_H][IN_W], int8_t output[OUT_C][OUT_H][OUT_W])
{
    static int8_t A[32][32];
    static int8_t B[32][32];
    static int32_t C[32][32];
    int32_t acc;

    buildA(A);

    for (int oh = 0; oh < OUT_H; oh++)
    {
        for (int ow = 0; ow < OUT_W; ow++)
        {
            // B for the single pixel
            buildB(input, oh, ow, B);

            // Strassen 32×32 at 2 levels: C = A * B
            strassen32_level2((const int8_t (*)[32])A, (const int8_t (*)[32])B, C);

            for (int oc = 0; oc < OUT_C; oc++)
            {
                acc = C[oc][0] + conv0_b[oc];
                output[oc][oh][ow] = relu(acc);
            }
        }
    }
}

int main()
{
    static int8_t input[IN_C][IN_H][IN_W];
    static int8_t output[OUT_C][OUT_H][OUT_W];

    for (int h = 0; h < IN_H; ++h) //example values
    {
        for (int w = 0; w < IN_W; ++w)
        {
            input[0][h][w] = 1;
            input[1][h][w] = 2;
            input[2][h][w] = 3;
        }
    }

    for (int oc = 0; oc < OUT_C; oc++)
    {
        conv0_b[oc] = 0;
        for (int ic = 0; ic < IN_C; ic++)
        {
            for (int kh = 0; kh < KERNEL_SIZE; kh++)
            {
                for (int kw = 0; kw < KERNEL_SIZE; kw++)
                {
                    conv0_w[oc][ic][kh][kw] = 1;
                }
            }
        }
    }

    uint64_t c0 = rdcycle();
    conv0_strassen(input, output);
    uint64_t c1 = rdcycle();

    uart_puts("conv0_strassen_2liv cycles: 0x");
    uart_puthex64(c1 - c0);
    uart_puts("\n");

    for (;;)
        ;
    return 0;
}
