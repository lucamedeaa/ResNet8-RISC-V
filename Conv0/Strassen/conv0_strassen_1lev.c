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
#define TILE_N 32
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

static inline int8_t relu(int32_t acc_q24)
{
    int32_t q = acc_q24 >> QSHIFT;
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

// Copy/operations on submatrix 16x16
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

// C = A * B
static void mm16_i16_i32(const int16_t *A, const int16_t *B, int32_t *C, int ldA, int ldB, int ldC)
{
    int32_t acc;
    const int16_t *ar;
    const int16_t *bc;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            acc = 0;
            ar = A + i * ldA;
            bc = B + j;
            for (int k = 0; k < 16; k++)
            {
                acc += (int32_t)ar[k] * (int32_t)bc[k * ldB];
            }
            C[i * ldC + j] = acc;
        }
    }
}

// C += X
static void add16_i32_inplace(int32_t *C, const int32_t *X, int ld)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            C[i * ld + j] += X[i * ld + j];
        }
    }
}
static void sub16_i32_inplace(int32_t *C, const int32_t *X, int ld)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            C[i * ld + j] -= X[i * ld + j];
        }
    }
}

// Strassen 32×32 (1 level) on int8 × int8 -> int32 Partition in 16x16 blocks and applies the 7 multiplications M1..M7.

static void strassen_mul(const int8_t A[32][32], const int8_t B[32][32], int32_t C[32][32])
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

    // M1 = (A11 + A22)*(B11 + B22)
    add16_i16(&A11[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M1[0][0], 16, 16, 16);

    // M2 = (A21 + A22)*B11
    add16_i16(&A21[0][0], &A22[0][0], &T1[0][0], 16);
    mm16_i16_i32(&T1[0][0], &B11[0][0], &M2[0][0], 16, 16, 16);

    // M3 = A11*(B12 - B22)
    sub16_i16(&B12[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&A11[0][0], &T2[0][0], &M3[0][0], 16, 16, 16);

    // M4 = A22*(B21 - B11)
    sub16_i16(&B21[0][0], &B11[0][0], &T2[0][0], 16);
    mm16_i16_i32(&A22[0][0], &T2[0][0], &M4[0][0], 16, 16, 16);

    // M5 = (A11 + A12)*B22
    add16_i16(&A11[0][0], &A12[0][0], &T1[0][0], 16);
    mm16_i16_i32(&T1[0][0], &B22[0][0], &M5[0][0], 16, 16, 16);

    // M6 = (A21 - A11)*(B11 + B12)
    sub16_i16(&A21[0][0], &A11[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B12[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M6[0][0], 16, 16, 16);

    // M7 = (A12 - A22)*(B21 + B22)
    sub16_i16(&A12[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B21[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M7[0][0], 16, 16, 16);

    // Combine into C
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            int32_t m1 = M1[i][j], m2 = M2[i][j], m3 = M3[i][j];
            int32_t m4 = M4[i][j], m5 = M5[i][j], m6 = M6[i][j], m7 = M7[i][j];

            C[i][j] = m1 + m4 - m5 + m7;           // C11
            C[i][16 + j] = m3 + m5;                // C12
            C[16 + i][j] = m2 + m4;                // C21
            C[16 + i][16 + j] = m1 - m2 + m3 + m6; // C22
        }
    }
}

static void buildA(int8_t A[32][32]) // copy 27 weights in order (ic,kh,kw),then zeros until 32
{
    int k;
    for (int oc = 0; oc < OUT_C; ++oc)
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

static void buildB(const int8_t input[IN_C][IN_H][IN_W], int tile_base, int8_t B[K_PAD][TILE_N])
{
    int idx, lin, oh, ow, ih, iw;
    int8_t v;
    for (int col = 0; col < TILE_N; col++)
    {
        idx = 0;
        lin = tile_base + col;
        oh = lin / OUT_W;
        ow = lin % OUT_W;

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
                    B[idx++][col] = v;
                }
            }
        }
        for (; idx < K_PAD; idx++)
        {
            B[idx][col] = 0;
        }
    }
}

void conv0_strassen(const int8_t input[IN_C][IN_H][IN_W], int8_t output[OUT_C][OUT_H][OUT_W])
{
    int lin, oh, ow;
    int32_t acc, b;
    static int8_t A[32][32];
    static int8_t B[K_PAD][TILE_N];
    static int32_t C[OUT_C][TILE_N];

    buildA(A);

    for (int tile_base = 0; tile_base < OUT_H * OUT_W; tile_base += TILE_N)
    {
        buildB(input, tile_base, B);

        strassen_mul((const int8_t (*)[32])A, (const int8_t (*)[32])B, C);

        for (int oc = 0; oc < OUT_C; oc++)
        {
            b = conv0_b[oc];
            for (int col = 0; col < TILE_N; col++)
            {
                lin = tile_base + col;
                oh = lin / OUT_W;
                ow = lin % OUT_W;
                acc = C[oc][col] + b;
                output[oc][oh][ow] = relu(acc);
            }
        }
    }
}

int main()
{
    static int8_t input[IN_C][IN_H][IN_W];
    static int8_t output[OUT_C][OUT_H][OUT_W];

    // Example values
    for (int h = 0; h < IN_H; ++h)
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

    uart_puts("conv0 strassen cycles: 0x");
    uart_puthex64(c1 - c0);
    uart_puts("\n");

    for (;;)
        ;
    return 0;
}
