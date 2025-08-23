#include <stdint.h>
#define IN_H 32
#define IN_W 32
#define IN_C 3
#define OUT_C 32
#define K 3 // kernel size
#define STRIDE 1
#define PADDING 1
#define OUT_H 32
#define OUT_W 32
#define K_PAD 32
#define TILE_N 32
#define QSHIFT 8
#define POOL_SHIFT 10 // average 32*32 = 1024 -> >>10
#define NUM_CLASSES 10

#define UART_TX 0x10000000UL

// UART print
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
    static const char H[] = "0123456789ABCDEF";
    for (int i = 15; i >= 0; i--)
    {
        uart_putc(H[(x >> (i * 4)) & 0xF]);
    }
}
static inline void uart_nl() { uart_putc('\n'); }
static inline uint64_t rdcycle()
{
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

static int8_t conv0_w[OUT_C][IN_C][K][K];
static int32_t conv0_b[OUT_C];

static int8_t rb1_w1[OUT_C][OUT_C][K][K], rb1_w2[OUT_C][OUT_C][K][K];
static int32_t rb1_b1[OUT_C], rb1_b2[OUT_C];

static int8_t rb2_w1[OUT_C][OUT_C][K][K], rb2_w2[OUT_C][OUT_C][K][K];
static int32_t rb2_b1[OUT_C], rb2_b2[OUT_C];

static int8_t rb3_w1[OUT_C][OUT_C][K][K], rb3_w2[OUT_C][OUT_C][K][K];
static int32_t rb3_b1[OUT_C], rb3_b2[OUT_C];

static int8_t fc_w[NUM_CLASSES][OUT_C];
static int32_t fc_b[NUM_CLASSES];

static inline int8_t relu(int32_t acc)
{
    int32_t q = acc >> QSHIFT;
    if (q < 0)
    {
        q = 0;
    }
    else if (q > 127)
    {
        q = 127;
    }

    return (int8_t)q;
}
static inline int8_t quant_clip(int32_t acc)
{
    int32_t q = acc >> QSHIFT;
    if (q < -128)
    {
        q = -128;
    }
    else if (q > 127)
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
    const int16_t *ar, *bc;
    int32_t acc;
    for (int i = 0; i < 16; i++)
    {
        ar = A + i * ldA;
        for (int j = 0; j < 16; j++)
        {
            bc = B + j;
            acc = 0;
            for (int k = 0; k < 16; k++)
            {
                acc += (int32_t)ar[k] * (int32_t)bc[k * ldB];
            }
            C[i * ldC + j] = acc;
        }
    }
}

// Strassen 32×32: (int8)x(int8)->int32 Partition in 16x16 blocks and applies the 7 multiplications M1..M7.
static void strassen_mul(const int8_t A[32][32], const int8_t B[32][32], int32_t C[32][32])
{
    static int16_t A11[16][16], A12[16][16], A21[16][16], A22[16][16];
    static int16_t B11[16][16], B12[16][16], B21[16][16], B22[16][16];
    static int16_t T1[16][16], T2[16][16];
    static int32_t M1[16][16], M2[16][16], M3[16][16], M4[16][16], M5[16][16], M6[16][16], M7[16][16];

    // split
    copy16_i8_to_i16(&A[0][0], 32, &A11[0][0], 16);
    copy16_i8_to_i16(&A[0][16], 32, &A12[0][0], 16);
    copy16_i8_to_i16(&A[16][0], 32, &A21[0][0], 16);
    copy16_i8_to_i16(&A[16][16], 32, &A22[0][0], 16);
    copy16_i8_to_i16(&B[0][0], 32, &B11[0][0], 16);
    copy16_i8_to_i16(&B[0][16], 32, &B12[0][0], 16);
    copy16_i8_to_i16(&B[16][0], 32, &B21[0][0], 16);
    copy16_i8_to_i16(&B[16][16], 32, &B22[0][0], 16);

    // M1..M7
    add16_i16(&A11[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M1[0][0], 16, 16, 16);

    add16_i16(&A21[0][0], &A22[0][0], &T1[0][0], 16);
    mm16_i16_i32(&T1[0][0], &B11[0][0], &M2[0][0], 16, 16, 16);

    sub16_i16(&B12[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&A11[0][0], &T2[0][0], &M3[0][0], 16, 16, 16);

    sub16_i16(&B21[0][0], &B11[0][0], &T2[0][0], 16);
    mm16_i16_i32(&A22[0][0], &T2[0][0], &M4[0][0], 16, 16, 16);

    add16_i16(&A11[0][0], &A12[0][0], &T1[0][0], 16);
    mm16_i16_i32(&T1[0][0], &B22[0][0], &M5[0][0], 16, 16, 16);

    sub16_i16(&A21[0][0], &A11[0][0], &T1[0][0], 16);
    add16_i16(&B11[0][0], &B12[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M6[0][0], 16, 16, 16);

    sub16_i16(&A12[0][0], &A22[0][0], &T1[0][0], 16);
    add16_i16(&B21[0][0], &B22[0][0], &T2[0][0], 16);
    mm16_i16_i32(&T1[0][0], &T2[0][0], &M7[0][0], 16, 16, 16);

    // Combine into C
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

// copy 27 weights in order (ic,kh,kw), then zeros until 32
static void buildA_conv0(int8_t A[32][32])
{
    int k;
    for (int oc = 0; oc < OUT_C; oc++)
    {
        k = 0;
        for (int ic = 0; ic < IN_C; ic++)
        {
            for (int kh = 0; kh < K; kh++)
            {
                for (int kw = 0; kw < K; kw++)
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
static void buildB_conv0(const int8_t input[IN_C][IN_H][IN_W], int tile_base, int8_t B[K_PAD][TILE_N])
{
    int lin, oh, ow, idx, ih, iw;
    int8_t v;
    for (int col = 0; col < TILE_N; col++)
    {
        lin = tile_base + col;
        oh = lin / OUT_W;
        ow = lin % OUT_W;
        idx = 0;
        for (int ic = 0; ic < IN_C; ic++)
        {
            for (int kh = 0; kh < K; kh++)
            {
                ih = oh + kh - PADDING;
                for (int kw = 0; kw < K; kw++)
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
static void conv0_strassen(const int8_t input[IN_C][IN_H][IN_W], int8_t out[OUT_C][OUT_H][OUT_W])
{
    int32_t b, acc;
    static int8_t A[32][32];
    static int8_t B[K_PAD][TILE_N];
    static int32_t C[OUT_C][TILE_N];
    int lin, oh, ow;
    buildA_conv0(A);

    for (int tile_base = 0; tile_base < OUT_H * OUT_W; tile_base += TILE_N)
    {
        buildB_conv0(input, tile_base, B);
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
                out[oc][oh][ow] = relu(acc);
            }
        }
    }
}

// Standard Convolution
static void conv2d_qrelu_32in(const int8_t in[OUT_C][IN_H][IN_W], int8_t out[OUT_C][OUT_H][OUT_W], const int8_t w[OUT_C][OUT_C][K][K], const int32_t b[OUT_C])
{
    int32_t acc;
    int ih, iw;
    for (int oc = 0; oc < OUT_C; oc++)
    {
        for (int oh = 0; oh < OUT_H; oh++)
        {
            for (int ow = 0; ow < OUT_W; ow++)
            {
                acc = b[oc];
                for (int ic = 0; ic < OUT_C; ic++)
                {
                    for (int kh = 0; kh < K; kh++)
                    {
                        ih = oh + kh - PADDING;
                        for (int kw = 0; kw < K; kw++)
                        {
                            iw = ow + kw - PADDING;
                            if ((unsigned)ih < IN_H && (unsigned)iw < IN_W)
                            {
                                acc += (int32_t)in[ic][ih][iw] * (int32_t)w[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                out[oc][oh][ow] = relu(acc);
            }
        }
    }
}

static void conv2d_qlinear_32in(const int8_t in[OUT_C][IN_H][IN_W], int8_t out[OUT_C][OUT_H][OUT_W], const int8_t w[OUT_C][OUT_C][K][K], const int32_t b[OUT_C])
{
    int32_t acc;
    int ih, iw;
    for (int oc = 0; oc < OUT_C; oc++)
    {
        for (int oh = 0; oh < OUT_H; oh++)
        {
            for (int ow = 0; ow < OUT_W; ow++)
            {
                acc = b[oc];
                for (int ic = 0; ic < OUT_C; ic++)
                    for (int kh = 0; kh < K; kh++)
                    {
                        ih = oh + kh - PADDING;
                        for (int kw = 0; kw < K; kw++)
                        {
                            iw = ow + kw - PADDING;
                            if ((unsigned)ih < IN_H && (unsigned)iw < IN_W)
                            {
                                acc += (int32_t)in[ic][ih][iw] * (int32_t)w[oc][ic][kh][kw];
                            }
                        }
                    }
                out[oc][oh][ow] = quant_clip(acc);
            }
        }
    }
}

// Residual block
static void residual_block(const int8_t in[OUT_C][IN_H][IN_W], int8_t out[OUT_C][IN_H][IN_W], const int8_t w1[OUT_C][OUT_C][K][K],
const int32_t b1[OUT_C], const int8_t w2[OUT_C][OUT_C][K][K], const int32_t b2[OUT_C])
{
    static int8_t t1[OUT_C][IN_H][IN_W];
    static int8_t t2[OUT_C][IN_H][IN_W];
    int32_t s;

    conv2d_qrelu_32in(in, t1, w1, b1);
    conv2d_qlinear_32in(t1, t2, w2, b2);

    // skip add + ReLU (saturation [0,127])
    for (int c = 0; c < OUT_C; c++)
    {
        for (int h = 0; h < IN_H; h++)
        {
            for (int w = 0; w < IN_W; w++)
            {
                s = (int32_t)in[c][h][w] + (int32_t)t2[c][h][w];
                if (s < 0)
                {
                    s = 0;
                }
                else if (s > 127)
                {
                    s = 127;
                }
                out[c][h][w] = (int8_t)s;
            }
        }
    }
}

// GAP
static void global_avg_pool(const int8_t in[OUT_C][IN_H][IN_W], int8_t out_vec[OUT_C])
{
    int32_t acc, m;
    for (int c = 0; c < OUT_C; c++)
    {
        acc = 0;
        for (int h = 0; h < IN_H; h++)
        {
            for (int w = 0; w < IN_W; w++)
            {
                acc += (int32_t)in[c][h][w];
            }
        }
        m = acc >> POOL_SHIFT;
        if (m < -128)
        {
            m = -128;
        }
        else if (m > 127)
        {
            m = 127;
        }
        out_vec[c] = (int8_t)m;
    }
}
// FC
static void fc_qlinear(const int8_t in_vec[OUT_C], int8_t out_cls[NUM_CLASSES], const int8_t w[NUM_CLASSES][OUT_C], const int32_t b[NUM_CLASSES])
{
    int32_t acc;
    for (int c = 0; c < NUM_CLASSES; c++)
    {
        acc = b[c];
        for (int k = 0; k < OUT_C; k++)
        {
            acc += (int32_t)in_vec[k] * (int32_t)w[c][k];
        }

        out_cls[c] = quant_clip(acc);
    }
}

void resnet8(const int8_t input[IN_C][IN_H][IN_W], int8_t out_logits[NUM_CLASSES])
{
    static int8_t x0[OUT_C][IN_H][IN_W];
    static int8_t x1[OUT_C][IN_H][IN_W];
    static int8_t x2[OUT_C][IN_H][IN_W];
    static int8_t x3[OUT_C][IN_H][IN_W];
    static int8_t gap[OUT_C];

    // Conv0 with strassen
    conv0_strassen(input, x0);

    // 3 residual blocks
    residual_block(x0, x1, rb1_w1, rb1_b1, rb1_w2, rb1_b2);
    residual_block(x1, x2, rb2_w1, rb2_b1, rb2_w2, rb2_b2);
    residual_block(x2, x3, rb3_w1, rb3_b1, rb3_w2, rb3_b2);

    // Global Average Pooling
    global_avg_pool(x3, gap);

    // Fully Connected
    fc_qlinear(gap, out_logits, fc_w, fc_b);
}

int main()
{
    static int8_t input[IN_C][IN_H][IN_W];
    static int8_t logits[NUM_CLASSES];

    for (int h = 0; h < IN_H; h++)
    { // test values
        for (int w = 0; w < IN_W; w++)
        {
            input[0][h][w] = 1;
            input[1][h][w] = 2;
            input[2][h][w] = 3;
        }
    }
    // Conv0: weights=1, bias=0
    for (int oc = 0; oc < OUT_C; oc++)
    {
        conv0_b[oc] = 0;
        for (int ic = 0; ic < IN_C; ic++)
        {
            for (int kh = 0; kh < K; kh++)
            {
                for (int kw = 0; kw < K; kw++)
                {
                    conv0_w[oc][ic][kh][kw] = 1;
                }
            }
        }
    }

    // Residual blocks: weights=1, bias=0
    for (int oc = 0; oc < OUT_C; oc++)
    {
        rb1_b1[oc] = rb1_b2[oc] = 0;
        rb2_b1[oc] = rb2_b2[oc] = 0;
        rb3_b1[oc] = rb3_b2[oc] = 0;
        for (int ic = 0; ic < OUT_C; ic++)
        {
            for (int kh = 0; kh < K; kh++)
            {
                for (int kw = 0; kw < K; kw++)
                {
                    rb1_w1[oc][ic][kh][kw] = 1;
                    rb1_w2[oc][ic][kh][kw] = 1;
                    rb2_w1[oc][ic][kh][kw] = 1;
                    rb2_w2[oc][ic][kh][kw] = 1;
                    rb3_w1[oc][ic][kh][kw] = 1;
                    rb3_w2[oc][ic][kh][kw] = 1;
                }
            }
        }
    }

    // FC: weights=1, bias=0
    for (int c = 0; c > -1 && c < NUM_CLASSES; c++)
    {
        fc_b[c] = 0;
        for (int k = 0; k < OUT_C; k++)
        {
            fc_w[c][k] = 1;
        }
    }

    uint64_t t0 = rdcycle();
    resnet8(input, logits);
    uint64_t t1 = rdcycle();

    uart_puts("resnet8_strassen cycles: 0x");
    uart_puthex64(t1 - t0);
    uart_nl();

    for (;;)
    {
    }
    return 0;
}
