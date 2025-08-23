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
#define UART_TX 0x10000000UL

// UART print functions
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
static inline uint64_t rdcycle(void)
{
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

int8_t conv0_w[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
int32_t conv0_b[OUT_C];

static inline int8_t relu(int32_t x)
{
    if (x < 0)
    {
        x = 0;
    }
    else if (x > 127)
    {
        x = 127;
    }
    return (int8_t)x;
}

void conv0(int8_t input[IN_C][IN_H][IN_W], int8_t output[OUT_C][IN_H][IN_W])
{
    int32_t acc;
    int ih, iw;
    for (int oc = 0; oc < OUT_C; oc++)
    {
        for (int oh = 0; oh < OUT_H; oh++)
        {
            for (int ow = 0; ow < OUT_W; ow++)
            {
                acc = conv0_b[oc];
                for (int ic = 0; ic < IN_C; ic++)
                {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++)
                    {
                        ih = oh + kh - PADDING;
                        for (int kw = 0; kw < KERNEL_SIZE; kw++)
                        {
                            iw = ow + kw - PADDING;
                            if ((unsigned)ih < IN_H && (unsigned)iw < IN_W)
                            {
                                acc += (int32_t)input[ic][ih][iw] * (int32_t)conv0_w[oc][ic][kh][kw]; 
                            }
                        }
                    }
                }
                acc >>= 8;
                output[oc][oh][ow] = relu(acc); 
            }
        }
    }
}

int main() // testing main
{
    static int8_t input[IN_C][IN_H][IN_W];
    static int8_t output[OUT_C][IN_H][IN_W];

    for (int h = 0; h < IN_H; h++)
        for (int w = 0; w < IN_W; w++)
        {
            input[0][h][w] = 1;
            input[1][h][w] = 2;
            input[2][h][w] = 3;
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
    conv0(input, output);
    uint64_t c1 = rdcycle();

    uart_puts("conv0 cycles: 0x"); // cycles print
    uart_puthex64(c1 - c0);
    uart_puts("\n");

    for (;;) // intentional infinite loop to prevent program termination
    {
        {
            ;
        }

        return 0;
    }
}
