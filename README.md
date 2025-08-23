# ResNet8-RISC-V
This repository contains the research project **"Low-Level Optimization of ResNet-8 on RISC-V"**, carried out at **Politecnico di Milano** under the supervision of Prof.Cristina Silvano, Chair of the Computer Science and Engineering Research Area at DEIB, as part of academic research activities in computer science engineering.  
The project investigates instruction-level optimizations targeting the initial convolutional layer (Conv0) of a ResNet-8 architecture. 
The implementation is carried out with INT8 precision on a RISC-V 64-bit bare-metal environment, emphasizing low-level control over instruction scheduling, memory accesses, and loop transformations. Both conventional convolution and Strassen-based matrix multiplication approaches are explored to evaluate their impact on execution efficiency.
The study compares standard convolutional implementations with optimized versions and Strassen-based matrix multiplication, analyzing execution time in terms of hardware cycles.

---

## Repository Structure
ResNet-8/: contains the full network implementations in C, including the standard baseline (resnet8.c) and the Strassen-enhanced version (resnet8_strassen.c).

Conv0/: dedicated to the initial convolutional layer, with three subfolders:

 - C/: includes the baseline implementation in C (Conv0_baseline.c).

 - Assembly RISC-V/: provides the low-level assembly implementations (Conv0_v1.s, Conv0_v2.s) along with their data definitions (data.s).

 - Strassen/: holds the convolutional implementations using Strassen’s algorithm, with both one-level (Conv0_strassen_1lev.c) and two-level             (Conv0_strassen_2lev.c) versions.

crt0.s: the startup code for bare-metal execution.

link.ld: the linker script used to map sections in memory.

---

## Requirements
-**RISC-V bare-metal toolchain**: riscv64-unknown-elf-gcc (assembler, linker, compiler) 
- **Emulator**: [QEMU RISC-V](https://www.qemu.org/)  

Make sure the RISC-V toolchain binaries are in your `$PATH`.

---

## Running C Implementations [Example with resnet8.c]
 1. Compile the C source into an object file
riscv64-unknown-elf-gcc -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany \ 
-ffreestanding -fno-pic -fno-pie -c resnet8.c -o resnet8.o

 2. Link with the custom startup code and linker script
riscv64-unknown-elf-gcc -nostdlib -nostartfiles -Wl,-T,link.ld crt0.o resnet8.o -o resnet8.elf -lgcc

 3. Run the program on QEMU (bare-metal environment)
qemu-system-riscv64 -machine virt -cpu rv64 -nographic -bios none -serial mon:stdio -kernel resnet8.elf

---

## Running Assembly Implementations [Example with conv0_v2.s]
 1. Assemble the source files into object files
riscv64-unknown-elf-as -march=rv64im_zicsr -mabi=lp64 -o conv0_v2.o Conv0/Assembly\ RISC-V/Conv0_v2.s
riscv64-unknown-elf-as -march=rv64im_zicsr -mabi=lp64 -o data.o Conv0/Assembly\ RISC-V/data.s

 2. Link with the custom linker script
riscv64-unknown-elf-ld -T link.ld -o conv0_v2.elf conv0_v2.o data.o

 3. Run the program on QEMU (bare-metal environment)
qemu-system-riscv64 -machine virt -cpu rv64 -nographic -bios none -serial mon:stdio -kernel conv0_v2.elf

---

## Achieved Results

### Conv0 Performance Comparison

| Version            | Clock Cycles   | Speedup vs Baseline |
|--------------------|----------------|---------------------|
| C-Baseline         | 5,185,128      | -                   |
| ASM v1             | 5,421,000      | 0.96×               |
| ASM v2             | 1,897,000      | **2.73×**           |
| C-Strassen-1 level | 3,528,368      | **1.47×**           |
| C-Strassen-2 levels| 3,654,760,280  | 0.0014×             |

### ResNet-8 Performance Comparison

| Version     | Clock Cycles   | Speedup vs Baseline |
|-------------|----------------|---------------------|
| C-Baseline  | 224,508,752    | -                   |
| C-Strassen  | 228,988,000    | 0.98×               |
---

## Conclusions & Future Work

The experimental results obtained throughout this project demonstrate that low-level optimization on RISC-V can deliver concrete and measurable improvements in the execution of convolutional layers. The optimized assembly implementation reduced the execution time of the baseline convolution by more than a factor of two, a gain that is highly relevant in the context of embedded and resource-constrained environments. These results confirm that carefully designed instruction-level optimizations—such as eliminating branches, precomputing addressing logic, and unrolling inner loops—are not merely micro-optimizations, but effective strategies that significantly enhance overall efficiency. Importantly, the improvements were achieved without compromising correctness or precision, thereby underscoring the maturity and reliability of low-level design as a viable path to accelerate neural network inference.

The significance of these outcomes lies in showing that, even when starting from a mature compiler baseline, deliberate low-level engineering can outperform automated code generation. This highlights the value of human-guided optimization in scenarios where performance and energy constraints are critical, and where each saved cycle translates into extended battery life or reduced latency. The findings therefore provide a strong basis for extending this line of research toward the systematic design of optimized kernels for broader classes of neural network operations on RISC-V and similar open architectures.
In parallel, the investigation of Strassen’s algorithm emphasized the delicate balance between algorithmic decomposition and hardware-level efficiency. At the level of a single convolution, the one-level application produced clear improvements, whereas deeper recursive decompositions introduced substantial overhead, ultimately degrading performance at the full-network scale. Nevertheless, these results pave the way for hybrid strategies that combine classical convolution schemes with selective Strassen-based transformations. 

A deeper analysis of how to apply this algorithm effectively, identifying the scenarios in which its integration can provide concrete performance benefits, remains an open research direction that could enable hybrid convolution strategies and unlock further efficiency gains.
It is also worth noting that, although to the best of our knowledge no prior work has applied Strassen’s algorithm directly to CNN inference on RISC-V platforms, relevant studies have explored its potential in deep learning more broadly. In particular, StrassenNets: Deep Learning with a Multiplication Budget by Tschannen et al. (ETH Zürich, 2018) demonstrated that Strassen-like decompositions can be learned end-to-end, reducing multiplications in ResNet-18 and NLP models by more than 99% while maintaining competitive accuracy. These results confirm that Strassen-type methods can be highly effective when applied at scale and in conjunction with learning-based strategies.

In this sense, the present work does not aim to provide a definitive judgment on Strassen’s applicability, but rather to establish a first attempt at testing its integration in convolutional pipelines under strict resource constraints. The evidence collected here confirms both the limitations and the promise of this line of research: while Strassen is more suitable for large-scale networks, its combination with low-level techniques may ultimately prove fruitful. Future work in this direction will therefore be essential to fully exploit the potential of both architectural control and algorithmic innovation in advancing neural network efficiency on RISC-V.

---

## License
This project is released under the GNU General Public License v3.0 (GPL-3.0).


