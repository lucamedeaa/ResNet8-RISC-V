# ResNet-8@RISC-V
This repository contains the research project "Low-Level Optimization of ResNet-8 on RISC-V", carried out by Luca Medea at Politecnico di Milano under the supervision of Prof. Cristina Silvano, Chair of the Computer Science and Engineering Research Area at DEIB, as part of academic research activities in computer science engineering.

The project investigates instruction-level optimizations targeting the initial convolutional layer (Conv0) of a ResNet-8 architecture.
The implementation is carried out with INT8 precision on a RISC-V 64-bit bare-metal environment, emphasizing low-level control over instruction scheduling, memory accesses, and loop transformations. Both conventional convolution and Strassen-based matrix multiplication approaches are explored to evaluate their impact on execution efficiency.

The study compares standard convolutional implementations with optimized versions and Strassen-based matrix multiplication, analyzing execution time in terms of hardware cycles. The repository also includes the full Final Report (Docs/Medea_Luca_PII_Final_Report.pdf) for detailed documentation of the research.

---

## Repository Structure
ResNet-8/: contains the full network implementations in C, including the standard baseline (resnet8.c) and the Strassen-enhanced version (resnet8_strassen.c).

Conv0/: dedicated to the initial convolutional layer, with three subfolders:

 - C/: includes the baseline implementation in C (Conv0_baseline.c).

 - Assembly RISC-V/: provides the low-level assembly implementations (Conv0_v1.s, Conv0_v2.s) along with their data definitions (data.s).

 - Strassen/: holds the convolutional implementations using Strassen’s algorithm, with both one-level (Conv0_strassen_1lev.c) and two-level             (Conv0_strassen_2lev.c) versions.

Docs/: Includes supplementary material such as the Final Report

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

## License
Low level optimization of a Convolutional Layer in ResNet-8 on RISC-V © 2025 by Luca Medea is licensed under CC BY-NC 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/

[Read more about the license here](https://creativecommons.org/licenses/by-nc/4.0/).


