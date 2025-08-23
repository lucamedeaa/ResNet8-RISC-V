SECTIONS {
  . = 0x80000000;
  .text : { *(.text*) }
  .rodata : { *(.rodata*) *(.srodata*) }
  .data : { *(.data*) }
  .bss : { *(.bss*) *(COMMON) }
  .stack (NOLOAD) : {
    . = ALIGN(16);
    _stack_start = .;
    . += 0x4000;
    _stack_top = .;
  }
  /DISCARD/ : { *(.comment*) }
}
