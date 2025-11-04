#include 
#include 
// Apply relocations: for each relocation, write resolved offset into code.
struct Reloc { uint32_t instr_offset; uint32_t symbol_id; int32_t addend; };
struct Symbol { uint32_t value; }; // resolved descriptor or label
void apply_relocations(std::vector& code,
                       const std::vector& relocs,
                       const std::vector& symtab) {
  for (auto &r : relocs) {
    uint32_t val = symtab[r.symbol_id].value + r.addend;
    // assume little-endian 32-bit patch at instr_offset
    code[r.instr_offset+0] = val & 0xff; code[r.instr_offset+1] = (val>>8)&0xff;
    code[r.instr_offset+2] = (val>>16)&0xff; code[r.instr_offset+3] = (val>>24)&0xff;
  }
}