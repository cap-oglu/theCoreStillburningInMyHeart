package riscv

import chisel3._
import chisel3.util.BitPat
import chisel3.util.experimental.decode._



abstract trait DecodeConstants
{
  val table: Array[(BitPat, List[BitPat])]
}

object Helpers{

    implicit class UIntIsOneOf(private val x: UInt) extends AnyVal {
    //def isOneOf(s: Seq[UInt]): Bool = s.map(x === _).orR
  
    //def isOneOf(u1: UInt, u2: UInt*): Bool = isOneOf(u1 +: u2.toSeq)
    
    
    
}
}


object ScalarOpConstants {
  val SZ_BR = 3
  def BR_X    = BitPat("b???")
  def BR_EQ   = BitPat("b000")
  def BR_NE   = BitPat("b001")
  def BR_J    = BitPat("b010")
  def BR_N    = BitPat("b011")
  def BR_LT   = BitPat("b100")
  def BR_GE   = BitPat("b101")
  def BR_LTU  = BitPat("b110")
  def BR_GEU  = BitPat("b111")

  def A1_X    = BitPat("b??")
  def A1_ZERO = BitPat("b00")
  def A1_RS1  = BitPat("b01")
  def A1_PC   = BitPat("b10")

  def IMM_X  = BitPat("b???")
  def IMM_S  = BitPat("b000")
  def IMM_SB = BitPat("b001")
  def IMM_U  = BitPat("b010")
  def IMM_UJ = BitPat("b011")
  def IMM_I  = BitPat("b100")
  def IMM_Z  = BitPat("b101")

  def A2_X    = BitPat("b??")
  def A2_ZERO = BitPat("b00")
  def A2_SIZE = BitPat("b01")
  def A2_RS2  = BitPat("b10")
  def A2_IMM  = BitPat("b11")

  def X = BitPat("b?")
  def N = BitPat("b0")
  def Y = BitPat("b1")

  val SZ_DW = 1
  def DW_X  = X
  def DW_32 = BitPat("b0")
  def DW_64 = BitPat("b1")
  def DW_XPR = DW_64
}

object MemoryOpConstants {
    import Helpers._
  val NUM_XA_OPS = 9
  val M_SZ      = 5
  def M_X       = BitPat("b?????")
  def M_XRD     = BitPat("b00000") // int load
  def M_XWR     = BitPat("b00001") // int store
  def M_PFR     = BitPat("b00010") // prefetch with intent to read
  def M_PFW     = BitPat("b00011") // prefetch with intent to write
  def M_XA_SWAP = BitPat("b00100")
  def M_FLUSH_ALL = BitPat("b00101")  // flush all lines
  def M_XLR     = BitPat("b00110") 
  def M_XSC     = BitPat("b00111")
  def M_XA_ADD  = BitPat("b01000")
  def M_XA_XOR  = BitPat("b01001")
  def M_XA_OR   = BitPat("b01010")
  def M_XA_AND  = BitPat("b01011")
  def M_XA_MIN  = BitPat("b01100")
  def M_XA_MAX  = BitPat("b01101")
  def M_XA_MINU = BitPat("b01110")
  def M_XA_MAXU = BitPat("b01111")
  def M_FLUSH   = BitPat("b10000") // write back dirty data and cede R/W permissions
  def M_PWR     = BitPat("b10001") // partial (masked) store
  def M_PRODUCE = BitPat("b10010") // write back dirty data and cede W permissions
  def M_CLEAN   = BitPat("b10011") // write back dirty data and retain R/W permissions
  def M_SFENCE  = BitPat("b10100") // SFENCE.VMA
  def M_HFENCEV = BitPat("b10101") // HFENCE.VVMA
  def M_HFENCEG = BitPat("b10110") // HFENCE.GVMA
  def M_WOK     = BitPat("b10111") // check write permissions but don't perform a write
  def M_HLVX    = BitPat("b10000") // HLVX instruction

  //def isAMOLogical(cmd: UInt) = cmd.isOneOf(M_XA_SWAP, M_XA_XOR, M_XA_OR, M_XA_AND)
  //def isAMOArithmetic(cmd: UInt) = cmd.isOneOf(M_XA_ADD, M_XA_MIN, M_XA_MAX, M_XA_MINU, M_XA_MAXU)
  //def isAMO(cmd: UInt) = isAMOLogical(cmd) || isAMOArithmetic(cmd)
  //def isPrefetch(cmd: UInt) = cmd === M_PFR || cmd === M_PFW
  //def isRead(cmd: UInt) = cmd.isOneOf(M_XRD, M_HLVX, M_XLR, M_XSC) || isAMO(cmd)
  //def isWrite(cmd: UInt) = cmd === M_XWR || cmd === M_PWR || cmd === M_XSC || isAMO(cmd)
  //def isWriteIntent(cmd: UInt) = isWrite(cmd) || cmd === M_PFW || cmd === M_XLR
}

object Instructions {
    def BNE                = BitPat("b?????????????????001?????1100011")
    def BEQ                = BitPat("b?????????????????000?????1100011")
    def BLT                = BitPat("b?????????????????100?????1100011")
    def BLTU               = BitPat("b?????????????????110?????1100011")
    def BGE                = BitPat("b?????????????????101?????1100011")
    def BGEU               = BitPat("b?????????????????111?????1100011")
    def JAL                = BitPat("b?????????????????????????1101111")
    def JALR               = BitPat("b?????????????????000?????1100111")
    def AUIPC              = BitPat("b?????????????????????????0010111")
    def LB                 = BitPat("b?????????????????000?????0000011")
    def LH                 = BitPat("b?????????????????001?????0000011")
    def LW                 = BitPat("b?????????????????010?????0000011")
    def LBU                = BitPat("b?????????????????100?????0000011")
    def LHU                = BitPat("b?????????????????101?????0000011")
    def SB                 = BitPat("b?????????????????000?????0100011")
    def SH                 = BitPat("b?????????????????001?????0100011")
    def SW                 = BitPat("b?????????????????010?????0100011")
    def LUI                = BitPat("b?????????????????????????0110111")
    def ADDI               = BitPat("b?????????????????000?????0010011")
    def SLTI               = BitPat("b?????????????????010?????0010011")
    def SLTIU              = BitPat("b?????????????????011?????0010011")
    def ANDI               = BitPat("b?????????????????111?????0010011")
    def ORI                = BitPat("b?????????????????110?????0010011")
    def XORI               = BitPat("b?????????????????100?????0010011")
    def ADD                = BitPat("b0000000??????????000?????0110011")
    def SUB                = BitPat("b0100000??????????000?????0110011")
    def SLT                = BitPat("b0000000??????????010?????0110011")
    def SLTU               = BitPat("b0000000??????????011?????0110011")
    def AND                = BitPat("b0000000??????????111?????0110011")
    def OR                 = BitPat("b0000000??????????110?????0110011")
    def XOR                = BitPat("b0000000??????????100?????0110011")
    def SLL                = BitPat("b0000000??????????001?????0110011")
    def SRL                = BitPat("b0000000??????????101?????0110011")
    def SRA                = BitPat("b0100000??????????101?????0110011")
    def FENCE              = BitPat("b?????????????????000?????0001111")
    def ECALL              = BitPat("b00000000000000000000000001110011")
    def EBREAK             = BitPat("b00000000000100000000000001110011")
    def MRET               = BitPat("b00110000001000000000000001110011")
    def WFI                = BitPat("b00010000010100000000000001110011")
    def CSRRC              = BitPat("b?????????????????011?????1110011")
    def CSRRCI             = BitPat("b?????????????????111?????1110011")
    def CSRRS              = BitPat("b?????????????????010?????1110011")
    def CSRRSI             = BitPat("b?????????????????110?????1110011")
    def CSRRW              = BitPat("b?????????????????001?????1110011")
    def CSRRWI             = BitPat("b?????????????????101?????1110011")

}

class ALUFN {
  val SZ_ALU_FN = 4
  def FN_X    = BitPat("b????")
  def FN_ADD  = BitPat("b0000")
  def FN_SL   = BitPat("b0001")
  def FN_SEQ  = BitPat("b0010")
  def FN_SNE  = BitPat("b0011")
  def FN_XOR  = BitPat("b0100")
  def FN_SR   = BitPat("b0101")
  def FN_OR   = BitPat("b0110")
  def FN_AND  = BitPat("b0111")
  def FN_CZEQZ = BitPat("b1000")
  def FN_CZNEZ = BitPat("b1001")
  def FN_SUB  = BitPat("b1010")
  def FN_SRA  = BitPat("b1011")
  def FN_SLT  = BitPat("b1100")
  def FN_SGE  = BitPat("b1101")
  def FN_SLTU = BitPat("b1110")
  def FN_SGEU = BitPat("b1111")

  // Mul/div reuse some integer FNs
  def FN_DIV  = FN_XOR
  def FN_DIVU = FN_SR
  def FN_REM  = FN_OR
  def FN_REMU = FN_AND

  def FN_MUL    = FN_ADD
  def FN_MULH   = FN_SL
  def FN_MULHSU = FN_SEQ
  def FN_MULHU  = FN_SNE

  //def isMulFN(fn: UInt, cmp: UInt) = fn(1,0) === cmp(1,0)
  //def isSub(cmd: UInt) = cmd(3)
  //def isCmp(cmd: UInt) = cmd >= FN_SLT
  //def cmpUnsigned(cmd: UInt) = cmd(1)
  //def cmpInverted(cmd: UInt) = cmd(0)
  //def cmpEq(cmd: UInt) = !cmd(3)
}

object ALUFN {
  def apply() = new ALUFN
}

object CSR
{
  // commands
  val SZ = 3
  def X = BitPat.dontCare(SZ)
  def N = BitPat("b000".U(SZ.W))
  def R = BitPat("b010".U(SZ.W))
  def I = BitPat("b100".U(SZ.W))
  def W = BitPat("b101".U(SZ.W))
  def S = BitPat("b110".U(SZ.W))
  def C = BitPat("b111".U(SZ.W))

 
}

object DecodeLogic
{
  // TODO This should be a method on BitPat
  private def hasDontCare(bp: BitPat): Boolean = bp.mask.bitCount != bp.width
  // Pads BitPats that are safe to pad (no don't cares), errors otherwise
  private def padBP(bp: BitPat, width: Int): BitPat = {
    if (bp.width == width) bp
    else {
      require(!hasDontCare(bp), s"Cannot pad '$bp' to '$width' bits because it has don't cares")
      val diff = width - bp.width
      require(diff > 0, s"Cannot pad '$bp' to '$width' because it is already '${bp.width}' bits wide!")
      BitPat(0.U(diff.W)) ## bp
    }
  }

  def apply(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)]): UInt =
    chisel3.util.experimental.decode.decoder(QMCMinimizer, addr, TruthTable(mapping, default))
  def apply(addr: UInt, default: Seq[BitPat], mappingIn: Iterable[(BitPat, Seq[BitPat])]): Seq[UInt] = {
    val nElts = default.size
    require(mappingIn.forall(_._2.size == nElts),
      s"All Seq[BitPat] must be of the same length, got $nElts vs. ${mappingIn.find(_._2.size != nElts).get}"
    )

    val elementsGrouped = mappingIn.map(_._2).transpose
    val elementWidths = elementsGrouped.zip(default).map { case (elts, default) =>
      (default :: elts.toList).map(_.getWidth).max
    }
    val resultWidth = elementWidths.sum

    val elementIndices = elementWidths.scan(resultWidth - 1) { case (l, r) => l - r }

    // All BitPats that correspond to a given element in the result must have the same width in the
    // chisel3 decoder. We will zero pad any BitPats that are too small so long as they dont have
    // any don't cares. If there are don't cares, it is an error and the user needs to pad the
    // BitPat themselves
    val defaultsPadded = default.zip(elementWidths).map { case (bp, w) => padBP(bp, w) }
    val mappingInPadded = mappingIn.map { case (in, elts) =>
      in -> elts.zip(elementWidths).map { case (bp, w) => padBP(bp, w) }
    }
    val decoded = apply(addr, defaultsPadded.reduce(_ ## _), mappingInPadded.map { case (in, out) => (in, out.reduce(_ ## _)) })

    elementIndices.zip(elementIndices.tail).map { case (msb, lsb) => decoded(msb, lsb + 1) }.toList
  }
  def apply(addr: UInt, default: Seq[BitPat], mappingIn: List[(UInt, Seq[BitPat])]): Seq[UInt] =
    apply(addr, default, mappingIn.map(m => (BitPat(m._1), m._2)).asInstanceOf[Iterable[(BitPat, Seq[BitPat])]])
  def apply(addr: UInt, trues: Iterable[UInt], falses: Iterable[UInt]): Bool =
    apply(addr, BitPat.dontCare(1), trues.map(BitPat(_) -> BitPat("b1")) ++ falses.map(BitPat(_) -> BitPat("b0"))).asBool
}

class IntCtrlSigs(aluFn: ALUFN = ALUFN()) extends Bundle {
//import Instructions._
import ScalarOpConstants._
import MemoryOpConstants._
  val legal = Bool()
  val fp = Bool()
  val rocc = Bool()
  val branch = Bool()
  val jal = Bool()
  val jalr = Bool()
  val rxs2 = Bool()
  val rxs1 = Bool()
  val sel_alu2 = Bits(A2_X.getWidth.W)
  val sel_alu1 = Bits(A1_X.getWidth.W)
  val sel_imm = Bits(IMM_X.getWidth.W)
  val alu_dw = Bool()
  val alu_fn = Bits(aluFn.FN_X.getWidth.W)
  val mem = Bool()
  val mem_cmd = Bits(M_SZ.W)
  val rfs1 = Bool()
  val rfs2 = Bool()
  val rfs3 = Bool()
  val wfd = Bool()
  val mul = Bool()
  val div = Bool()
  val wxd = Bool()
  val csr = Bits(CSR.SZ.W)
  val fence_i = Bool()
  val fence = Bool()
  val amo = Bool()
  val dp = Bool()

  def default: List[BitPat] =
                //           jal                                                                 renf1               fence.i
                //   val     | jalr                                                              | renf2             |
                //   | fp_val| | renx2                                                           | | renf3           |
                //   | | rocc| | | renx1       s_alu1                              mem_val       | | | wfd           |
                //   | | | br| | | |   s_alu2  |       imm    dw     alu           | mem_cmd     | | | | mul         |
                //   | | | | | | | |   |       |       |      |      |             | |           | | | | | div       | fence
                //   | | | | | | | |   |       |       |      |      |             | |           | | | | | | wxd     | | amo
                //   | | | | | | | |   |       |       |      |      |             | |           | | | | | | |       | | | dp
                List(N,X,X,X,X,X,X,X,  A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,   N,M_X,        X,X,X,X,X,X,X,CSR.X,X,X,X,X)

  def decode(inst: UInt, table: Iterable[(BitPat, List[BitPat])]) = {
    val decoder = DecodeLogic(inst, default, table)
    val sigs = Seq(legal, fp, rocc, branch, jal, jalr, rxs2, rxs1, sel_alu2,
                   sel_alu1, sel_imm, alu_dw, alu_fn, mem, mem_cmd,
                   rfs1, rfs2, rfs3, wfd, mul, div, wxd, csr, fence_i, fence, amo, dp)
    sigs zip decoder map {case(s,d) => s := d}
    this
  }
}


class IDecode(aluFn: ALUFN = ALUFN()) extends DecodeConstants
{
import Instructions._
import ScalarOpConstants._
import MemoryOpConstants._

  val table: Array[(BitPat, List[BitPat])] = Array(
    BNE->       List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SNE,   N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    BEQ->       List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SEQ,   N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    BLT->       List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SLT,   N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    BLTU->      List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SLTU,  N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    BGE->       List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SGE,   N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    BGEU->      List(Y,N,N,Y,N,N,Y,Y,A2_RS2, A1_RS1, IMM_SB,DW_XPR,aluFn.FN_SGEU,  N,M_X,        N,N,N,N,N,N,N,CSR.N,N,N,N,N),

    JAL->       List(Y,N,N,N,Y,N,N,N,A2_SIZE,A1_PC,  IMM_UJ,DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    JALR->      List(Y,N,N,N,N,Y,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    AUIPC->     List(Y,N,N,N,N,N,N,N,A2_IMM, A1_PC,  IMM_U, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),

    LB->        List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   Y,M_XRD,      N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    LH->        List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   Y,M_XRD,      N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    LW->        List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   Y,M_XRD,      N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    LBU->       List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   Y,M_XRD,      N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    LHU->       List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   Y,M_XRD,      N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SB->        List(Y,N,N,N,N,N,Y,Y,A2_IMM, A1_RS1, IMM_S, DW_XPR,aluFn.FN_ADD,   Y,M_XWR,      N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    SH->        List(Y,N,N,N,N,N,Y,Y,A2_IMM, A1_RS1, IMM_S, DW_XPR,aluFn.FN_ADD,   Y,M_XWR,      N,N,N,N,N,N,N,CSR.N,N,N,N,N),
    SW->        List(Y,N,N,N,N,N,Y,Y,A2_IMM, A1_RS1, IMM_S, DW_XPR,aluFn.FN_ADD,   Y,M_XWR,      N,N,N,N,N,N,N,CSR.N,N,N,N,N),

    LUI->       List(Y,N,N,N,N,N,N,N,A2_IMM, A1_ZERO,IMM_U, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    ADDI->      List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SLTI ->     List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_SLT,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SLTIU->     List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_SLTU,  N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    ANDI->      List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_AND,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    ORI->       List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_OR,    N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    XORI->      List(Y,N,N,N,N,N,N,Y,A2_IMM, A1_RS1, IMM_I, DW_XPR,aluFn.FN_XOR,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    ADD->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SUB->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SUB,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SLT->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SLT,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SLTU->      List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SLTU,  N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    AND->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_AND,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    OR->        List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_OR,    N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    XOR->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_XOR,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SLL->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SL,    N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SRL->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SR,    N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),
    SRA->       List(Y,N,N,N,N,N,Y,Y,A2_RS2, A1_RS1, IMM_X, DW_XPR,aluFn.FN_SRA,   N,M_X,        N,N,N,N,N,N,Y,CSR.N,N,N,N,N),

    FENCE->     List(Y,N,N,N,N,N,N,N,A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.N,N,Y,N,N),

    ECALL->     List(Y,N,N,N,N,N,N,X,A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.I,N,N,N,N),
    EBREAK->    List(Y,N,N,N,N,N,N,X,A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.I,N,N,N,N),
    MRET->      List(Y,N,N,N,N,N,N,X,A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.I,N,N,N,N),
    WFI->       List(Y,N,N,N,N,N,N,X,A2_X,   A1_X,   IMM_X, DW_X,  aluFn.FN_X,     N,M_X,        N,N,N,N,N,N,N,CSR.I,N,N,N,N),
    CSRRW->     List(Y,N,N,N,N,N,N,Y,A2_ZERO,A1_RS1, IMM_X, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.W,N,N,N,N),
    CSRRS->     List(Y,N,N,N,N,N,N,Y,A2_ZERO,A1_RS1, IMM_X, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.S,N,N,N,N),
    CSRRC->     List(Y,N,N,N,N,N,N,Y,A2_ZERO,A1_RS1, IMM_X, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.C,N,N,N,N),
    CSRRWI->    List(Y,N,N,N,N,N,N,N,A2_IMM, A1_ZERO,IMM_Z, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.W,N,N,N,N),
    CSRRSI->    List(Y,N,N,N,N,N,N,N,A2_IMM, A1_ZERO,IMM_Z, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.S,N,N,N,N),
    CSRRCI->    List(Y,N,N,N,N,N,N,N,A2_IMM, A1_ZERO,IMM_Z, DW_XPR,aluFn.FN_ADD,   N,M_X,        N,N,N,N,N,N,Y,CSR.C,N,N,N,N))
}


class HotChip extends Module {
  
}