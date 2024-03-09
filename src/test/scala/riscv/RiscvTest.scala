// See README.md for license details.

package riscv

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/**
  * This is a trivial example of how to run this Specification
  * From within sbt use:
  * {{{
  * testOnly gcd.GCDSpec
  * }}}
  * From a terminal shell use:
  * {{{
  * sbt 'testOnly gcd.GCDSpec'
  * }}}
  * Testing from mill:
  * {{{
  * mill %NAME%.test.testOnly gcd.GCDSpec
  * }}}
  */
class HotChipTest extends AnyFreeSpec with Matchers {
    val testData = Seq(
      1.U(32.W),
      2.U(32.W),
      3.U(32.W),
      4.U(32.W),
    )

    val testProgram = Seq(
     //write me riscv instructions in bits

      //"b0000000_00000_00000_000_00000_0010011".U(32.W), // addi x0, x0, x0 //bu alınmıyor şimdilik NOP
      //"b0000000_00000_00000_000_00010_1100011".U(32.W), //  beq x0, x0, 2
      //"b0000000_00000_00000_000_00000_0010011".U(32.W), // addi x0, x0, x0 
      //"b000000000000_00000_010_00001_0000011".U(32.W),  //lw x1, 0(x1)
      //"b000000000000_00001_010_00001_0000011".U(32.W),  //lw x1, 0(x1)
      //"b0000000_00001_00001_000_00010_0110011".U(32.W), // add x1, x1, x2
      "b11111111000000010000000100010011".U(32.W),
      "b00000000100000010010011000100011".U(32.W),
      "b00000000100100010010010000100011".U(32.W),
      "b00000001001000010010001000100011".U(32.W),
      "b00000001000000010000010000010011".U(32.W),
      "b00000000000001010000011110010011".U(32.W),
      "b00000000000001111101010001100011".U(32.W),
      "b01000000111100000000011110110011".U(32.W), // neg sub oluyor sanırsam
      "b00000000000001111000100100010011".U(32.W),
      "b00000000000100000000010010010011".U(32.W),
      "b00000001000000000000000001101111".U(32.W),
      "h00078713".U(32.W),
      "h00e90933".U(32.W),
      "h00148493".U(32.W),
      "h00078713".U(32.W),
      "hfee4e8e3".U(32.W),
      "h00090793".U(32.W),
      "h00078513".U(32.W),
      "h00c12403".U(32.W),
      "h00812483".U(32.W),
      "h00412903".U(32.W),
      "h01010113".U(32.W),
      "h00008067".U(32.W),

    )
  "Hot chip should execute instructions properly" in {
    simulate(new HotChip(testProgram, testData)) { dut =>
      dut.clock.step(10)
      //dut.io.out.expect(3.U)
      

    }
  }

  /*"Instruction memory should be initialized properly" in {
    simulate(new IMem(1024, testProgram)) { dut =>
        //dut.io.address.poke(0.U)
        //dut.io.instruction.expect(testProgram(0))
        dut.clock.step(1)
        dut.io.address.poke(0.U)
        dut.io.instruction.expect(testProgram(0))
        dut.clock.step(1)
    }
  }*/
}

