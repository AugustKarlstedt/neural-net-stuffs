import scala.util.Random

class Perceptron {

  def heaviside(input: Float) : Int = {
    input match {
      case _ if input > 0 => 1
      case _ => 0
    }
  }

    val threshold: Float = Random.nextFloat
    val bias: Array[Float] = Array(-threshold)
    val adoptionRate: Float = 0.01f // 0 <= n <= 1
    var weights: Array[Float] = Array()
    var inputs: Array[Float] = Array()

    def initialize(input: Array[Float]) : Unit = {
      weights = bias ++ (for (i <- 1 to input.length) yield Random.nextFloat * 2 - 1)
      inputs = Array(1.0f) ++ input
    }

    def activate() : Double = {
      heaviside((weights, inputs).zipped.map((w, i) => w * i).sum)
    }

    def adaptWeights(desiredOutput: Int) : Unit = {
      var output = activate()

      println(s"Perceptron output: $output desiredOutput: $desiredOutput")

      if (output == desiredOutput) {
        //weights are fine
      }
      else if (output == 0 && desiredOutput == 1) {
        weights = (weights, inputs).zipped.map((w, i) => w + adoptionRate * i)
      }
      else if (output == 1 && desiredOutput == 0) {
        weights = (weights, inputs).zipped.map((w, i) => w - adoptionRate * i)
      }
    }

    def process(iterations: Int, desiredOutput: Int) : Unit = {
      iterations match {
        case 0 => Unit
        case _ => {
          adaptWeights(desiredOutput)
          process(iterations - 1, desiredOutput)
        }
      }
    }
}
