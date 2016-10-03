import scala.util.Random

def heaviside(input: Float) : Int = {
  input match {
    case _ if input > 0 => 1
    case _ => 0
  }
}

class Perceptron() {

  val threshold: Float = Random.nextFloat
  val bias: Array[Float] = Array(-threshold)
  val adoptionRate: Float = 0.01f // 0 <= n <= 1
  var weights: Array[Float] = Array()
  var inputs: Array[Float] = Array()

  def initialize(input: Array[Float]) : Unit = {
    weights = bias ++ (for (i <- 1 to input.length) yield Random.nextFloat * 2 - 1)
    inputs = Array(1.0f) ++ input
  }

  def activate() : Int = {
    heaviside((weights, inputs).zipped.map(_ * _).sum)
  }

  def adaptWeights(desiredOutput: Int) : Unit = {
    var output = activate()

    println("output: " + output + " desiredOutput: " + desiredOutput)

    if (output == desiredOutput) {
      //weights are fine
    }
    else if (output == 0 && desiredOutput == 1) {
      weights = (weights, inputs).zipped.map(_ + adoptionRate * _).toArray
    }
    else if (output == 1 && desiredOutput == 0) {
      weights = (weights, inputs).zipped.map(_ - adoptionRate * _).toArray
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

val p = new Perceptron()
p.initialize(Array(1,2,3,4,5,6,7,8,9))
p.process(10, 0)
