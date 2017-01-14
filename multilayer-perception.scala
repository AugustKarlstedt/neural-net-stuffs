import scala.math
import scala.util.Random

var k: Double = 1.0 // controls "spread" of the function

def sigmoid(input: Double) : Double = {
  1.0 / 1.0 + math.pow(math.E, -input * k)
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

  def activate() : Double = {
    sigmoid((weights, inputs).zipped.map(_ * _).sum)
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

class Layer() {

  var nodes: Array[Perceptron] = Array()

}

class MultilayerPerceptron() {

  var inputLayer: Layer = new Layer()
  var hiddenLayer: Layer = new Layer()
  var outputLayer: Layer = new Layer()

  def initialize() : Unit = {

  }

}

val mp = new MultilayerPerceptron()

val inputs: Array[Perceptron] = Array(new Perceptron(), new Perceptron(), new Perceptron(), new Perceptron())
inputs(0).initialize(Array(0,0))
inputs(1).initialize(Array(0,1))
inputs(2).initialize(Array(1,0))
inputs(3).initialize(Array(1,1))

val desiredOutputs: Array[Int] = Array(0, 1, 1, 0)

mp.hiddenLayer.nodes = Array(new Perceptron())
mp.outputLayer.nodes = Array(new Perceptron())
