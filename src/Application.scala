/**
  * Created by augustk on 3/20/2017.
  */
object Application {

  def main(args: Array[String]): Unit = {

    val p = new Perceptron()
    p.initialize(Array(0,1))
    p.process(100, 1)

  }

}
