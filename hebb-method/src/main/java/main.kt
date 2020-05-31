@file:JvmName("MainKt")
import com.ai.task1.Neuron
import com.ai.task2.DigitRecognizer

fun main() {
    firstHomeTask()
}

fun secondHomeTask() {
    val digitRecognizer = DigitRecognizer()
    digitRecognizer.recognizeDigits()
}

//-------------------------------
fun firstHomeTask() {
    var actFun: (Double, Int) -> (Int) = { net, t ->
        if (net < t) 0 else 1
    }
    println("Conjunction")
    println(turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 2, actFun))
    println(turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 2, actFun))
    println(turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 2, actFun))
    println(turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 2, actFun))

    println("Disjunction")
    println(turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 1, actFun))
    println(turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 1, actFun))
    println(turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 1, actFun))
    println(turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 1, actFun))

    println("Inversion")
    println(turnOnNeuron(listOf(0.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0, actFun))
    println(turnOnNeuron(listOf(1.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0, actFun))
//    actFun = (net:Double, t: Int ->
//    println("Conjunction with shifting")
//    turnOnNeuron(listOf(0.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(0.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
}

fun turnOnNeuron(inputs: List<Double>,
                 weights: List<Double>,
                 threshold: Int,
                 activationFun: (net: Double, threshold: Int) -> (Int)): List<Int> {
    val neuron = Neuron.Builder()
            .setListOfValues(inputs)
            .setListOfWeights(weights)
            .setThreshold(threshold)
            .setActivationFun(activationFun)
            .create()
    return listOf(neuron.start())
}