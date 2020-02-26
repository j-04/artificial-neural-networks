import com.ai.neuron.Neuron

fun main() {
    val actFun: (Double, Int) -> (Int) = { net, t ->
        if (net < t) 0 else 1
    }
    println("Conjunction")
    turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 2, actFun)
    turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 2, actFun)
    turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 2, actFun)
    turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 2, actFun)
    println("Disjunction")
    turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 1, actFun)
    turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 1, actFun)
    turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 1, actFun)
    turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 1, actFun)
    println("Inversion")
    turnOnNeuron(listOf(0.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0, actFun)
    turnOnNeuron(listOf(1.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0, actFun)

//    println("Conjunction with shifting")
//    turnOnNeuron(listOf(0.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(0.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
}

fun turnOnNeuron(inputs: List<Double>,
                 weights: List<Double>,
                 threshold: Int,
                 activationFun: (net: Double, threshold: Int) -> (Int)) {
    val neuron = Neuron.Builder()
            .setListOfValues(inputs)
            .setListOfWeights(weights)
            .setThreshold(threshold)
            .setActivationFun(activationFun)
            .create()

    val out = neuron.start()
    println(out)
}