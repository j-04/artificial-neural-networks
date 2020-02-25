import com.ai.neuron.Neuron

fun main() {
    println("Conjunction")
    turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 2)
    turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 2)
    turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 2)
    turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 2)
    println("Disjunction")
    turnOnNeuron(listOf(0.0, 0.0), listOf(1.0, 1.0), 1)
    turnOnNeuron(listOf(0.0, 1.0), listOf(1.0, 1.0), 1)
    turnOnNeuron(listOf(1.0, 0.0), listOf(1.0, 1.0), 1)
    turnOnNeuron(listOf(1.0, 1.0), listOf(1.0, 1.0), 1)
    println("Inversion")
    turnOnNeuron(listOf(0.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0)
    turnOnNeuron(listOf(1.0, 1.0), listOf(-1.0, (1 / 2).toDouble()), 0)


//    println("Conjunction with shifting")
//    turnOnNeuron(listOf(0.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(0.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 0.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
//    turnOnNeuron(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, (- 3 / 2).toDouble()), 0)
}

fun turnOnNeuron(inputs: List<Double>, weights: List<Double>, threshold: Int) {
    val neuron = Neuron.Builder()
            .setListOfValues(inputs)
            .setListOfWeights(weights)
            .setThreshold(threshold)
            .setActivationFun { net, t ->
                if (net < t) 0 else 1
            }
            .create()

    val out = neuron.start()
    println(out)
}