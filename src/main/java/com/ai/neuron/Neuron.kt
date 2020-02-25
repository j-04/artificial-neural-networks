package com.ai.neuron

class Neuron private constructor() {
    var listOfValues: List<Double> = listOf()
    var listOfWeights: List<Double> = listOf()
    var threshold: Int = 0
    var activationFun: (net: Double, threshold: Int) -> (Int) = { _, _ -> 0 }

    fun start(): Int {
        if (listOfValues.size != listOfWeights.size)
            throw IllegalArgumentException("Quantity of values and weights are not equal!")
        var net = 0.0
        for (i in listOfValues.indices) {
            net += listOfValues[i] * listOfWeights[i]
        }
        return activationFun(net, threshold)
    }

    class Builder {
        private var neuron = Neuron()

        fun setListOfValues(listOfValues: List<Double>): Builder{
            neuron.listOfValues = listOfValues
            return this
        }

        fun setListOfWeights(listOfWeights: List<Double>): Builder {
            neuron.listOfWeights = listOfWeights
            return this
        }

        fun setThreshold(threshold: Int): Builder {
            neuron.threshold = threshold
            return this
        }

        fun setActivationFun(activationFun: (Double, Int) -> (Int)): Builder {
            neuron.activationFun = activationFun
            return this
        }

        fun create(): Neuron {
            return neuron
        }
    }
}