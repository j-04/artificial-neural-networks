package com.ai.task2

import com.google.gson.annotations.Expose
import org.slf4j.LoggerFactory

class Neuron private constructor() {
    @Expose(serialize = false, deserialize = false) private val logger = LoggerFactory.getLogger(this.javaClass)

    @Expose(serialize = false, deserialize = false) var id = 0
    @Expose(serialize = true, deserialize = true) var listOfValues: List<Double> = listOf()
    @Expose(serialize = true, deserialize = true) var listOfWeights: List<Double> = listOf()
    @Expose(serialize = false, deserialize = false) var threshold: Int = 0
    @Expose(serialize = false, deserialize = false) var activationFun: (net: Double, threshold: Int) -> (Int) = { _, _ -> 0 }

    fun start(): Int {
        if (listOfValues.size != listOfWeights.size)
            throw IllegalArgumentException("Quantity of values and weights are not equal!")
        var net = 0.0
        for (i in listOfValues.indices) {
            net += listOfValues[i] * listOfWeights[i]
        }
        val out = activationFun(net, threshold)
        logger.info("Neuron Id: $id; Values: $listOfValues; NET: $net; OUT: $out")
        return out
    }

    class Builder {
        private var neuron = Neuron()

        fun setId(id: Int): Builder {
            neuron.id = id
            return this
        }

        fun setListOfValues(listOfValues: List<Double>): Builder {
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