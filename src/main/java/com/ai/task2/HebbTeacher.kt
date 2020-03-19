package com.ai.task2

class HebbTeacher {
    fun teachNeuron(neuron: Neuron, expectedResult: Double) {
        val copyValues = neuron.listOfValues
        val copyWeights = neuron.listOfWeights

        val lastIndex = copyValues.lastIndex

        val newWeights = mutableListOf<Double>()
        for (index in 0..(copyValues.size - 2)) {
            val newWeight = calculateNewWeights(copyWeights[index], copyValues[index], expectedResult)
            newWeights.add(newWeight)
        }
        val threshold = calculateThreshold(copyWeights[lastIndex], expectedResult)
        newWeights.add(threshold)
        neuron.listOfWeights = newWeights
    }

    private fun calculateNewWeights(oldWeight: Double, value: Double, expectedResult: Double) = oldWeight + value * expectedResult

    private fun calculateThreshold(prevThreshold: Double, expectedResult: Double) = prevThreshold - expectedResult
}