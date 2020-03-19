package com.ai.task2

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

class NeuralNetworkTeacher() {
    var gson: Gson = GsonBuilder()
            .setPrettyPrinting()
            .serializeNulls()
            .excludeFieldsWithoutExposeAnnotation()
            .create()

    val teacher = HebbTeacher()

    var image: Map<String, String> = mapOf()
    val neurons: MutableList<Neuron> = mutableListOf()

    init {
        val reader = BufferedReader(FileReader("digits.json"))
        val token = object: TypeToken<Map<String, String>>() {}.type
        image = gson.fromJson(reader, token)
    }

    fun startMachineLearning() {
        prepareNeurons()
        for ((i, neuron) in neurons.withIndex()) {
            var taught = false

            val weights = mutableListOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            neuron.id = i
            neuron.listOfWeights = weights

            while(!taught) {
                var resultIndex = 0
                for ((key, value) in image) {
                    var expectedResult = image.keys.toMutableList()[i][resultIndex].toString().toDouble()
                    if (expectedResult == 0.0)
                        expectedResult = -1.0
                    val values = value.split("").filter { it.isNotEmpty() }.map { it.toDouble() }.toMutableList()
                    values.add(0.0)
                    neuron.listOfValues = values
                    val actualResult = neuron.start().toDouble()
                    if (expectedResult != actualResult) {
                        teacher.teachNeuron(neuron, expectedResult)
                        break
                    }
                    resultIndex++
                    if (image.entries.last().key == key)
                        taught = true
                }
            }
        }
        saveTaughtNeurons()
    }

    private fun prepareNeurons() {
        val actFun: (Double, Int) -> (Int) = { net, _ -> if (net > 0) 1 else -1 }

        for (i in 0..9) {
            val neuron = Neuron.Builder()
                    .setActivationFun(actFun)
                    .setThreshold(0)
                    .create()
            neurons.add(neuron)
        }
    }

    private fun saveTaughtNeurons() {
        val file = File("neurons.json")
        if (!file.exists())
            file.createNewFile()
        val json = gson.toJson(neurons)
        file.writeText(json)
    }
}