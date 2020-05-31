package com.ai.task2

import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import org.slf4j.LoggerFactory
import java.io.BufferedReader
import java.io.FileReader

class DigitRecognizer {
    private val logger = LoggerFactory.getLogger(this.javaClass)
    var neurons: List<Neuron> = mutableListOf()

    fun recognizeDigits() {
        // Teaching neurons
        val neuralNetworkTeacher = NeuralNetworkTeacher()
        neuralNetworkTeacher.startMachineLearning()

        // Use neurons to recognize images of digits
        val digitRecognizer = DigitRecognizer()
        digitRecognizer.neurons = neuralNetworkTeacher.neurons

        //
        val images: Map<String, String> = GsonBuilder().create().fromJson(BufferedReader(FileReader("digits.json")), object: TypeToken<Map<String, String>>() {}.type)
        for ((_, value) in images) {
            digitRecognizer.recogniseDigit(getImage(value))
        }
    }

    private fun recogniseDigit(image: List<Double>) {
        val results = mutableListOf<Int>()
        for (neuron in neurons) {
            neuron.listOfValues = image
            val result = neuron.start()
            results.add(result)
        }
        determineResult(results)
    }

    private fun determineResult(results: List<Int>) {
        when (results) {
            listOf(1, -1, -1, -1, -1, -1, -1, -1, -1, -1) -> println(0)
            listOf(-1, 1, -1, -1, -1, -1, -1, -1, -1, -1) -> println(1)
            listOf(-1, -1, 1, -1, -1, -1, -1, -1, -1, -1) -> println(2)
            listOf(-1, -1, -1, 1, -1, -1, -1, -1, -1, -1) -> println(3)
            listOf(-1, -1, -1, -1, 1, -1, -1, -1, -1, -1) -> println(4)
            listOf(-1, -1, -1, -1, -1, 1, -1, -1, -1, -1) -> println(5)
            listOf(-1, -1, -1, -1, -1, -1, 1, -1, -1, -1) -> println(6)
            listOf(-1, -1, -1, -1, -1, -1, -1, 1, -1, -1) -> println(7)
            listOf(-1, -1, -1, -1, -1, -1, -1, -1, 1, -1) -> println(8)
            listOf(-1, -1, -1, -1, -1, -1, -1, -1, -1, 1) -> println(9)
            else -> println(-1)
        }
    }

    private fun getImage(image: String): List<Double> {
        val result = image.split("").filter { it.isNotEmpty() }.map { it.toDouble() }.toMutableList()
        result.add(0.0)
        return result
    }
}