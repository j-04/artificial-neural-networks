package com.dragosh.teacher;

import com.dragosh.neuron.Neuron;
import com.dragosh.neuron.NeuronDirector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Iterator;

public class BackPropagationTeacher {
    private int inputU1;
    private int inputU2;
    private float teachRate = 0.5f;
    private byte goal;
    private short iterations;
    private Neuron hiddenNeuronU3;
    private Neuron hiddenNeuronU4;
    private Neuron outputLayer;
    private NeuronDirector director;

    private static final Logger LOG = LoggerFactory.getLogger(BackPropagationTeacher.class);

    public BackPropagationTeacher() {
        iterations = 5000;
        director = new NeuronDirector();
    }

    public void start() {
        prepareNeurons();
        startTeaching();
    }

    private void prepareNeurons() {
        Neuron hiddenNeuronU3 = director.buildNeuron(List.of(0f, 1f), List.of(1f, 0.5f), 1, 1);
        Neuron hiddenNeuronU4 = director.buildNeuron(List.of(0f, 1f), List.of(-1.0f, 2f), 1, 1);
        Neuron outputNeuronU5 = director.buildNeuron(List.of(1f, 1f), List.of(1.5f, -1.0f), 1, 1);

        this.hiddenNeuronU3 = hiddenNeuronU3;
        this.hiddenNeuronU4 = hiddenNeuronU4;

        outputLayer = outputNeuronU5;

        inputU1 = 0;
        inputU2 = 1;

        goal = 1;
    }

    private void startTeaching() {
        for (int i = 0; i < iterations; i++) {
            float result = getActualGoal();
            LOG.info(String.format("Goal: %d; Actual: %f", goal, result));
            LOG.info("Starting teaching process...");
            teachProcess(result);
        }
        float result = getActualGoal();
        LOG.info(String.format("Goal: %d; Actual: %f", goal, result));
    }

    private float getActualGoal() {
        float outU3 = getNeuronSum(hiddenNeuronU3);
        director.setOut(hiddenNeuronU3, outU3);
        float outU4 = getNeuronSum(hiddenNeuronU4);
        director.setOut(hiddenNeuronU4, outU4);
        Map<Float, Float> shift = outputLayer.getShift();
        float shiftValue = shift.keySet().iterator().next();
        float shiftWeight = shift.get(shiftValue);
        Neuron rebuiltOutputNeuron = director.rebuildNeuron(outputLayer,
                List.of(outU3, outU4), outputLayer.getWeights(),
                shiftValue, shiftWeight);
        return getNeuronSum(rebuiltOutputNeuron);
    }

    private float getNeuronSum(Neuron neuron) {
        Iterator<Float> values = neuron.getValues().iterator();
        Iterator<Float> weights = neuron.getWeights().iterator();
        Map<Float, Float> shift = neuron.getShift();
        float shiftValue = shift.keySet().iterator().next();
        float shiftWeight = shift.get(shiftValue);
        float sum = 0.0f;
        while(values.hasNext() && weights.hasNext()) {
            float value = values.next();
            float weight = weights.next();
            sum += value * weight;
        }
        sum += shiftValue * shiftWeight;
        return  ((float) (double) neuron.getActivateFunction().apply(sum));
    }

    private void teachProcess(float actualResult) {
        float outU3 = hiddenNeuronU3.getOut();
        float outU4 = hiddenNeuronU4.getOut();
        float weight53 = outputLayer.getWeights().get(0);
        float weight54 = outputLayer.getWeights().get(1);

        float E = (0.5f * (float) Math.pow((1 - actualResult), 2));
        LOG.info(String.format("The total squared error: %f", E));
        float outputErrorRate = actualResult * (1 - actualResult) * (goal - actualResult);
        LOG.info(String.format("Output error rate: %f", outputErrorRate));

        float errorRateU3 = outU3 * (1 - outU3) * (weight53 * outputErrorRate);
        LOG.info(String.format("Error rate of neuron U3: %f", errorRateU3));

        float errorRateU4 = outU4 * (1 - outU4) * (weight54 * outputErrorRate);
        LOG.info(String.format("Error rate of neuron U4: %f", errorRateU4));

        //Updating weight between output layer and hidden layer
        float newWeight53 = weight53 + teachRate * outputErrorRate * outU3;
        LOG.info(String.format("Update weight 53 of neuron U5: %f", newWeight53));

        float newWeight54 = weight54 + teachRate * outputErrorRate * outU4;
        LOG.info(String.format("Update weight 54 of neuron U5: %f", newWeight54));

        Map<Float, Float> newShiftU5 = updateShift(outputLayer, outputErrorRate);
        float shiftValue = newShiftU5.keySet().iterator().next();
        float shiftWeight = newShiftU5.get(shiftValue);
        outputLayer = director.rebuildNeuron(outputLayer,
                outputLayer.getValues(), List.of(newWeight53, newWeight54),
                shiftValue, shiftWeight);

        updateNeuron(hiddenNeuronU3, errorRateU3);
        updateNeuron(hiddenNeuronU4, errorRateU4);
    }

    private void updateNeuron(Neuron neuron, float errorRate) {
        float weight1 = neuron.getWeights().get(0);
        float weight2 = neuron.getWeights().get(1);
        float value1 = neuron.getValues().get(0);
        float value2 = neuron.getValues().get(1);

        float newWeight1 = weight1 + teachRate * errorRate * value1;
        float newWeight2 = weight2 + teachRate * errorRate * value2;

        Map<Float, Float> newShift = updateShift(neuron, errorRate);
        float shiftValue = newShift.keySet().iterator().next();
        float shiftWeight = newShift.get(shiftValue);

        director.rebuildNeuron(neuron,
                neuron.getValues(), List.of(newWeight1, newWeight2),
                shiftValue, shiftWeight);
    }

    private Map<Float, Float> updateShift(Neuron neuron, float outputErrorRate) {
        Map<Float, Float> shift = neuron.getShift();
        float shiftValue = shift.keySet().iterator().next();
        float shiftWeight = shift.get(shiftValue);
        shiftWeight = shiftValue + teachRate * outputErrorRate * shiftValue;
        return Map.of(shiftValue, shiftWeight);
    }
}
