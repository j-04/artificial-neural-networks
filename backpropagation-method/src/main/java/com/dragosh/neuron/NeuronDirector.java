package com.dragosh.neuron;

import java.util.List;
import java.util.function.Function;

public class NeuronDirector {
    public Neuron.Builder builder = new Neuron.Builder();

    public Neuron buildNeuron(List<Float> values,
                              List<Float> weights,
                              float shiftValue,
                              float shiftWeight) {
        Function<Float, Double> actFun = (Float net) -> (double) (1 / (1 + Math.pow(Math.E, (-1) * net)));
        builder.start()
                .setValues(values)
                .setWeights(weights)
                .setShift(shiftValue, shiftWeight)
                .setActivateFunction(actFun);

        return builder.build();
    }

    public Neuron rebuildNeuron(Neuron neuron,
                                List<Float> values,
                                List<Float> weights,
                                float shiftValue,
                                float shiftWeight) {
        builder.rebuildNeuron(neuron)
                .setValues(values)
                .setWeights(weights)
                .setShift(shiftValue, shiftWeight);
        return builder.build();
    }

    public Neuron setOut(Neuron neuron, float out) {
        builder.rebuildNeuron(neuron)
                .setOut(out);
        return builder.build();
    }
}
