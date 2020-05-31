package com.dragosh.neuron;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

@ToString
@EqualsAndHashCode
@Getter
public class Neuron {
    private float out;
    private Map<Float, Float> shift;
    private List<Float> values;
    private List<Float> weights;
    private Function<Float, Double> activateFunction;

    private Neuron () {

    }

    public static class Builder {
        private Neuron currentNeuron;

        public Builder start() {
            currentNeuron = new Neuron();
            return this;
        }

        public Builder setValues(List<Float> values) {
            Objects.requireNonNull(values, "Nullable list of values!");
            currentNeuron.values = values;
            return this;
        }

        public Builder setWeights(List<Float> weights) {
            Objects.requireNonNull(weights, "Nullable list of weights!");
            currentNeuron.weights = weights;
            return this;
        }

        public Builder setShift(Float value, Float weight) {
            Objects.requireNonNull(value, "Nullable value of shift!");
            Objects.requireNonNull(value, "Nullable weight of shift!");
            currentNeuron.shift = Map.of(value, weight);
            return this;
        }

        public Builder setActivateFunction(Function<Float, Double> activateFunction) {
            Objects.requireNonNull(activateFunction);
            currentNeuron.activateFunction = activateFunction;
            return this;
        }

        public Builder setOut(float out) {
            currentNeuron.out = out;
            return this;
        }

        public Builder rebuildNeuron(Neuron neuron) {
            Objects.requireNonNull(neuron, "Nullable neuron!");
            currentNeuron = neuron;
            return this;
        }

        public Builder reset() {
            currentNeuron = new Neuron();
            return this;
        }

        public Neuron build() {
            final Neuron n = currentNeuron;
            Objects.requireNonNull(n.values, "Nullable list of values!");
            Objects.requireNonNull(n.weights, "Nullable list of weights!");
            Objects.requireNonNull(n.activateFunction, "Nullable activate function!");
            if (n.values.size() != n.weights.size()) {
                throw new IllegalArgumentException("Different sizes of values and weights lists! Must be same!");
            }
            return n;
        }
    }
}
