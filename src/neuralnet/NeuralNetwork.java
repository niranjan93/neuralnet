package neuralnet;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import java.io.Serializable;

public class NeuralNetwork implements Serializable {
    private DoubleMatrix w1;
    private DoubleMatrix w2;
    private int numFeatures;
    private int hiddenLayerSize;

    public NeuralNetwork(int numFeatures, int hiddenLayerSize, int numClasses) {
        this.numFeatures = numFeatures;
        this.hiddenLayerSize = hiddenLayerSize;
        w1 = DoubleMatrix.randn(numFeatures + 1, hiddenLayerSize + 1).muli(0.01);
        w1.mulColumn(hiddenLayerSize, 0.0);
        w1.put(numFeatures, hiddenLayerSize, Double.POSITIVE_INFINITY);
        w2 = DoubleMatrix.randn(hiddenLayerSize + 1, numClasses).muli(0.01);
    }

    public DoubleMatrix predict(DoubleMatrix image) {
        return sigmoidi(sigmoidi(image.mmul(w1)).mmul(w2));
    }

    public void trainEpoch(DoubleMatrix[] trainingData, DoubleMatrix[] trainingLabels, double stepSize) {
        int i;
        DoubleMatrix x2, h, backProp;
        for (i = 0; i < trainingData.length; i++) {
            x2 = sigmoidi(trainingData[i].mmul(w1));
            h = sigmoidi(x2.mmul(w2));
            backProp = h.mul(h.rsub(1.0)).muli(h.sub(trainingLabels[i])).transpose();
            w1.subi(x2.mul(x2.rsub(1.0)).transpose().muli(w2.mmul(backProp)).mmul(trainingData[i]).transpose().muli(stepSize));
            w1.mulColumn(hiddenLayerSize, 0.0);
            w1.put(numFeatures, hiddenLayerSize, Double.POSITIVE_INFINITY);
            w2.subi(backProp.mmul(x2).muli(stepSize));
        }
    }

    private static DoubleMatrix sigmoidi(DoubleMatrix mat) {
        return MatrixFunctions.expi(mat.rsubi(0.0)).addi(1.0).rdivi(1.0);
    }
}
