package neuralnet;

import org.jblas.DoubleMatrix;
import java.lang.ClassNotFoundException;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.IOException;

public class Test {
    public static final int NUM_EPOCHS = 306;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        System.out.println("Loading test data...");
        DoubleMatrix[] testData = DataUtils.loadImageData("/Users/niranjankumar/source/neuralnet/t10k-images-idx3-ubyte");
        DoubleMatrix[] testLabels = DataUtils.loadLabels("/Users/niranjankumar/source/neuralnet/t10k-labels-idx1-ubyte");
        System.out.println("Loading neural network...");
        ObjectInputStream nnLoad = new ObjectInputStream(new FileInputStream("/Users/niranjankumar/source/neuralnet/neural-net-epoch" + Integer.toString(NUM_EPOCHS)));
        NeuralNetwork neuralNet = (NeuralNetwork) nnLoad.readObject();
        nnLoad.close();
        DataUtils.getPredictionStats(neuralNet, testData, testLabels, "test");
    }
}
