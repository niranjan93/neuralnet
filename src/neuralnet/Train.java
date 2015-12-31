package neuralnet;

import org.jblas.DoubleMatrix;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Train {
    public static final double INITIAL_STEP_SIZE = 0.5;

    public static void main(String[] args) throws IOException {
        int i, j, epoch, total, numTrainingImages, numValidationImages;
        System.out.println("Loading training data...");
        DoubleMatrix[] data = DataUtils.loadImageData("/Users/niranjankumar/source/neuralnet/train-images-idx3-ubyte");
        DoubleMatrix[] labels = DataUtils.loadLabels("/Users/niranjankumar/source/neuralnet/train-labels-idx1-ubyte");
        total = data.length;
        numTrainingImages = (int) Math.floor(total * 0.9);
        numValidationImages = total - numTrainingImages;
        NeuralNetwork neuralNet = new NeuralNetwork(data[0].columns - 1, 200, 10);
        ObjectOutputStream nnStore;
        ArrayList<Integer> order = new ArrayList<Integer>(total);
        for (i = 0; i < total; i++) {
            order.add(i);
        }
        Collections.shuffle(order);
        DoubleMatrix[] trainingData = new DoubleMatrix[numTrainingImages];
        DoubleMatrix[] trainingLabels = new DoubleMatrix[numTrainingImages];
        for (i = 0; i < numTrainingImages; i++) {
            trainingData[i] = data[order.get(i)];
            trainingLabels[i] = labels[order.get(i)];
        }
        DoubleMatrix[] validationData = new DoubleMatrix[numValidationImages];
        DoubleMatrix[] validationLabels = new DoubleMatrix[numValidationImages];
        for (j = 0, i = numTrainingImages; j < numValidationImages; j++, i++) {
            validationData[j] = data[order.get(i)];
            validationLabels[j] = labels[order.get(i)];
        }
        System.out.printf("Randomly split into %d training images and %d validation images.\n", numTrainingImages, numValidationImages);
        System.out.println("Training...");

        epoch = 1;
        while (true) {
            System.out.printf("\nEpoch %d\n", epoch);

            // Perform one epoch of SGD
            neuralNet.trainEpoch(trainingData, trainingLabels, INITIAL_STEP_SIZE / Math.sqrt(epoch));

            // Calculate training and validation set accuracy
            DataUtils.getPredictionStats(neuralNet, trainingData, trainingLabels, "training");
            DataUtils.getPredictionStats(neuralNet, validationData, validationLabels, "validation");

            // Write network to a file
            nnStore = new ObjectOutputStream(new FileOutputStream("/Users/niranjankumar/source/neuralnet/neural-net-epoch" + Integer.toString(epoch)));
            nnStore.writeObject(neuralNet);
            nnStore.close();

            epoch++;
        }
    }
}
