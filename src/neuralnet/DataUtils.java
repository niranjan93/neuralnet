package neuralnet;

import org.jblas.DoubleMatrix;
import java.io.FileInputStream;
import java.io.IOException;

public class DataUtils {

    public static int bytearrToInt(byte[] arr) {
        return (((int) arr[0]) << 24) |
                ((0xFF & (int) arr[1]) << 16) |
                ((0xFF & (int) arr[2]) << 8) |
                (0xFF & (int) arr[3]);
    }

    public static DoubleMatrix[] loadImageData(String fileName) throws IOException {
        int readFlag, dim1, dim2, dim3, imgArea, i, j;
        DoubleMatrix[] result;
        FileInputStream stream = new FileInputStream(fileName);
        byte[] fourByteBuffer = new byte[4];
        byte[] imgRow;
        readFlag = stream.read(fourByteBuffer);
        assert (readFlag == 4);
        assert (fourByteBuffer[0] == (byte) 0);
        assert (fourByteBuffer[1] == (byte) 0);
        assert (fourByteBuffer[2] == (byte) 8);
        assert (fourByteBuffer[3] == (byte) 3);
        readFlag = stream.read(fourByteBuffer);
        assert (readFlag == 4);
        dim1 = bytearrToInt(fourByteBuffer);
        readFlag = stream.read(fourByteBuffer);
        assert (readFlag == 4);
        dim2 = bytearrToInt(fourByteBuffer);
        readFlag = stream.read(fourByteBuffer);
        assert (readFlag == 4);
        dim3 = bytearrToInt(fourByteBuffer);
        imgArea = dim2 * dim3;
        imgRow = new byte[imgArea];
        result = new DoubleMatrix[dim1];
        for (i = 0; i < dim1; i++) {
            result[i] = new DoubleMatrix(1, imgArea + 1);
            readFlag = stream.read(imgRow);
            assert (readFlag == imgArea);
            for (j = 0; j < imgArea; j++) {
                result[i].put(0, j, (double) (((int) imgRow[j]) & 0xFF));
            }
            result[i].put(0, imgArea, 0.0);
            result[i].divi(result[i].norm2());
            result[i].put(0, imgArea, 1.0);
        }
        stream.close();
        System.out.printf("Loaded %d %dx%d images.\n", dim1, dim2, dim3);
        return result;
    }

    public static DoubleMatrix[] loadLabels(String fileName) throws IOException {
        int readFlag, numLabels, i;
        DoubleMatrix[] result;
        FileInputStream stream = new FileInputStream(fileName);
        byte[] fourByteBuffer = new byte[4];
        byte[] labels;
        readFlag = stream.read(fourByteBuffer);
        assert(readFlag == 4);
        assert(fourByteBuffer[0] == (byte) 0);
        assert(fourByteBuffer[1] == (byte) 0);
        assert(fourByteBuffer[2] == (byte) 8);
        assert(fourByteBuffer[3] == (byte) 1);
        readFlag = stream.read(fourByteBuffer);
        assert(readFlag == 4);
        numLabels = bytearrToInt(fourByteBuffer);
        labels = new byte[numLabels];
        readFlag = stream.read(labels);
        assert(readFlag == numLabels);
        result = new DoubleMatrix[numLabels];
        for (i = 0; i < numLabels; i++) {
            result[i] = DoubleMatrix.zeros(1, 10);
            result[i].put(0, labels[i], 1.0);
        }
        stream.close();
        System.out.printf("Loaded %d labels.\n", numLabels);
        return result;
    }

    public static void getPredictionStats(NeuralNetwork neuralNet, DoubleMatrix[] data, DoubleMatrix[] labels, String dataset) {
        int i, numCorrect, total;
        numCorrect = 0;
        total = data.length;
        for (i = 0; i < total; i++) {
            if (neuralNet.predict(data[i]).argmax() == labels[i].argmax()) {
                numCorrect++;
            }
        }
        System.out.printf("Correctly classified %d out of %d images in %s set.\n", numCorrect, total, dataset);
        System.out.printf("Accuracy in %s set: %f\n", dataset, ((double) numCorrect) / total);
    }
}
