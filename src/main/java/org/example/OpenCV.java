package org.example;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.videoio.VideoCapture;

import static java.awt.Image.SCALE_SMOOTH;
import static java.nio.charset.StandardCharsets.*;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_COMPLEX;
import static org.opencv.imgproc.Imgproc.FONT_ITALIC;

public class OpenCV {
    private static OpenCV openCV;
    public static OpenCV getInstance() {
        if (openCV == null) {
            synchronized (OpenCV.class) {
                openCV = new OpenCV();
            }
        }
        return openCV;
    }

    private final static String imagePath = new File("opencv/car.png").getAbsolutePath();
    private final static String weights = new File("opencv/yolov3.weights").getAbsolutePath();
    private final static String cfg = new File("opencv/yolov3.cfg").getAbsolutePath();
    private final static Path namesPath = Path.of("opencv/coco.names");
    private final VideoCapture vc;
    private final JFrame frame;
    private final JLabel label;
    private final Net net;

    private final Size sz = new Size(288, 288);
    private final List<String> names = new ArrayList<>();
    private final List<Mat> result = new ArrayList<>();
    private final List<String> outBlobNames;

    private List<String> getOutputNames() {
        List<String> names = net.getLayerNames();
        return net.getUnconnectedOutLayers()
                .toList()
                .stream()
                .map(v -> --v)
                .map(names::get)
                .toList();
    }

    private BufferedImage getImage(Mat image) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", image, matOfByte);
        byte[] bytes = matOfByte.toArray();
        InputStream in = new ByteArrayInputStream(bytes);
        try {
            return ImageIO.read(in);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private OpenCV() {
        System.load(new File("opencv/libopencv_java460.dylib").getAbsolutePath());
        vc = new VideoCapture(imagePath);
        this.frame = new JFrame();
        this.label = new JLabel();
        frame.setContentPane(label);
        frame.setSize(600, 600);
        frame.setVisible(true);

        this.net = Dnn.readNetFromDarknet(cfg, weights);
        this.outBlobNames = getOutputNames();
        try {
            Files.readAllLines(namesPath, UTF_8)
                    .stream()
                    .map(String::strip)
                    .forEach(names::add);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void start() {
        video();
    }

    private void video() {
        final Mat frame = new Mat();
        while (vc.read(frame)) {
            Mat blob = Dnn.blobFromImage(frame, 0.00392, sz, new Scalar(0), true, false);
            this.net.setInput(blob);
            this.net.forward(result, outBlobNames);

            float confThreshold = 0.6f;
            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect2d> rects = new ArrayList<>();

            for (Mat level : result) {
                for (int j = 0; j < level.rows(); ++j) {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());
                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float) mm.maxVal;
                    Point classIdPoint = mm.maxLoc;
                    if (confidence > confThreshold) {
                        int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                        int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                        int width   = (int) (row.get(0, 2)[0] * frame.cols());
                        int height  = (int) (row.get(0, 3)[0] * frame.rows());
                        int left    = centerX - width / 2;
                        int top     = centerY - height / 2;

                        clsIds.add((int) classIdPoint.x);
                        confs.add(confidence);
                        rects.add(new Rect2d(left, top, width, height));
                    }
                }
            }

            clsIds.stream()
                    .map(names::get)
                    .forEach(System.out::println);
            System.out.println(confs);

            float nmsThresh = 0.5f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
            MatOfRect2d boxes = new MatOfRect2d(boxesArray);
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

            int[] ind = indices.toArray();
            Collections.reverse(clsIds);
            for (int i = 0; i<ind.length; i++) {
                Rect2d box = boxesArray[ind[i]];
                Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
                String name = names.get(clsIds.get(i)) + " " + confs.get(i);
                Imgproc.putText(
                        frame, name, box.tl(),
                        FONT_ITALIC, 1,
                        new Scalar(0, 0, 255), 2);
            }

            ImageIcon image = new ImageIcon(getImage(frame));
            Image img = image.getImage().getScaledInstance(600, 600, SCALE_SMOOTH);
            label.setIcon(new ImageIcon(img));
            label.repaint();
        }
    }
}
