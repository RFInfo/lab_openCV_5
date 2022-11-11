import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class FaceDetectDnn {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String tensorFlowWeightFile = "./models/dnn/opencv_face_detector_uint8.pb";
        String tensorFlowConfigFile = "./models/dnn/opencv_face_detector.pbtxt";

        String caffeWeightFile = "./models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel";
        String caffeConfigFile = "./models/dnn/deploy.prototxt";

        Net net;
        net = Dnn.readNetFromTensorflow(tensorFlowWeightFile,tensorFlowConfigFile);
//        net = Dnn.readNetFromCaffe(caffeConfigFile,caffeWeightFile);
        net.setPreferableBackend(Dnn.DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        Mat src = new Mat();
        Mat srcCopy = new Mat();
        Mat blob = new Mat();

        long frameCount = 0;
        long timeDnn = 0;
        long startTime, stopTime;
        double fpsDnn = 0.0;


        VideoCapture videoCapture = new VideoCapture(0);
//        VideoCapture videoCapture = new VideoCapture("./test_data/head-pose-face-detection-female-and-male.mp4");
//        VideoCapture videoCapture = new VideoCapture("./test_data/face-demographics-walking-and-pause.mp4");

        if(!videoCapture.isOpened()) return;

        while (videoCapture.read(src)){
            if(src.empty()) break;
            frameCount++;

            startTime = System.currentTimeMillis();
            src.copyTo(srcCopy);

//            blob = Dnn.blobFromImage(srcCopy);
            blob = Dnn.blobFromImage(srcCopy,1.0,new Size(300,300),new Scalar(104,117,123));

            net.setInput(blob);

            Mat detections = new Mat();
            detections = net.forward();

            detections = detections.reshape(1,(int)detections.total()/7);

            for (int i = 0; i < detections.rows(); i++) {
                double confidence = detections.get(i,2)[0];
                if(confidence > 0.7){
                    int left = (int)(detections.get(i,3)[0] * srcCopy.cols());
                    int top = (int)(detections.get(i,4)[0] * srcCopy.rows());

                    int right = (int)(detections.get(i,5)[0] * srcCopy.cols());
                    int bottom = (int)(detections.get(i,6)[0] * srcCopy.rows());

                    Imgproc.rectangle(srcCopy,new Point(left,top), new Point(right,bottom),new Scalar(0,255,0));
                }
            }

            stopTime = System.currentTimeMillis();
            timeDnn += (stopTime-startTime);
            fpsDnn = (double) frameCount/((double)timeDnn/1000);
            System.out.println("FPS Dnn " + fpsDnn);

            HighGui.imshow("Output",srcCopy);

            int key = HighGui.waitKey(10);
            if(key == 27)
                break;
        }
        HighGui.destroyAllWindows();
        System.exit(0);

    }
}
