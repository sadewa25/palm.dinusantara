from model.YOLOOnnx import YOLOOnnx
import cv2

detection = YOLOOnnx(onnx_model= "/Users/sadewawicak/Researchs/Dinusantara/palm.dinusantara/yolov8n_development/20250106_092214/weights/best.onnx", input_image= "/Users/sadewawicak/Researchs/Dinusantara/palm.dinusantara/sample/assignment_test_palm.jpeg", confidence_thres=0.1, iou_thres=0.1, classes=['palm trees'])
output_image = detection.main()
# Display the output image in a window
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", output_image)

# Wait for a key press to exit
cv2.waitKey(0)

