import cv2
import numpy as np
import onnxruntime as ort

img = cv2.imread("sample/assignment_test_palm.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))

# Prepare the image for the model
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)  # Add batch dimension
img = np.transpose(img, (0, 3, 1, 2))  # Change to NCHW format

# Load the ONNX model
session = ort.InferenceSession("model/count_best.onnx")

# Run the model
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img})

# Process the results
results = outputs[0][0]  # Assuming the first output is the detection results

# Convert image back to HWC format for visualization
img = np.transpose(img[0], (1, 2, 0)).astype(np.uint8)

for idx, box in enumerate(results, start=1):
    x1, y1, x2, y2, conf, class_id = box[:6]
    if conf > 0.25:  # Confidence threshold
        cv2.rectangle(img, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (0, 0, 255), 1)
        
        cv2.putText(img,
                   f'{idx}',
                   (int(x1), int(y1-5)), # position slightly above box
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, # font scale
                   (0, 0, 255), # color (BGR)
                   1)

# Save the image with results
cv2.imwrite("output/result.jpg", img)

# Display the image with results
cv2.imshow("Detection Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()