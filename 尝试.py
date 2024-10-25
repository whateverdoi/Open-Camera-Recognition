from ultralytics import YOLO
import cv2

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 确保模型文件路径正确

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        # 读取帧
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # 在帧上运行YOLOv8检测
        results = model(frame, device='cpu')  # 使用CPU

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 显示结果
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # 按下 'q' 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # 按 'p' 键暂停
            cv2.waitKey(-1)  # 等待任意键继续

# 释放摄像头并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()

# 确保所有窗口关闭
for i in range(5):
    cv2.waitKey(1)