import sys
import numpy as np
import cv2

# 비디오 파일을 열려면 파일 이름을 넣고 카메라를 열려면 0(확인 필요)
Video_name = 'eagle_video.mp4'
resized_w = 600
resized_h = 400

# https://pjreddie.com/darknet/yolo/
model = 'yolo_v3/yolov3.weights'
config = 'yolo_v3/yolov3.cfg'

# https://github.com/pjreddie/darknet/blob/master/data/coco.names
class_labels = 'yolo_v3/coco.names'

confThreshold = 0.5
nmsThreshold = 0.4

count_bird = 0

#################################### Yolo_v3 ####################################

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기
classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 출력 레이어 이름 받아오기(총 3개의 출력 layer)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

#################################### Open Video ####################################

cap = cv2.VideoCapture(Video_name)

if not cap.isOpened():
    print("open failed!")
    sys.exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Frame width:', w)
print('Frame height:', h)
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS:', fps)

delay = round(1000 / fps)

#################################### Save Video ####################################

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

_out = cv2.VideoWriter('yolo_result.avi', fourcc, fps, (resized_w, resized_h))

if not _out.isOpened():
    print("open failed!")
    sys.exit()

#################################### 비디오 매 프레임 처리 ####################################
while True:
    ret, frame = cap.read()

    if not ret:
        break

    #################################### Resize Image ####################################
    
    srcQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32) #좌상단, 우상단, 우하단, 좌하단
    dstQuad = np.array([[0, 0], [resized_w-1, 0], [resized_w-1, resized_h-1], [0, resized_h-1]], np.float32) #좌상단, 우상단, 우하단, 좌하단

    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad) #3x3 투시변환 행렬
    dst = cv2.warpPerspective(frame, pers, (resized_w, resized_h))

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(dst, 1/255., (320, 320), swapRB=True) #(320, 320), (416, 416), (608, 608)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # outs는 3개의 ndarray 리스트.
    # outs[0].shape=(507, 85), 13*13*3=507
    # outs[1].shape=(2028, 85), 26*26*3=2028
    # outs[2].shape=(8112, 85), 52*52*3=8112

    class_ids = []
    confidences = []
    boxes = []

    for out in outs: #outs에 있는 모든 ndarray
        for detection in out: #하나의 행(85(4+1+80)개짜리 element)
            # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence)
            scores = detection[5:]
            class_id = np.argmax(scores)
            # Bird만 표시
            if int(class_id) == 14:
                confidence = scores[class_id]
                if confidence > confThreshold:
                    # 바운딩 박스 중심 좌표 & 박스 크기(0~1사이로 Normalize되어있음)
                    cx = int(detection[0] * resized_w)
                    cy = int(detection[1] * resized_h)
                    bw = int(detection[2] * resized_w)
                    bh = int(detection[3] * resized_h)

                    # 바운딩 박스 좌상단 좌표
                    sx = int(cx - bw / 2)
                    sy = int(cy - bh / 2)

                    boxes.append([sx, sy, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

                # 프레임별로 bird를 감지할때마다 count. 일정 수를 넘어가면 실제로 bird가 있다고 결정함
                count_bird += 1

        if count_bird > 0:
            count_bird -= 1
        # count_bird에서 더하고 빼는 크기와 실제로 bird가 있는지 판단하는 기준의 크기는 나중에 직접 실험해보면서 체크해야함
        

    # 비최대 억제(Non Maximnum Supression -> boxes의 box들 중에 confThreshold보다 큰 것들 중에 nmsThreshold(box들이 겹쳐있는 비율)보다 큰 box들 중에 confidence가 가장 큰 것)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold) # 몇번쨰 box를 쓸건지 index 정보 반환(N행 1열?)

    for i in indices:
        i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
        color = colors[class_ids[i]]
        cv2.rectangle(dst, (sx, sy, bw, bh), color, 2)
        cv2.putText(dst, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    t, _ = net.getPerfProfile() #Performance에 대한 Profile을 측정. Inference에 걸리는 시간 측정?
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(dst, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    _out.write(dst)

    cv2.imshow('dst', dst)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()