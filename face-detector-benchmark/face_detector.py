import dlib
import numpy as np

class DlibHOGFaceDetector():
    def __init__(self, nrof_upsample=0, det_threshold=0.0):
        self.hog_detector = dlib.get_frontal_face_detector()
        self.nrof_upsample = int(nrof_upsample)
        self.det_threshold = float(det_threshold)

    def detect_face(self, image):
        if image is None:
            raise ValueError("이미지가 로드되지 않았습니다. 이미지 경로를 확인하세요.")

        # dlib의 run 메소드 사용
        dets, scores, idx = self.hog_detector.run(image, self.nrof_upsample, self.det_threshold)

        # 감지된 얼굴 정보를 추출
        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.left())
            y1 = int(d.top())
            x2 = int(d.right())
            y2 = int(d.bottom())
            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)
    
class DlibCNNFaceDetector():
    def __init__(self, nrof_upsample=0, model_path='models/mmod_human_face_detector.dat'):
        if dlib.DLIB_USE_CUDA:
            print("CUDA is enabled in dlib, using GPU for CNN face detection.")
        else:
            print("CUDA is not enabled in dlib, using CPU for CNN face detection.")
        
        self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
        self.nrof_upsample = nrof_upsample

    def detect_face(self, image):

        dets = self.cnn_detector(image, self.nrof_upsample)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.rect.left())
            y1 = int(d.rect.top())
            x2 = int(d.rect.right())
            y2 = int(d.rect.bottom())
            score = float(d.confidence)

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)