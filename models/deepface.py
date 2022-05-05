from deepface import DeepFace
from deepface.detectors import FaceDetector

BORDER_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

# Face Detection
FACE_DETECTOR_BACKEND = "ssd"
FACE_DETECTOR_MODEL = FaceDetector.build_model(FACE_DETECTOR_BACKEND)
FACE_DETECTOR_PARAMS = {
    'face_detector': FACE_DETECTOR_MODEL,
    'detector_backend': FACE_DETECTOR_BACKEND,
    'align': False
}

# Facial Feature Analysis
FACIAL_FEATURES = ['emotion']
FACIAL_FEATURE_ANALYSIS_MODELS = {
    feature: DeepFace.build_model(feature.capitalize())
    for feature in FACIAL_FEATURES
}
FACIAL_FEATURE_ANALYSIS_PARAMS = {
    'models': FACIAL_FEATURE_ANALYSIS_MODELS,
    'actions': FACIAL_FEATURES,
    'detector_backend': 'skip',
    'enforce_detection': False,
    'prog_bar': False
}

# Face Recognition
FACE_RECOGNITION_MODEL_NAME = 'VGG-Face'
FACE_RECOGNITION_MODEL = DeepFace.build_model(FACE_RECOGNITION_MODEL_NAME)
FACE_RECOGNITION_PARAMS = {
    'db_path': "face_recog_db",
    'model_name': FACE_RECOGNITION_MODEL_NAME,
    'model': FACE_RECOGNITION_MODEL,
    'detector_backend': 'skip',
    'enforce_detection': False,
    'prog_bar': False
}


def extract_facial_feature_info(features):
    extracted_features = {}

    for feature in FACIAL_FEATURES:
        if feature in ["age", "gender"]:
            extracted_features[feature] = features[feature]
        else:
            extracted_features[feature] = features['dominant_' + feature]

    return extracted_features


def extract_face_recognition_data(match):
    return match['identity'].split('/')[-1][:-4]


class FacialAnalyser():
    def analyse(self, frame):
        result = []
        faces = FaceDetector.detect_faces(img=frame, **FACE_DETECTOR_PARAMS)

        target_face = None
        person_counter = 1
        for face, bbox in faces:
            target_face = face
            matches = DeepFace.find(face, **FACE_RECOGNITION_PARAMS)

            if len(matches):
                name = extract_face_recognition_data(matches.iloc[0])
            else:
                name = f"Person {person_counter}"
                person_counter += 1

            facial_features = DeepFace.analyze(
                face, **FACIAL_FEATURE_ANALYSIS_PARAMS)
            extracted_facial_features = extract_facial_feature_info(
                facial_features)

            result.append((
                bbox,
                name,
                extracted_facial_features
            ))

        return result, target_face
