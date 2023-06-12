from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import mediapipe as mp
import numpy as np

# importing drawing utils to visualise our poses
mp_drawing = mp.solutions.drawing_utils
# mp_pose-> importing our pose estimation model
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Second
    c = np.array(c)  # Third

    #    y2-y1  ,  x2-x1
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


@csrf_exempt
def shoulder_estimation1(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    #             RIGHT SIDE
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle

                    left_upper = calculate_angle(left_wrist, left_elbow, left_shoulder)
                    left_middle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    left_bottom = calculate_angle(left_hip, left_knee, left_ankle)

                    right_upper = calculate_angle(right_wrist, right_elbow, right_shoulder)
                    right_middle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    right_bottom = calculate_angle(right_hip, right_knee, right_ankle)

                    evaluation = "FAIL"

                    if 167 < left_upper < 190 and 170 < left_middle < 190 and 170 < left_bottom < 190:
                        if 167 < right_upper < 190 and 170 < right_middle < 190 and 170 < right_bottom < 190:
                            evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def shoulder_estimation2(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    #             RIGHT SIDE
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle

                    left_upper = calculate_angle(left_wrist, left_elbow, left_shoulder)
                    left_middle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    left_bottom = calculate_angle(left_hip, left_knee, left_ankle)

                    right_upper = calculate_angle(right_wrist, right_elbow, right_shoulder)
                    right_middle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    right_bottom = calculate_angle(right_hip, right_knee, right_ankle)

                    evaluation = "FAIL"

                    if 75 < left_upper < 100 and 75 < left_middle < 100 and 170 < left_bottom < 190:
                        if 75 < right_upper < 100 and 75 < right_middle < 100 and 170 < right_bottom < 190:
                            evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def left_knee_extension(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                    # Calculate angle
                    left_knee_estimate = calculate_angle(left_hip, left_knee, left_heel)
                    left_back_estimate = calculate_angle(left_shoulder, left_hip, left_knee)

                    evaluation = "FAIL"
                    if 165 < left_knee_estimate < 190 and 85 < left_back_estimate < 110:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "left_knee_estimate": left_knee_estimate,
                             "left_back_estimate": left_back_estimate})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def right_knee_extension(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             RIGHT SIDE
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                    # Calculate angle
                    right_knee_estimate = calculate_angle(right_hip, right_knee, right_heel)
                    right_back_estimate = calculate_angle(right_shoulder, right_hip, right_knee)

                    evaluation = "FAIL"
                    if 165 < right_knee_estimate < 190 and 85 < right_back_estimate < 110:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "right_knee_estimate": right_knee_estimate,
                             "right_back_estimate": right_back_estimate})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def left_elbow_flexicon(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate angle
                    left_elbow_main = np.abs(calculate_angle(left_wrist, left_elbow, left_shoulder) - 180.0)
                    left_elbow_support = calculate_angle(left_elbow, left_shoulder, left_hip)

                    evaluation = "FAIL"
                    if 133 < left_elbow_main < 156 and 0 < left_elbow_support < 12:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "left_elbow_main": left_elbow_main,
                             "left_elbow_support": left_elbow_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def right_elbow_flexicon(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             RIGHT SIDE
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate angle
                    right_elbow_main = np.abs(calculate_angle(right_wrist, right_elbow, right_shoulder) - 180.0)
                    right_elbow_support = calculate_angle(right_elbow, right_shoulder, right_hip)

                    evaluation = "FAIL"
                    if 133 < right_elbow_main < 156 and 0 < right_elbow_support < 12:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "right_elbow_main": right_elbow_main,
                             "right_elbow_support": right_elbow_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def left_ankle_dorsiflexion(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                    # Calculate angle
                    left_ankle_main = calculate_angle(left_knee, left_ankle, left_foot_index)
                    left_ankle_support = calculate_angle(left_hip, left_knee, left_ankle)

                    evaluation = "FAIL"
                    if 8 < left_ankle_main < 12 and 170 < left_ankle_support < 190:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "left_ankle_main": left_ankle_main,
                             "left_ankle_support": left_ankle_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def right_ankle_dorsiflexion(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             RIGHT SIDE
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                    # Calculate angle
                    right_ankle_main = calculate_angle(right_knee, right_ankle, right_foot_index)
                    right_ankle_support = calculate_angle(right_hip, right_knee, right_ankle)

                    evaluation = "FAIL"
                    if 8 < right_ankle_main < 12 and 170 < right_ankle_support < 190:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "right_ankle_main": right_ankle_main,
                             "right_ankle_support": right_ankle_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def left_ankle_plantar_flexion(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default.
                # we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                    # Calculate angle
                    left_ankle_main = calculate_angle(left_knee, left_ankle, left_foot_index)
                    left_ankle_support = calculate_angle(left_hip, left_knee, left_ankle)

                    evaluation = "FAIL"
                    if 38 < left_ankle_main < 52 and 170 < left_ankle_support < 190:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "left_ankle_main": left_ankle_main,
                             "left_ankle_support": left_ankle_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def right_ankle_plantar_flexion(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             RIGHT SIDE
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                    # Calculate angle
                    right_ankle_main = calculate_angle(right_knee, right_ankle, right_foot_index)
                    right_ankle_support = calculate_angle(right_hip, right_knee, right_ankle)

                    evaluation = "FAIL"
                    if 38 < right_ankle_main < 52 and 170 < right_ankle_support < 190:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse(
            {"status": 200, "result": evaluation, "right_ankle_main": right_ankle_main,
             "right_ankle_support": right_ankle_support})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def left_elbow_extension(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             LEFT SIDE
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate angle

                    left_upper = calculate_angle(left_wrist, left_elbow, left_shoulder)
                    left_middle = calculate_angle(left_elbow, left_shoulder, left_hip)

                    evaluation = "FAIL"

                    if 175 < left_upper < 185 and 0 < left_middle < 8:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse({"status": 200, "result": evaluation, "left_upper": left_upper, "left_middle": left_middle})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})


@csrf_exempt
def right_elbow_extension(request):
    if request.method == 'POST':
        # get the video feed from the frontend
        video = request.FILES.get('video')
        with open('temp_video.webm', 'wb') as f:
            f.write(video.read())

        cap = cv2.VideoCapture('temp_video.webm')

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            # accuracy currently set to 50%. this line is going to be accessible by 'pose'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Recolor image to RGB. in opencv our image feed is in bgr format by default. we are just reordering the 3 layers
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # opencv accepts img in bgr format

                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    #           Get coordinates

                    #             RIGHT SIDE
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate angle

                    right_upper = calculate_angle(right_wrist, right_elbow, right_shoulder)
                    right_middle = calculate_angle(right_elbow, right_shoulder, right_hip)

                    evaluation = "FAIL"

                    if 175 < right_upper < 185 and 0 < right_middle < 8:
                        evaluation = "PASS"

                except:
                    pass
                break

        return JsonResponse(
            {"status": 200, "result": evaluation, "right_upper": right_upper, "right_middle": right_middle})
    else:
        return JsonResponse({"status": 400, "msg": 'Invalid request method'})
