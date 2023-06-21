import cv2

output_path = 'output/'
emotion_data_path = output_path + 'emotion_data/'

# opencv自带的一个面部识别分类器
detection_model_path = 'models/haarcascade_frontalface_default.xml'

# 表情模型
emotion_model = 'models/resnet_model.pt'

# resource
background_image = 'resource/star_eyes.png'
start_icon = 'resource/start.png'
stop_icon = 'resource/stop.png'
select_image_button = 'resource/button-image.png'
main_icon = 'resource/main_icon.png'

font = cv2.FONT_HERSHEY_SIMPLEX

# 表情标签
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

frame_window = 10
emotion_window = []
