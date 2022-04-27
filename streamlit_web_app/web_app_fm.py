import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tempfile
import imutils
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
import time

def face_detection(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces

def mask_prediction(crop, model):
    ans = {0: 'Mask', 1: 'No Mask'}
    img1 = crop
    img1 = cv2.resize(img1, (100, 100))
    img1 = np.array(img1).reshape((1, 100, 100, 3))
    Y_pred = model.predict(img1)
    y_pred = np.argmax(Y_pred, axis=1)
    text = ans[y_pred[0]]
    return text

def final_func(faces, img, final_img, model):
    color = {'Mask': (0, 255, 0), 'No Mask': (255, 0, 0)}
    for i in faces:
        if i['confidence'] > 0.92:
            crop = img[i['box'][1]:i['box'][1]+i['box'][3],
                       i['box'][0]:i['box'][0]+i['box'][2], :]
            if crop.shape[0]+crop.shape[1] > 185:
                text = mask_prediction(crop, model)
                cv2.putText(final_img, text, (i['box'][0], i['box'][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color[text], 6)
                cv2.rectangle(final_img, (i['box'][0], i['box'][1]), (i['box']
                              [0]+i['box'][2], i['box'][1]+i['box'][3]), color[text], 6)

    return final_img

def for_image(img, model):
    img = Image.open(img)
    img = np.array(img)
    img = imutils.resize(img, width=3000)

    final_img = img
    faces = face_detection(img)
    answer = final_func(faces, img, final_img, model)

    st.image(answer, caption='Final Output', width=700)

def for_video(vs, model, live):
    count = 0
    stframe = st.empty()

    output = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'H264'), 3, (700,400))

    while True:
        (grabbed, img) = vs.read()
        count += 1

        if count % 20 != 0:
            continue
        if not grabbed:
            break

        img = imutils.resize(img, width=3000)
        final_img = img
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)


        faces = face_detection(img)
        final_img = final_func(faces, img, final_img, model)
        
        
        if live:
            final_img = imutils.resize(final_img, width=700)
            stframe.image(final_img, caption='Final Output', width=700)
        else:
            final_img = cv2.resize(final_img, (700,400))
            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            output.write(final_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    vs.release()
    output.release()

@st.cache
def model_load():
    return load_model('C:\\Users\\Sanmit\\Desktop\\Projects\\Mask_SD\\models\\face_mask.h5')

def app():
    st.title('Face Mask Detection')

    model = model_load()

    st.subheader("How do you want to Upload a Video/Image")

    option = ''
    option = st.selectbox('Options', options=('Choose an option from this list', 'Upload Image from Device', 'Upload Video from Device', 'Take Pic from Camera', 'Live Video from Camera'))

    img = ''
    g=0

    if option == 'Upload Image from Device':
        g=0
        img = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
        if st.button('Upload Image'):
            with st.spinner('Wait for it...'):
                for_image(img, model)
                g=1

    elif option == 'Upload Video from Device':
        g=0
        f = st.file_uploader("Upload Video")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        if st.button('Upload Video'):
            with st.spinner('Wait for it...'):
                tfile.write(f.read())
                vs2 = cv2.VideoCapture(tfile.name)
                for_video(vs2, model,0)
                video_file = open('output.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                g=1

    elif option == 'Live Video from Camera':
        g=0
        if st.button('Start Camera'):
            vs3 = cv2.VideoCapture(0)

            for_video(vs3, model,1)
            g=1

    elif option == 'Take Pic from Camera':
        g=0
        img = st.camera_input('Click a Pic')
        if st.button('Upload Image'):
            with st.spinner('Wait for it...'):
                for_image(img, model)
                g=1

    else:
        g=0
        st.write()

    if g:
        st.success('Green Box: Mask Present')
        st.error('Red Box: No Mask Present')
        st.write()
        with st.expander('Techniques, Models, Algorithms Used:'):
            st.write('Deep Learning')
            st.write('VGG16')
            st.write('MTCNN')

if __name__ == '__main__':
    main()
