import streamlit as st
import streamlit.components.v1 as components
import tempfile
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image


def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results

def draw_box(frame, labels, net, ln):
    results = detect_people(frame, net, ln, personIdx=labels.index("person"))

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    if len(results)>=10:
        color = (255, 0, 0)

    text = "Crowd Count: {}".format(len(results))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    return frame

def for_video(vs, labels, net, ln):
    
    stframe = st.empty()
    f_count=0
    p_count=0
    count_list=[]

    output = cv2.VideoWriter('sd_output.mp4',cv2.VideoWriter_fourcc(*'X264'), 3, (700,400))

    while True:
                
        (grabbed, frame) = vs.read()
        
        f_count+=1

        if not grabbed:
            break
        
        if f_count%10!=0:
            continue

        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=labels.index("person"))
        
        if len(results)>8:
            count_list.append(len(results))
        else:
            count_list.clear()
        
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
        
        if len(results)>=10:
            color = (0,0,255)
        else:
            color = (0,255,0)
        
        
        if len(count_list)>=10:
            p_count+=1
            text = "Alert: Getting Crowded, Please Spread Out"
            color=(0,0,255)
            
            if p_count==8:
                p_count=0
                count_list.clear()
        else:
            text = "Crowd Count: {}".format(len(results))
            
        
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        final_img = cv2.resize(frame, (700,400))
        output.write(final_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break
    vs.release()
    output.release()

def for_image(img, labels, net, ln):
    img = Image.open(img)
    frame = np.array(img)
    frame = imutils.resize(frame, width=700)
    frame = draw_box(frame, labels, net, ln)
    st.image(frame, caption='Final Output', width=700)

def model_load():
    labels = None
    with open('C:\\Users\\Sanmit\\Desktop\\Projects\\Mask_SD\\models\\coco.names', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    weightsPath = 'C:\\Users\\Sanmit\\Desktop\\Projects\\Mask_SD\\models\\yolov3.weights'
    configPath = 'C:\\Users\\Sanmit\\Desktop\\Projects\\Mask_SD\\models\\yolov3.cfg'
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return labels, net, ln


def app():

    labels, net, ln = model_load()

    st.title('Social Distancing')

    st.subheader("How do you want to Upload a Video/Image")

    option=''
    option = st.selectbox('Options', options=('Choose an option from this list', 'Upload Image from Device', 'Upload Video from Device', 'Take Pic from Camera', 'Live Video from Camera'))

        
    stframe = st.empty()
    f=''
    img=''
    g=0

    if option=='Upload Video from Device':
        g=0
        f = st.file_uploader("Upload file")
        if st.button('Upload Video'):
            with st.spinner('Wait for it...'):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                if f:
                    tfile.write(f.read())
                vs = cv2.VideoCapture(tfile.name)
                for_video(vs, labels, net, ln)
                video_file = open('sd_output.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                g=1

    elif option=='Take Pic from Camera':
        g=0
        img = st.camera_input('Click a Pic')
        if st.button('Upload Image'):
            with st.spinner('Wait for it...'):
                for_image(img, labels, net, ln)
                g=1


    elif option=='Upload Image from Device':
        g=0
        img = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if st.button('Upload Image'):
            with st.spinner('Wait for it...'):
                for_image(img, labels, net, ln)
                g=1

    elif 'Live Video from Camera':
        g=0
        if st.button('Start Camera'):
            vs1 = cv2.VideoCapture(0)
            for_video(vs1, labels, net, ln)
            g=1

    else:
        g=0
        st.write()

    if g:
        with st.expander('Techniques, Models, Algorithms Used:'):
            st.write('Computer Vision')
            st.write('')

if __name__ == '__main__':
    main()
