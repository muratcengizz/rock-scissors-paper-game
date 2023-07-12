from ultralytics import YOLO 
import os
import cv2 

model = YOLO("best.pt")

path = os.chdir("C:/Users/murat/Documents/computer_vision3/rock_paper_scissors/test/images")
files = os.listdir(path)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


hand1_score, hand2_score = 0, 0
dir = 0

while True:
    
    retval, img = video.read()
    img = cv2.flip(src=img, flipCode=1)
    #print(dir)
    predict = model.predict(img)
    #img = predict[0].plot()
    for pred in predict:
        boxes = pred.boxes.cpu().numpy()
        for box in boxes:
            
            if len(boxes) == 2:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                hand1_x1, hand1_y1, hand1_x2, hand1_y2 = boxes[0].xyxy[0].astype(int)
                hand1_name = pred.names[int(boxes[0].cls[0])]
                
                conf1, conf2 = 0, 0
                sayac = 0
                for element in box.conf:
                    sayac += 1
                    if sayac == 1:
                        conf1 = element
                        sayac += 1
                    if sayac == 2:
                        conf2 = element
                
                hand2_x1, hand2_y1, hand2_x2, hand2_y2 = boxes[1].xyxy[0].astype(int)
                hand2_name = pred.names[int(boxes[1].cls[0])]
                
                
                cv2.rectangle(
                    img=img,
                    pt1=(hand1_x1, hand1_x2),
                    pt2=(hand1_x2, hand1_y2),
                    color=(0, 0, 255),
                    thickness=3
                )
                
                cv2.rectangle(
                    img=img,
                    pt1=(hand2_x1, hand2_y1),
                    pt2=(hand2_x2, hand2_y2),
                    color=(0, 0, 255),
                    thickness=3
                )
                
                cv2.putText(img=img,
                            text=f"{hand1_name}  {conf1:.2f}",
                            org=(hand1_x1, hand1_y1-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.6,
                            color=(0, 0, 255),
                            thickness=2)
                
                cv2.putText(img=img,
                            text=f"{hand2_name}   {conf2:.2f}",
                            org=(hand2_x1, hand2_y1-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.6,
                            color=(0, 0, 255),
                            thickness=2)
                
                
                """
                cv2.rectangle(img=img,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=(0, 0, 255),
                            thickness=2)
                """
                if dir == 0:
                    dir = 1
                    if hand1_name == "Rock":
                        if dir == 0:
                            if hand2_name == "Scissors":
                                hand1_score += 1
                                
                            elif hand2_name == "Paper":
                                hand2_score += 1
                                
                    elif hand1_name == "Paper":
                        if dir == 1:
                            if hand2_name == "Rock":
                                hand1_score += 1
                                
                            elif hand2_name == "Scissors":
                                hand2_score += 1
                                
                                
                    elif hand1_name == "Scissors":
                        if dir == 0:
                            if hand2_name == "Rock":
                                hand2_score += 1
                                
                            elif hand2_name == "Paper":
                                hand1_score += 1   
                                        
                    
                
    cv2.putText(img=img,
                text=f"Hand1 Score: {hand1_score}  | Hand2 Score: {hand2_score}",
                org=(30, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(0, 0, 255),
                thickness=3)
        
    
    """
    for file in files:
        img = cv2.imread(filename=file)
        
        predict = model.predict(img)
        plotted = predict[0].plot()
        
        cv2.imshow(winname="detection", mat=plotted)
        if cv2.waitKey(0) == ord("q"): continue
    """
    
    cv2.imshow(winname="detection", mat=img)
    if cv2.waitKey(1) == ord("q"): break
    
video.release()
cv2.destroyAllWindows()
