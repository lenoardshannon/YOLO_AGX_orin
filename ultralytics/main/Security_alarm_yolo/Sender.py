from time import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator,colors


# Email and function
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
# 18161938566@163.com
# leonard.zhang@scania.com
# Save creation and authentication
password = "oebd exio saoh mymd"
from_email = "leonardy20b@gmail.com"  # must match the email used to generate the password
to_email = "joy.ding@scania.com.cn"  # receiver email

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(from_email, password)

def send_email(to_email,from_email,people_detected=1,img=None):
    ''' Sends an email notifiaction indicating the number of objects detected
    defaults to 1 object '''
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"

    # Image captured
    if img is not None:
        image_data = img.tobytes()
        msg_img = MIMEImage(image_data) # constructing image object
        msg_img.add_header("Captured Image",'attachment',filename="Index.jpg")
        # Add image attachment
        message.add_header('Content-ID','<0>')
        message.attach(msg_img)

    # Add in the message body
    message_body = f"ALERT - {people_detected} person with mask has been detected ! "
    message.attach(MIMEText(message_body, 'plain'))

    server.sendmail(from_email, to_email, message.as_string())

'''--------------------------------------------------------------------------------'''
class ObjectDetection:
    def __init__(self,capture_index):
        """Initilize an object detection instance with a given camera index."""
        self.capture_index = capture_index
        self.email_sent =False

        # model information
        self.model = YOLO("/home/leonardzhang/ultralytics/PPE.pt")

        # Visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = '0,1' if torch.cuda.is_available() else 'cpu'

    def predict(self,im0):
        """Run predicton using YOLO model for the input image im0"""
        results = self.model(im0)
        return results

    def display_fps(self,im0):
        """Displays the FPS on an image 'im0' by calculating and overlayding as white text on a black rectangle"""
        self.end_time = time()

        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS:{int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap,70 - text_size[1]- gap),
            (20 + text_size[0]+ gap,70 + gap),
            (255,255,255),
            -1,
        )
        cv2.putText(im0,text,(20,70),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,0),2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
         # xyxys=[]
        # confidences = []
        # class_ids = []
        # # Extract detections for person class
        # for results in results[0]:
        #     class_id = results.boxes.cls.cpu().numpy().astype(int)
        #     # COCO dataset id=0 means person
        #     if class_id  == 0:
        #         xyxys.append(results.boxes.xyxys.cpu().numpy())
        #         confidences.append(results.boxes.conf.cpu().numpy())
        #         class_ids.append(results.boxes.classes.cpu().numpy().astype(int))
        # # Setup detection for visualization
        # detections = sv.Detections.from_ultralytics(results[0])
        #
        # frame = self.box_annotator.annotate(scene=frame,detections=detections)
        #
        # return frame, class_ids

        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names

        for box, cls in zip(boxes, clss):
            if cls == 1:
                class_ids.append(cls)
                self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        """Run object detection on video frames from a camera stream, plotting and showing the results."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)
            
            _,encoded_img = cv2.imencode('.jpg',im0)
            
            if len(class_ids) > 0:  # Only send email If not sent before
                if not self.email_sent:
                    send_email(to_email, from_email, len(class_ids),encoded_img)
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow("YOLO8 Detection", im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()


detector = ObjectDetection(0)
detector()



