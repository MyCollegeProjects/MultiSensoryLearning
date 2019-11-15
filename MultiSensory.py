from Tkinter import *
import ImageTk
import cv2
import Image
import threading
import speech
import sys

root = Tk()
root.wm_title("Machine Learning")
menu = Menu(root)
root.config(menu=menu)
canvas = Canvas(root, width=640, height=480)
canvas.grid(row=3, column=0)
cap = cv2.VideoCapture(1)
q = None
faceLen = None

def saveFaces(p):
	global faceLen
	if faceLen > 0:
		cv2.imwrite("faces/"+p+".jpg", q)
	else:
		cv2.imwrite("objects/"+p+".jpg", q)

def callback(phrase, listener):
    print ": %s" % phrase
    if phrase != "":
    	saveFaces("jam")
    if phrase == "turn off":
        speech.say("Goodbye.")
        listener.stoplistening()
        sys.exit()

listener = speech.listenforanything(callback)

def drawCanvasImage():
#	cap = cv2.VideoCapture(1)
	global faceLen
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	while(True):
		global q
		ret, frame = cap.read()
		image = frame
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.5,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		faceLen = len(faces)
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

		imgObj = Image.fromarray(image)
		canImg = ImageTk.PhotoImage(imgObj)
		canvas.create_image(0, 0, image=canImg, anchor='nw')
		q = image
		# cv2.imshow("frame", image)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break

	cap.release()
	cv2.destroyAllWindows()

fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Exit")

KBMenu = Menu(menu)
menu.add_cascade(label="Knowledge Base", menu=KBMenu)
KBMenu.add_command(label="Face DB")
KBMenu.add_command(label="Object DB")

helpMenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpMenu)
helpMenu.add_command(label="About")

micLabel = Label(root, text="Microphone Status")
micLabel.grid(row=0, column=0)

camLabel = Label(root, text="Camera Status")
camLabel.grid(row=0, column=1)

speakLabel = Label(root, text="Speaker Status")
speakLabel.grid(row=0, column=2)

dbLabel = Label(root, text="MongoDB Status")
dbLabel.grid(row=0, column=3)

micBtn = Button(root, text="ON/OFF", bg="GREEN")
micBtn.grid(row=1, column=0)

camBtn = Button(root, text="ON/OFF", bg="GREEN")
camBtn.grid(row=1, column=1)

speakBtn = Button(root, text="ON/OFF", bg="RED")
speakBtn.grid(row=1, column=2)

dbBtn = Button(root, text="ON/OFF", bg="RED")
dbBtn.grid(row=1, column=3)

vid = Label(root, text="Video Percept")
vid.grid(row=2, column=0)

aud = Label(root, text="Audio Sequence")
aud.grid(row=2, column=1)


text = Text(root, height=25, width=50)
text.grid(row=3, column=1)

response = Label(root, text="Response:")
response.grid(row=4, column=0)

msg = Label(root, text="There are two faces found!", bg="GREEN")
msg.grid(row=4, column=1)

t = threading.Thread(target=drawCanvasImage)
t.start()

root.mainloop()
