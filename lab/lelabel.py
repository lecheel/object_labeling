import glob
import csv
import cv2
import os
import sys
import argparse
import zmq
import threading
import time
import watcher
import shutil


boxes = []
emit_imgName = ""
gRefresh = 0

def thread_sub_lab():
    ctx_s = zmq.Context()
    sock_s = ctx_s.socket(zmq.SUB)
    sock_s.connect("tcp://localhost:5209")
    print("sub tcp://localhost:5209")
    sock_s.subscribe("") # Subscribe to all topics
    global gRefresh
    while True:
        msg = sock_s.recv_string()
        #print("%s" % msg.rstrip())
        if not os.path.exists(emit_imgName+"_"):
            if os.path.exists(emit_imgName):
                shutil.copy(emit_imgName,emit_imgName+"_")

        f = open(emit_imgName, "w")
        f.write(msg.rstrip())
        f.close()
        gRefresh=1
    sock_s.close()

class ImgLabel(object):
    def __init__(self, data_path):
        """
        """
        self.data_path = data_path
        self.label = []
        self.icons = {}
        classTxt=data_path+"/classes.txt"
        if (os.path.exists(classTxt)):
            txt_file = open(classTxt, "r")
            self.label = [line.rstrip() for line in txt_file]
        self.maxClass = len(self.label)

        idx=0
        for it in self.label:
            icon_name="./icons/"+it.lower()+".jpg"
            if (os.path.exists(icon_name)):
                icon = cv2.imread(icon_name)
                #icon2 = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
            else:
                icon = cv2.imread("./icons/na.jpg")
            #self.icons[it.lower()]=icon
            self.icons[idx]=icon
            idx=idx+1


        #self.niva = cv2.imread("./niva.jpg")
        #self.nivaIcon = cv2.cvtColor(self.niva, cv2.COLOR_BGR2BGRA)

        self.fw = 0
        self.fh = 0
        self.addicon = False
        self.select = 0
        self.xa = {}
        self.img = []
        self.lastidx = 9999
        self.line = []
        self.refresh = 0
        self.drawing = 0
        self.ix = -1
        self.iy = -1
        self.cx = -1
        self.cy = -1
        self.toggleIcon = 1
        self.mousePress = False
        self.showHelp = 0
        self.imgName = ""

    def addIcon(self,frame,icon,x1,y1):
        #print(icon)
        iw=frame.shape[1]
        ih=frame.shape[0]
        #print(iw,ih,x1,y1,icon.shape)
        if (iw<200)|(ih<200):
            icon = cv2.resize(icon, (24,24), interpolation = cv2.INTER_AREA)
        if ((ih-y1)<100):
            y1=10
        if ((iw-x1)<113):
            x1=0
        frame[y1:y1+icon.shape[0], x1:x1+icon.shape[1]]=icon
        return frame

    def addIconResize(self,frame,icon,x1,y1):
        #print(icon)
        iw=frame.shape[1]
        ih=frame.shape[0]
        dim = (32, 32)
        resized = cv2.resize(icon, dim, interpolation = cv2.INTER_AREA)
        if (x1<0):
            x1=0
        if (x1>iw):
            x1=iw
        if (y1<0):
            y1=0
        if (y1>ih):
            y1=ih
        frame[y1:y1+resized.shape[0], x1:x1+resized.shape[1]]=resized
        return frame

    def CurrSel(self,frame):
        if (self.select < self.maxClass):
            self.addIcon(frame,self.icons[self.select],10,100)
            cv2.rectangle(frame,(5,95),(80,165),(0,255,0),2)
            msg ="%s %d" %(self.label[self.select],self.select)
            cv2.putText(frame,msg,(100,150), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2,cv2.LINE_AA)


    def Help(self,frame):
        if (self.showHelp):
            help = cv2.imread("./icons/help.png")
            iw=frame.shape[1]
            ih=frame.shape[0]
            x1 = int((iw-help.shape[1])/2)
            y1 = int((ih-help.shape[0])/2)
            frame[y1:y1+help.shape[0], x1:x1+help.shape[1]]=help
            cv2.putText(frame,"w   -- Toggle Draw and Label mode",(x1+10,y1+180), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,"a d -- Perv/Next Image",(x1+10,y1+210), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,"z x -- Perv/Next Selection Label",(x1+10,y1+240), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,"o   -- Toggle Label icon",(x1+10,y1+270), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,"i   -- Toggle Symbol in left side",(x1+10,y1+300), cv2.FONT_HERSHEY_SIMPLEX,1.0,(10,10,10),1,cv2.LINE_AA)
            cv2.putText(frame,"=  -- auto label from yolo",(x1+10,y1+330), cv2.FONT_HERSHEY_SIMPLEX,1.0,(10,10,10),1,cv2.LINE_AA)
            cv2.putText(frame,"-  -- Toggle label",(x1+10,y1+360), cv2.FONT_HERSHEY_SIMPLEX,1.0,(10,10,10),1,cv2.LINE_AA)

            return frame

    def AddInfo(self,frame):
        if (self.drawing):
            msg ="Box"
        else:
            msg ="Label"
        cv2.putText(frame,msg,(5,90), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2,cv2.LINE_AA)

        if (self.addicon):
            for i in range(0,self.maxClass):
                self.addIcon(frame,self.icons[i],10,200+i*80)

    def pointInRect(self,point,rect):
        x1, y1, w, h = rect
        x2, y2 = x1+w, y1+h
        x, y = point
        if (x1 < x and x < x2):
            if (y1 < y and y < y2):
                return True
        return False

    def checkObj(self,info,fname,wh,pt,pat,delmode):
        msg=""
        for it in info:
            it=it.rstrip()
            data=it.split(" ")
            cidx=int(data[0])

            x=float(data[1])
            y=float(data[2])
            w=float(data[3])
            h=float(data[4])

            x,y,w,h=self.convertCenter(x,y,w,h,wh[0],wh[1])
            box=[x,y,w,h]
            if(self.pointInRect(pt,box)):
                #print("True",pt,box)
                data[0]=pat
                if not delmode:
                    msg=msg+"{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(data[0],float(data[1]),float(data[2]),float(data[3]),float(data[4]))
            else:    
                msg=msg+"{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(data[0],float(data[1]),float(data[2]),float(data[3]),float(data[4]))
        msg = os.linesep.join([s for s in msg.splitlines() if s])
        f = open(fname, "w")
        f.write(msg)
        f.close()
        self.refresh = 1

    def writeObj(self,info,fname,wh,box,content):
        msg=""
        for it in info:
            it=it.rstrip()
            data=it.split(" ")
            cidx=int(data[0])

            x=float(data[1])
            y=float(data[2])
            w=float(data[3])
            h=float(data[4])

            x,y,w,h=self.convertCenter(x,y,w,h,wh[0],wh[1])
            box=[x,y,w,h]
            msg=msg+"{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(data[0],float(data[1]),float(data[2]),float(data[3]),float(data[4]))
        msg=msg+content
        #print(msg)
        f = open(fname, "w")
        f.write(msg)
        f.close()
 

    def on_mouse(self,event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.checkObj(params[0]["line"],params[0]["img"],params[0]["wh"],[x,y],"0",False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.checkObj(params[0]["line"],params[0]["img"],params[0]["wh"],[x,y],"1",False)

    def on_mouse_rds(self,event, x1, y1, flags, params):
        if (self.drawing):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mousePress = True
                self.ix,self.iy = x1,y1
            elif event == cv2.EVENT_LBUTTONUP:
                self.mousePress = False
                box=[self.ix,self.iy,x1,y1]
                if ((abs(x1-self.ix)>8) & (abs(y1-self.iy)>8)):
                    yolo_line = self.get_object_params(box,params[0]["wh"])
                    #key=self.label[self.select].lower()
                    #tag ="%d" %(list(self.icons.keys()).index(key))
                    msg="{} {:.5f} {:.5f} {:.5f} {:.5f}".format(self.select,yolo_line[0],yolo_line[1],yolo_line[2],yolo_line[3])
                    self.writeObj(params[0]["line"],params[0]["img"],params[0]["wh"],box,msg)
                self.refresh=1

            elif event == cv2.EVENT_MOUSEMOVE:
                if (self.mousePress):
                    self.cx=x1
                    self.cy=y1
                self.refresh=1
            if event == cv2.EVENT_RBUTTONDOWN:
                msg ="%d" %(self.select)
                self.checkObj(params[0]["line"],params[0]["img"],params[0]["wh"],[x1,y1],msg,True)
               
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                msg ="%d" %(self.select)
                self.checkObj(params[0]["line"],params[0]["img"],params[0]["wh"],[x1,y1],msg,False)

            if event == cv2.EVENT_RBUTTONDOWN:
                msg ="%d" %(self.select)
                self.checkObj(params[0]["line"],params[0]["img"],params[0]["wh"],[x1,y1],msg,True)

    def get_object_params(self,box, size):
        image_width = 1.0 * size[0]
        image_height = 1.0 * size[1]

        absolute_x = box[0] + 0.5 * (box[2] - box[0])
        absolute_y = box[1] + 0.5 * (box[3] - box[1])

        absolute_width = box[2] - box[0]
        absolute_height = box[3] - box[1]

        x = absolute_x / image_width
        y = absolute_y / image_height
        width = absolute_width / image_width
        height = absolute_height / image_height

        return x, y, width, height

    def convertCenter(self,x,y,w,h,iw,ih):
        x = x-(w/2)
        y = y-(h/2)
        x=int(x*iw)
        y=int(y*ih)
        w=int(w*iw)
        h=int(h*ih)
        return (x,y,w,h)

    def to_red(self,info,fname):
        f = open(fname, "w")
        for it in info:
            data=it.split(" ")
            data[0] = "1"
            for item in data:
                f.write("%s " % item)
            f.write("\n")
        f.close()

    def to_green(self,info,fname):
        f = open(fname, "w")
        for it in info:
            data=it.split(" ")
            data[0] = "0"
            for item in data:
                f.write("%s " % item)
            f.write("\n")
        f.close()

    def trlight_red_green(self,xx,idx,line,txtName):
        if xx==49:   # 0 - RED
            self.to_red(line,txtName)

        if xx==48:   # 1 - GREEN
            self.to_green(line,txtName)

        if (xx==81) | (xx==97):   # left, a
            idx=idx-1
        if (xx==83) | (xx==100):  # right, d
            idx=idx+1
        return(idx)

    def ImageLoop(self,idx,items,xb,vid_title):
        tColor= [(0,0,255),(0,255,0),(255,0,0),(127,127,0),(0,0,0),(127,0,127),(100,100,100)]
        del xb[:]
        xa={}
        txt=items[idx][:-3]+"txt"
        global gRefresh
        if ((self.lastidx != idx) | (self.refresh==1) | (gRefresh==1)):
            self.refresh=0
            gRefresh=0
            img_name=items[idx]
            img=cv2.imread(img_name)
            ih,iw,c=img.shape
            if os.path.exists(txt):
                text_file = open(txt, "r")
                lines = text_file.read()
                line=lines.splitlines()
                xa["line"]=line
                xa["img"]=txt
                xa["wh"]=[iw,ih]
                self.xa = xa
                xb.append(xa)
            else:
                line=""
            self.line = line
            self.img = img
            self.lastidx = idx
        else:
            img = self.img
            ih,iw,c=img.shape
            xa=self.xa
            xb.append(xa)
            line = self.line

        for item in line:
            data=item.split(" ")
            cidx=int(data[0])

            x=float(data[1])
            y=float(data[2])
            w=float(data[3])
            h=float(data[4])

            x,y,w,h=self.convertCenter(x,y,w,h,iw,ih)

            #cv2.rectangle(img,(x,y),(x+w,y+h),tColor[cidx],2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(img,str(cidx),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1.0,tColor[cidx],2,cv2.LINE_AA)
            if (cidx < self.maxClass):
                if self.toggleIcon:
                    self.addIconResize(img,self.icons[cidx],x-20,y-20)
        if (self.mousePress):
            cv2.rectangle(img,(self.ix,self.iy),(self.cx,self.cy),(0,255,0),2)
        cv2.putText(img,txt,(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
        self.AddInfo(img)
        self.CurrSel(img)
        self.Help(img)
        cv2.imshow(vid_title,img)
        return(idx,xb,line,txt)

    def LabelImgTrlight(self,dir):
        vid_title="LabelImgTrlight"
        cv2.namedWindow(vid_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(vid_title, 1280, 720)
        xb=[]
        cv2.setMouseCallback(vid_title, self.on_mouse,param=xb)
        items=sorted(glob.glob(dir+"/*.jpg"))
        idx=0
        max=len(items)
        while(1):
            idx,xb,line,txtName=self.ImageLoop(idx,items,xb,vid_title)
            xx=cv2.waitKey(100)
            #print(xx)
            if xx & 0xFF == ord('q'):
                break
            idx=self.trlight_red_green(xx,idx,line,txtName)
            if xx==194:   # F5
                print("F5")
            if idx==max:
                idx=0

    def LabelImgRds(self,dir):
        vid_title="LabelImgRds"
        cv2.namedWindow(vid_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(vid_title, 1280, 720)
        xb=[]
        cv2.setMouseCallback(vid_title, self.on_mouse_rds,param=xb)
        items=sorted(glob.glob(dir+"/*.jpg"))
        idx=0
        max=len(items)
        if max < 1:
            print("{} folder is empty".format(dir))
            return

        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock_pub = ctx.socket(zmq.PUB)
        sock_pub.bind("tcp://*:5208")


        while(1):
            idx,xb,line,txtName=self.ImageLoop(idx,items,xb,vid_title)
            xx=cv2.waitKey(60)
            #print(xx)
            if xx & 0xFF == ord('='):
                msg= os.getcwd()+"/"+txtName[:-3]+"jpg"
                self.imgName=msg[:-3]+"txt"
                global emit_imgName
                emit_imgName=self.imgName
                print("lab {}".format(msg))
                sock_pub.send_string(msg)
                time.sleep(0.2)
                self.refresh = 1
            if xx & 0xFF == ord('-'):
                msg= os.getcwd()+"/"+txtName[:-3]+"jpg"
                self.imgName=msg[:-3]+"txt"
                if (os.path.exists(self.imgName+"_")):
                    shutil.move(self.imgName,self.imgName+"+")
                    shutil.move(self.imgName+"_",self.imgName)
                    shutil.move(self.imgName+"+",self.imgName+"_")
                    self.refresh = 1

            if xx & 0xFF == ord('w'):
                if (self.drawing):
                    self.drawing = False
                else:
                    self.drawing = True
                self.refresh = 1

            if xx & 0xFF == ord('o'):
                if (self.toggleIcon):
                    self.toggleIcon = False
                else:
                    self.toggleIcon = True
                self.refresh = 1


            if xx & 0xFF == ord('h'):
                if (self.showHelp):
                    self.showHelp = False
                else:
                    self.showHelp = True
                self.refresh = 1

            if xx & 0xFF == ord('i'):
                if (self.addicon):
                    self.addicon = False
                else:
                    self.addicon = True
                self.refresh = 1

            if (xx == 82) | (xx == 91) | (xx == 122):
                if (self.select>0):
                    self.select = self.select-1
                else:
                    self.select = self.maxClass-1
                self.refresh = 1
            if (xx == 84) | (xx == 93) | (xx == 120):
                if (self.select<self.maxClass-1):
                    self.select = self.select +1
                else:
                    self.select = 0
                self.refresh = 1

            if xx & 0xFF == ord('q'):
                break
            idx=self.trlight_red_green(xx,idx,line,txtName)
            if xx==194:   # F5
                print("F5")
            if idx==max:
                idx=0

def LabelVideo(video_name):
    if not (os.path.exists("./cap")):
            os.mkdir("cap")
    prefix=video_name.split("/")[-1][:-4]
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: %4.2f %d" % (fps,amount_of_frames))
    #cap.set(cv2.CAP_PROP_POS_FRAMES,frame_seq);
    vid_title="XXX"
    cv2.namedWindow(vid_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(vid_title, 1280, 720)
    idx=0

    while(1):
        ret, vframe = cap.read()
        if ret == False:
            continue
        idx=idx+1
        img=vframe.copy()
        myName="%d" % idx
        cv2.putText(vframe,myName,(60,60), cv2.FONT_HERSHEY_SIMPLEX,2.1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow(vid_title,vframe)
        xx=cv2.waitKey(33)
        if xx==61:   # =
            print("=")
        if xx==32:   # SPACE
            while(1):
                aa=cv2.waitKey(33)
                if aa==32:
                    break
                elif aa==112:
                    jname = "./cap/%s_%05d.jpg" % (idx,prefix)
                    print("%s OK" % jname)
                    cv2.imwrite(jname, img)
                    break

        elif xx==82:  # up arrow
            frame_seq = (idx*fps)
            jump=idx+3*fps
            idx=idx+3*fps
            cap.set(cv2.CAP_PROP_POS_FRAMES,jump);
            #print(idx)
        elif ((xx==83) | (xx == 100)): # right
            frame_seq = (idx*fps)
            jump=idx+30*fps
            idx=idx+30*fps
            cap.set(cv2.CAP_PROP_POS_FRAMES,jump);
        elif (xx==81) | (xx == 97): # left
            frame_seq = (idx*fps)
            if (idx>30*fps):
                jump=idx-30*fps
                idx=idx-30*fps
                cap.set(cv2.CAP_PROP_POS_FRAMES,jump);
            else:
                print("Error!!!")
        elif xx==112:  # p -- Print
            jname = "./cap/%s_%05d.jpg" % (prefix,idx)
            print("%s OK" % jname)
            cv2.imwrite(jname, img)

        elif xx==84:  # down arrow
            frame_seq = (idx*fps)
            if (idx>3*fps):
                jump=idx-3*fps
                idx=idx-3*fps
                cap.set(cv2.CAP_PROP_POS_FRAMES,jump);

        if xx & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def LabelImg(Name):
    xx=watcher.Watcher()
    x=threading.Thread(target=thread_sub_lab, args=()).start()
    lab = ImgLabel(data_path=Name)
    lab.LabelImgRds(Name)
    xx.kill()

parser = argparse.ArgumentParser(description="A simple utility to Select PNG from Video.")
parser.add_argument("--input", dest="mp4_path", help="Path to the MP4 file")
parser.add_argument("--label", help="labelImg", action="store_true")
args = parser.parse_args()
if args.mp4_path:
    if args.label:
        LabelImg(args.mp4_path)
    else:
        LabelVideo(args.mp4_path)

