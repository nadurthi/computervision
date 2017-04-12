import cv2
import os
import json

class DrawBoundingBox(object):
    def __init__(self,img):
        self.img=img
        self.drawing=False
        self.box=[]

    def getboundingbox(self):
        return self.box

    def resetbox(self):
        self.drawing=False
        self.box=[]


    def __call__(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing=True
            self.box.append( [(x,y),()] )
            print self.box
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.rectangle(self.img,self.box[-1][0],(x,y),(0,255,0),1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.box[-1][1]=(x,y)
            print self.box
            cv2.rectangle(self.img,self.box[-1][0],self.box[-1][1],(0,255,0),1)

class Imagefiles(object):

    def __init__(self,folder='.',savefile='boundingbox.json'):
        self.folder=folder
        self.savefile=savefile
        self.data={}
        self.files=os.listdir(self.folder)
        self.files=[ff for ff in self.files if os.path.isfile(ff) and ('.jpg' in ff  or '.jpeg' in ff or '.png' in ff) ]
        self.N=len(self.files)
        self.n=self.N
        print self.files

        if os.path.isfile(self.savefile)==False:
            with open(self.savefile,'w') as savefile:
                json.dump({},savefile)

    def __iter__(self):
        return self
    
    def next(self):
        if self.n==0:
            raise StopIteration
        else:
            self.n=self.n-1
            print "working on image = ",self.files[self.n]

            img = cv2.imread(self.files[self.n],1)
            # cv2.rectangle(img,(12,26),(136,121),(0,255,0),2)
            cv2.namedWindow('image')
            draw_rect=DrawBoundingBox(img)
            cv2.setMouseCallback('image',draw_rect)

            while(1):
                cv2.imshow('image',img)
                k = cv2.waitKey(1) & 0xFF
                if k!=255:
                    print k
                if k==114:
                    img = cv2.imread(self.files[self.n],1)
                    cv2.namedWindow('image')
                    draw_rect=DrawBoundingBox(img)
                    cv2.setMouseCallback('image',draw_rect)

                if k==115:
                    box=draw_rect.getboundingbox()
                    print "saving bounding box data"
                    self.data[self.files[self.n]]=box
                    with open(self.savefile,'r') as savefile:
                        data=json.load(savefile)

                    if self.files[self.n] in data:
                        print "image ",self.files[self.n], " in data, appending"
                        for bb in box:
                            bb=[list(b) for b in bb]
                            if bb not in data[self.files[self.n]]:
                                data[self.files[self.n]].append(bb)
                        # data[self.files[self.n]]=data[self.files[self.n]]+box
                    else:
                        data.update({self.files[self.n]:box})

                    with open(self.savefile,'w') as savefile:
                        json.dump(data,savefile)


                if k == 27:
                    break

            cv2.destroyAllWindows()


if __name__=='__main__':
    print "ok"
    F=Imagefiles(savefile='boundingbox.json')


    for f in F:
        print "working on f"



