from __future__ import division
import cv2
import os
import json
import pdb

class DrawBoundingBox(object):
    def __init__(self,img):
        self.img=img
        self.drawing=False
        self.box=[]
        self.cnt=0
    def getboundingbox(self):
        return self.box

    def resetbox(self):
        self.drawing=False
        self.box=[]


    def __call__(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing=True
            self.cnt=0
            self.box.append( [(x,y),()] )
            print self.box
        elif event == cv2.EVENT_MOUSEMOVE:

            if self.drawing == True:
                self.cnt=self.cnt+1
                if self.cnt%10==0:
                    print 'plotting'
                    cv2.rectangle(self.img,self.box[-1][0],(x,y),(0,255,0),1)


        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.box[-1][1]=(x,y)
            print self.box
            cv2.rectangle(self.img,self.box[-1][0],self.box[-1][1],(0,255,0),1)





class Imagefiles(object):

    def __init__(self,folder='.',savefile='boundingbox.json',skipdone=False):
        self.folder=folder
        self.savefile=savefile
        self.data={}
        self.files=os.listdir(self.folder)
        self.files=[ff for ff in self.files if os.path.isfile(ff) and ('.jpg' in ff  or '.jpeg' in ff or '.png' in ff) ]

        original_N=len(self.files)

        if os.path.isfile(self.savefile)==False:
            with open(self.savefile,'w') as savefile:
                json.dump({},savefile)
        else:
            with open(self.savefile,'r') as savefile:
                data=json.load(savefile)

            if skipdone:
                donefiles=data.keys()
                for dd in donefiles:
                    if dd in self.files:
                        self.files.remove(dd)

        self.N=len(self.files)
        self.n=self.N
        print "to do = ",len(self.files)," original = ",original_N

    def getcroppedimages(self,image=None):
        with open(self.savefile,'r') as savefile:
            data=json.load(savefile)
        if os.path.isdir('cropped')==False:
            os.mkdir('cropped')


        if image is None:
            files=self.files
        else:
            files=[image]
            print "save copped image = ", image

        for image in files:
            # pdb.set_trace()
            # print image in data.keys()
            if image in data:
                ss=data[image]
                if len(ss)==0:
                    continue
                img = cv2.imread(image,1)
                for i in range(len(ss)):
                    sa=ss[i]
                    if len(sa[0])==0 or len(sa[1])==0:
                        continue
                    print sa
                    x=sorted([sa[0][0],sa[1][0]])
                    y=sorted([sa[0][1],sa[1][1]])

                    img2=img[ y[0]:y[1], x[0]:x[1] ]
                    # cv2.namedWindow('image_'+str(i) )
                    # cv2.imshow('image_'+str(i) ,img2)
                    print "copped image = ", 'cropped/cropped_'+str(i)+'_'+image
                    cv2.imwrite('cropped/cropped_'+str(i)+'_'+image,img2)

    def __iter__(self):
        return self

    def next(self):
        if self.n==0:
            raise StopIteration
        else:
            self.n=self.n-1
            print "working on image = ",self.files[self.n]
            image=self.files[self.n]
            img = cv2.imread(image,1)
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

                    self.getcroppedimages(self.files[self.n])

                if k == 27 or k==113:
                    break

            cv2.destroyAllWindows()


if __name__=='__main__':
    print "ok"

    jsonfiles=[jsonfiles for jsonfiles in os.listdir('.') if '.json' in jsonfiles and 'boundingbox' in jsonfiles]
    if len(jsonfiles)==0:
        jsonfiles=['boundingbox.json']

    # F=Imagefiles(savefile=jsonfiles[0],skipdone=False)
    # F.getcroppedimages()

    F=Imagefiles(savefile=jsonfiles[0],skipdone=True)
    for f in F:
        print "----------------------"
