import os
WD = 'C:\\git\\HuaweiHackaton\\'
os.chdir(WD)
import datetime
import classify as classme
from tabulate import tabulate#pretty print
import pandas as pd
import Tree


def getUnIdentifier():
    unid = datetime.datetime.now()
    unid = str(unid).replace('-','').replace(' ','').replace(':','').replace('.','')
    return unid

import cv2
#import sys

#faceCascade = cv2.CascadeClassifier(os.path.join(WD, "haarcascade_frontalface_default.xml"))
#faceCascade.empty()

camera = cv2.VideoCapture(0)

try:
    while True:

        return_value,image = camera.read()


        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = gray
        
        cv2.imshow('image',gray)
        
        ## Check the Keyboard input
        ch = 0xFF & cv2.waitKey(1) 
        
        if ch == ord('q'):
            break
        
        if ch == ord('0'):
            cv2.imwrite('training_dataset\\happy\\{}.jpg'.format(getUnIdentifier()),image)

        if ch == ord('9'):
            cv2.imwrite('training_dataset\\sad\\{}.jpg'.format(getUnIdentifier()),image)

        if ch == ord('8'):
            cv2.imwrite('training_dataset\\neutral\\{}.jpg'.format(getUnIdentifier()),image)

#        if ch == ord('7'):
#            cv2.imwrite('training_dataset\\angry\\{}.jpg'.format(getUnIdentifier()),image)

        if ch == ord('l'):
            cv2.imwrite('test_img\\test_img.jpg',image)
            result = classme.classifyNewPic(os.path.join(WD,'test_img\\test_img.jpg'))
            resultdf = pd.DataFrame([f for f in result.items()], columns=['Emotion', 'Probability'])
            resultdf = resultdf.sort_values(by='Probability', ascending=False)
            
            emotion = resultdf.iloc[0].Emotion
            prob = round(float(resultdf.iloc[0].Probability) * 100,2) 
            
            if emotion != 'neutral':
                print ("There is a {}% chance you are feeling {}".format(prob, emotion))
                suggestion = Tree.getSuggestion(emotion, 'Sunday', 'Evening')
                print (suggestion)
                print ('\n \n')
                print ('Probabilities:')
                print (tabulate(resultdf, headers='keys', tablefmt='psql'))

            else:
                print ('no emotion detected')
            print ('')
            print ('')
            print ('')
finally:
    camera.release()
    cv2.destroyAllWindows()


