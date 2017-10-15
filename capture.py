import os
WD = 'C:\\git\\HuaweiHackaton\\'
os.chdir(WD)
import datetime
import classify as classme
from tabulate import tabulate#pretty print
import pandas as pd
import Tree
import random


def getUnIdentifier():
    unid = datetime.datetime.now()
    unid = str(unid).replace('-','').replace(' ','').replace(':','').replace('.','')
    return unid

import cv2
camera = cv2.VideoCapture(0)

try:
    while True:

        return_value,image = camera.read()
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #try to center head
        #NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        image  = image [0:330, 100:500]
                    
        cv2.imshow('image',image)
        
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

        if ch == ord('7'):
            cv2.imwrite('training_dataset\\angry\\{}.jpg'.format(getUnIdentifier()),image)

        if ch == ord('l'):
            cv2.imwrite('test_img\\test_img.jpg',image)
            result = classme.classifyNewPic(os.path.join(WD,'test_img\\test_img.jpg'))
            resultdf = pd.DataFrame([f for f in result.items()], columns=['Emotion', 'Probability'])
            resultdf = resultdf.sort_values(by='Probability', ascending=False)
            
            emotion = resultdf.iloc[0].Emotion
            prob = round(float(resultdf.iloc[0].Probability) * 100,2) 
            
            dow = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'])
            tod = random.choice(['Morning', 'Evening'])
            
            if emotion != 'neutral':
                print ("There is a {}% chance you are feeling {}".format(prob, emotion))
                print ("(Day of the week: {} {}) \n".format(dow, tod))
                suggestion = Tree.getSuggestion(emotion, dow, tod)
                print (suggestion)
                #print ('Probabilities:')
                #print (tabulate(resultdf, headers='keys', tablefmt='psql'))
                print ('\n \n')
            else:
                print ('no emotion detected')
            print ('')
            print ('')
            print ('')
finally:
    camera.release()
    cv2.destroyAllWindows()


