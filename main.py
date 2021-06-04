from __future__ import division
import cv2
import numpy as np
import time
import Adafruit_PCA9685
    
pwm = Adafruit_PCA9685.PCA9685()

servo_min = 150
servo_max = 600

def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)
    
pwm.set_pwm_freq(60)

def nothing(x):
    pass


cap = cv2.VideoCapture(0)


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-S", "Trackbars", 167, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 94, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars") 
    l_S = cv2.getTrackbarPos("L-S", "Trackbars")
    l_V = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_S = cv2.getTrackbarPos("U-S", "Trackbars")
    u_V = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_S, l_V])
    upper_red = np.array([u_h, u_S, u_V])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            if len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
                pwm.set_pwm(0, 0, servo_min)
                time.sleep(1)
                pwm.set_pwm(0, 0, servo_max)
                time.sleep(1)
                
            elif len(approx) == 4:
                cv2.putText(frame, "rectangle", (x, y), font, 1, (0, 0, 0))
                pwm.set_pwm(1, 0, servo_min)
                time.sleep(1)
                pwm.set_pwm(1, 0, servo_max)
                time.sleep(1)
            
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))
                pwm.set_pwm(2, 0, servo_min)
                time.sleep(1)
                pwm.set_pwm(2, 0, servo_max)
                time.sleep(1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
