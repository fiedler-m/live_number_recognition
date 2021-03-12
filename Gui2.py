import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # needed to override restrictions


def draw_num(event, x, y, flags, param):
    global x1, y1, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (x1, y1), (x, y), (255), 15)
            x1 = x
            y1 = y
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (x1, y1), (x, y), (255), 15)

    return x, y


#Intialize matplotlib and parameters
y_probability = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
fig, ax = plt.subplots(figsize=(9, 7))
ax.bar(np.arange(0, 10), y_probability, label='Prediction Accuracy')
plt.xticks(np.arange(0, 10))
plt.ylim((0, 100))
plt.xlabel('Number Prediction', fontsize=16, weight='bold')
plt.ylabel('Prediction Probability (%)', fontsize=16, weight='bold')
plt.rcParams.update({'font.size': 16})

plt.ion()
plt.show()

# load trained model

model = load_model('/Users/mfiedler/pythonProject4/model_save/mnist.1')


drawing = False

# Initialize canvas
img = np.zeros((28, 28, 1), np.uint8)
img = cv2.resize(img, (500, 500))
clear_img = img.copy()
cv2.namedWindow('Draw Number')
cv2.setMouseCallback('Draw Number', draw_num)

# Loop will keep running to display the image window along with the bar plot
while True:
    cv2.imshow('Draw Number', img)
    image_drawn = img.copy()

    if image_drawn.max() > 1:
        image_drawn = image_drawn.astype('float32') / image_drawn.max()
        image_drawn = cv2.resize(image_drawn, (28, 28))
        image_reshaped = image_drawn.reshape((28, 28, 1))
        final_image = np.array([image_reshaped])

        prediction = model.predict(final_image)
        y_probability = prediction[0].round(2) * 100

        # matplotlib predictions
        plt.cla()
        ax.bar(np.arange(0, 10), y_probability, label='Prediction Accuracy')
        plt.xticks(np.arange(0, 10))
        plt.xlabel('Numbers Predicted', fontsize=16, weight='bold')
        plt.ylabel('Prediction Probability (%)', fontsize=16, weight='bold')
        plt.ylim((0, 100))
        plt.rcParams.update({'font.size': 16})
        fig.canvas.draw()

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        img = np.zeros((28, 28, 1), np.uint8)
        img = cv2.resize(img, (500, 500))
        image_drawn = img

        y_probability = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        plt.cla()
        ax.bar(np.arange(0, 10), y_probability, label='Prediction Accuracy')
        plt.xticks(np.arange(0, 10))
        plt.xlabel('Numbers Predicted', fontsize=16, weight='bold')
        plt.ylabel('Prediction Probability (%)', fontsize=16, weight='bold')
        plt.ylim((0, 100))
        plt.rcParams.update({'font.size': 16})
        fig.canvas.draw()

cv2.destroyAllWindows()