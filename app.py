# import pygame,sys
# from pygame.locals import *
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# import cv2


# print(tf.__version__)
# print("Keras version:", keras.__version__)
# WHITE=(255,255,255)
# BLACK=(0,0,0)
# RED=(255,0,0)
# BOUNDRYINC=5
# image_cnt=1
# IMAGESAVE=False
# PREDICT=True
# #pygame.font.init()



# MODEL=keras.models.load_model("my_model.keras")

# LABELS={0:"Zero",1: "One",2: "Two", 3: "Three",4: "Four", 5: "Five",6:"Six", 7:"Seven",
# 8: "Eight", 9: "Nine"}

# WINDOWSIZEX=640
# WINDOWSIZEY=480
# #Initialise the game
# pygame.init()
# FONT = pygame.font.SysFont('Arial', 18)
# #FONT=pygame.font.Font("freesansbold.tff",18)

# DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

# pygame.display.set_caption("Digit Board")

# iswriting=False

# number_xcord=[]
# number_ycord=[]

# while True:
#     for event in pygame.event.get():
#        if event.type == QUIT:
#            pygame.quit ()
#            sys.exit()
        
#        if event.type==MOUSEMOTION and iswriting:
#             # postions of our coordinates
#             xcord,ycord=event.pos 
#             pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)

#             #keep on updating/appending new coordinates

#             number_xcord.append(xcord)
#             number_ycord.append(ycord)
        
#        if event.type==MOUSEBUTTONDOWN:
#             iswriting=True
        
#        if event.type==MOUSEBUTTONUP:
#            iswriting=False
#            number_xcord=sorted(number_xcord)
#            number_ycord=sorted(number_ycord)

#            #we are trying to make a rectangle around the number
#            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0 ), min(WINDOWSIZEX, number_xcord[-1]+BOUNDRYINC)
#            #rect_min_Y,rect_max_Y=max(number_ycord[0] - BOUNDRYINC), min(number_ycord[-1]+BOUNDRYINC, WINDOWSIZEX)
#            rect_min_Y = number_ycord[0] - BOUNDRYINC
#            rect_max_Y = min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEX)

#            # Create a rectangle around the number
#            #rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
#            #rect_max_x = min(number_xcord[-1] + BOUNDRYINC, WINDOWSIZEX)
#            #rect_min_Y = max(number_ycord[0] - BOUNDRYINC, 0)
#            #rect_max_Y = min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

#            number_xcord=[]
#            number_ycord =[]

#            #extract
#            #img_arr =np.array(pygame.PixelArray (DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y, rect_max_Y].T.astype(np.float32)
#            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)
  
#            if IMAGESAVE:
#                cv2.imwrite("image.png")
#                image_cnt+=1

#             #incorporate python with ml
#            if PREDICT:
               
#                #resize the image
#                image=cv2.resize(img_arr,(28,28))
#                image=np.pad(image, (10,10), 'constant', constant_values = 0 )
#                image =cv2.resize(image, (28,28)) /255
               
#                #model prediction
#                label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28,28, 1)))])

#                textSurface= FONT.render(label, True, RED, WHITE) 
#                textRecObj=textSurface.get_rect()
#                textRecObj.left, textRecObj.bottom = rect_min_x,rect_max_Y

#                DISPLAYSURF.blit(textSurface, textRecObj)
               
#            if event.type == KEYDOWN:
#               if event.unicode == "n":
#                  DISPLAYSURF.fill(BLACK)

#     pygame.display.update()
               



# import pygame, sys
# from pygame.locals import *
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model  # Correct import for loading model
# import cv2

# print(tf.__version__)
# print("Keras version:", keras.__version__)

# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# BOUNDRYINC = 5
# image_cnt = 1
# IMAGESAVE = False
# PREDICT = True

# MODEL = load_model("my_model.keras")
# input_data = np.random.random((1, 28, 28, 1))  # Shape should match your model's input

# # Make a prediction
# try:
#     predictions = MODEL.predict(input_data)
#     print("Prediction made successfully:", predictions)
# except Exception as e:
#     print(f"Error during prediction: {e}")
# LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
#           8: "Eight", 9: "Nine"}

# WINDOWSIZEX = 640
# WINDOWSIZEY = 480

# # Initialize the game
# pygame.init()
# FONT = pygame.font.SysFont('Arial', 18)

# DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
# pygame.display.set_caption("Digit Board")

# iswriting = False

# number_xcord = []
# number_ycord = []

# while True:
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             pygame.quit()
#             sys.exit()

#         if event.type == MOUSEMOTION and iswriting:
#             # positions of our coordinates
#             xcord, ycord = event.pos
#             pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

#             # keep on updating/appending new coordinates
#             number_xcord.append(xcord)
#             number_ycord.append(ycord)

#         if event.type == MOUSEBUTTONDOWN:
#             iswriting = True

#         if event.type == MOUSEBUTTONUP:
#             iswriting = False
#             number_xcord = sorted(number_xcord)
#             number_ycord = sorted(number_ycord)

#             # we are trying to make a rectangle around the number
#             rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
#             rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
#             rect_min_Y = max(number_ycord[0] - BOUNDRYINC, 0)
#             rect_max_Y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

#             number_xcord = []
#             number_ycord = []

#             # extract the image from the screen
#             img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

#             if IMAGESAVE:
#                 cv2.imwrite("image.png")
#                 image_cnt += 1

#             if PREDICT:
#                 # resize the image to 28x28 for the model
#                 image = cv2.resize(img_arr, (28, 28))

#                 # pad the image with zeros (if needed) and normalize
#                 image = np.pad(image, (10, 10), 'constant', constant_values=0)
#                 image = cv2.resize(image, (28, 28)) / 255

#                 # model prediction
#                 image = image.reshape(1, 28, 28, 1)  # Reshaping for the model
#                 label = str(LABELS[np.argmax(MODEL.predict(image))])

#                 textSurface = FONT.render(label, True, RED, WHITE)
#                 textRecObj = textSurface.get_rect()
#                 textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y

#                 DISPLAYSURF.blit(textSurface, textRecObj)

#         if event.type == KEYDOWN:
#             if event.unicode == "n":
#                 DISPLAYSURF.fill(BLACK)

#     pygame.display.update()

        





# import pygame
# import sys
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# import cv2

# print("TensorFlow Version:", tf.__version__)
# print("Keras Version:", tf.keras.__version__)

# # Constants
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# BOUNDRYINC = 5
# IMAGESAVE = False
# PREDICT = True

# # Load model
# MODEL = load_model("my_model.keras")

# LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 
#           5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# WINDOWSIZEX = 640
# WINDOWSIZEY = 480

# # Initialize pygame
# pygame.init()
# FONT = pygame.font.SysFont('Arial', 18)

# DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
# pygame.display.set_caption("Digit Board")

# # State Variables
# iswriting = False
# number_xcord = []
# number_ycord = []

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#         if event.type == pygame.MOUSEMOTION and iswriting:
#             xcord, ycord = event.pos
#             pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
#             number_xcord.append(xcord)
        #     number_ycord.append(ycord)

        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     iswriting = True

        # if event.type == pygame.MOUSEBUTTONUP:
        #     iswriting = False
        #     if number_xcord and number_ycord:  # Ensure there are points drawn
        #        number_xcord = sorted(number_xcord)
        #        number_ycord = sorted(number_ycord)

        # # Define rectangle boundaries
        #        rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
        #        rect_max_x = min(number_xcord[-1] + BOUNDRYINC, WINDOWSIZEX)
        #        rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
        #        rect_max_y = min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

        # # Draw the red rectangle
        #        pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

        # # Extract and preprocess the image
        #        img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

        # Preprocess the image
            #    image = cv2.resize(img_arr, (28, 28))
            #    image = np.pad(image, ((2, 2), (2, 2)), 'constant', constant_values=0)
            #    image = cv2.resize(image, (28, 28))
            #    image = image / 255.0
            #    image = image.reshape(1, 28, 28, 1)
               # Resize to 28x28
#                image = cv2.resize(img_arr, (28, 28))

# # Add a channel dimension for grayscale (if needed)
#                if len(image.shape) == 2:  # If the image is 2D (no channels)
#                     image = image.reshape(28, 28, 1)

# Normalize to the range [0, 1]
#                image = image / 255.0

# # Add a batch dimension (needed for the model to process the input)
#                image = np.expand_dims(image, axis=0)  # Final shape: (1, 28, 28, 1)


#         # Prediction
#                predictions = MODEL.predict(image)
#                predicted_label = np.argmax(predictions)

#         # Display prediction
#                label_text = f"Predicted: {LABELS[predicted_label]}"
#                print(f"Prediction probabilities: {predictions}")
#                print(f"Predicted label index: {predicted_label}")

#         # Render prediction text
#                text_surface = FONT.render(label_text, True, RED, WHITE)
#                text_rect = text_surface.get_rect()
#                text_rect.left, text_rect.bottom = rect_min_x, rect_max_y
#                DISPLAYSURF.blit(text_surface, text_rect)

#         # Reset coordinates
#                number_xcord = []
#                number_ycord = []

        # if event.type == pygame.MOUSEBUTTONUP:
        #     iswriting = False
        #     number_xcord = sorted(number_xcord)
        #     number_ycord = sorted(number_ycord)

        #     rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
        #     rect_min_Y = number_ycord[0] - BOUNDRYINC
        #     rect_max_Y = min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

        #     # Capture the drawn area
        #     img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

        #     if IMAGESAVE:
        #         cv2.imwrite("image.png")
                
        #     if PREDICT:
        #         # Preprocess the image
        #         image = cv2.resize(img_arr, (28, 28))
        #         image = np.pad(image, ((2, 2), (2, 2)), 'constant', constant_values=0)  # Pad with 0s
        #         image = cv2.resize(image, (28, 28))  # Resize again to 28x28
        #         image = image.astype(np.float32) / 255.0  # Normalize to 0-1
        #         image = image.reshape(1, 28, 28, 1)  # Shape should be (1, 28, 28, 1)

        #         # Make prediction
        #         prediction = MODEL.predict(image)
        #         predicted_label = np.argmax(prediction)
        #         print("Prediction probabilities:", prediction)


        #         print(f"Predicted label index: {predicted_label}")

        #         label = str(LABELS[predicted_label])
                

        #         # Display prediction result
        #         textSurface = FONT.render(label, True, RED, WHITE)
        #         textRecObj = textSurface.get_rect()
        #         textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y
        #         DISPLAYSURF.blit(textSurface, textRecObj)

    #     if event.type == pygame.KEYDOWN:
    #         if event.unicode == "n":
    #             DISPLAYSURF.fill(BLACK)

    # pygame.display.update()




# import pygame
# import numpy as np
# import keras
# from pygame.locals import *
# import cv2  # OpenCV for image processing

# # Load the trained model
# model = keras.models.load_model('my_model.keras')

# # Initialize pygame
# pygame.init()

# # Set up the screen dimensions
# WIDTH, HEIGHT = 280, 280
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Digit Drawing Board")

# # Create a surface to draw on
# canvas = pygame.Surface((WIDTH, HEIGHT))
# canvas.fill((255, 255, 255))  # Fill with white color

# # Set up font for displaying prediction
# font = pygame.font.SysFont('Arial', 30)

# # Function to preprocess the drawn image
# def preprocess_image(image):
#     # Convert the drawn image to grayscale (already white background)
#     image = pygame.surfarray.array3d(image)
#     image = np.mean(image, axis=-1)  # Convert to grayscale

#     # Threshold the image to get a clear black-and-white image
#     _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

#     # Find the contours of the drawn digit to crop it tightly
#     contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Find the bounding box of the largest contour
#         x, y, w, h = cv2.boundingRect(contours[0])
        
#         # If the digit is very small or not well-drawn, avoid cropping too much
#         if w > 5 and h > 5:  # Only crop if the bounding box has a reasonable size
#             # Crop the image to the bounding box
#             cropped_image = image[y:y+h, x:x+w]
#         else:
#             cropped_image = image
#     else:
#         # If no contour is found, return a blank image (this handles empty canvas)
#         cropped_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

#     # Resize the cropped image to 28x28 pixels
#     resized_image = cv2.resize(cropped_image, (28, 28))

#     # Normalize the resized image to [0, 1]
#     resized_image = resized_image / 255.0
#     resized_image = np.expand_dims(resized_image, axis=-1)  # Add channel dimension (28, 28, 1)
#     resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension (1, 28, 28, 1)
#     return resized_image

# # Function to predict the digit
# def predict_digit(image):
#     # Preprocess the image
#     processed_image = preprocess_image(image)
#     # Get the model's prediction
#     prediction = model.predict(processed_image)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return predicted_class

# # Main loop
# running = True
# drawing = False
# prediction = None

# while running:
#     screen.fill((255, 255, 255))  # Fill screen with white
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             running = False

#         elif event.type == MOUSEBUTTONDOWN:
#             if event.button == 1:  # Left click to start drawing
#                 drawing = True
#                 # Clear the canvas before drawing new
#                 canvas.fill((255, 255, 255))

#         elif event.type == MOUSEBUTTONUP:
#             if event.button == 1:  # Left click to stop drawing
#                 drawing = False
#                 # Predict digit once the user stops drawing
#                 prediction = predict_digit(canvas)

#         elif event.type == MOUSEMOTION:
#             if drawing:
#                 pygame.draw.circle(canvas, (0, 0, 0), event.pos, 5)

#     # Display the drawn canvas on the screen
#     screen.blit(canvas, (0, 0))

#     # If prediction is made, display the predicted digit
#     if prediction is not None:
#         # Display the predicted result
#         prediction_text = font.render(f"Prediction: {prediction}", True, (0, 0, 0))
#         screen.blit(prediction_text, (WIDTH // 2 - prediction_text.get_width() // 2, HEIGHT - 40))

#     # Update the screen
#     pygame.display.update()

# pygame.quit()




import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Colors and constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BOUNDRYINC = 5  # Boundary increment for the rectangle
image_cnt = 1
IMAGESAVE = False
PREDICT = True

# Initialize Pygame
pygame.init()
FONT = pygame.font.SysFont('Arial', 18)  # Use Arial font
DISPLAYSURF = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Digit Board")

# Load the pre-trained model
MODEL = load_model("my_model.keras")

# Label mapping
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Flags and tracking
iswriting = False
number_xcord = []
number_ycord = []

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:  # Ensure there are valid points
                # Sort and calculate rectangle bounds
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)
                rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
                rect_max_x = min(number_xcord[-1] + BOUNDRYINC, 640)
                rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
                rect_max_y = min(number_ycord[-1] + BOUNDRYINC, 480)

                # Draw the rectangle
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y,
                                                    rect_max_x - rect_min_x,
                                                    rect_max_y - rect_min_y), 2)

                # Extract the drawn image
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,
                                                                   rect_min_y:rect_max_y].T.astype(np.float32)
                number_xcord = []
                number_ycord = []

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                    image_cnt += 1

                # Prediction
                if PREDICT:
                    # Preprocess the image
                    image = cv2.resize(img_arr, (28, 28))  # Resize to 28x28
                    image = 255 - image                    # Invert colors
                    image = image / 255.0                  # Normalize
                    image = image.reshape(1, 28, 28, 1)    # Reshape for the model

                    # Predict
                    prediction = MODEL.predict(image)
                    predicted_label = str(LABELS[np.argmax(prediction)])
                    print(f"Prediction: {predicted_label}")

                    # Display the prediction
                    textSurface = FONT.render(predicted_label, True, RED, WHITE)
                    textRectObj = textSurface.get_rect()
                    textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y
                    DISPLAYSURF.blit(textSurface, textRectObj)

        # Clear screen on 'n' key press
        if event.type == KEYDOWN and event.unicode == "n":
            DISPLAYSURF.fill(BLACK)

    pygame.display.update()
