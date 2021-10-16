# Dog-Breed Predictor
### Capstone Project Overview:

This project uses Convolutional Neural Networks (CNNs)! In this project, expected to learn how to build a pipeline to process real-world, user-supplied images.

### Project Definition: 

In this Udacity Capstone project, I have developed an algorithm that could be used as part of a mobile or web app. The code will accept any user-supplied image as input of the following results returned:
- If a dog is detected in the image, it will provide an estimate of the dog's breed. 
- If a human is detected, it will provide an estimate of the dog breed that is most resembling.
- If neither dog or human are detected, the algorithm will return wrong image loaded for prediction.

### Project Analysis:

In this project, I have used transfer learning using pre-trained model to create a CNN that can identify dog breed from images to attain at least 60% accuracy on the test set.
I have analysis the accuracy of 3 models: 
- VGG19: 45.5742%
- Inception: 79.4258%
- ResNet50: 81.6986%

Using ResNet50 model for improvement and further implementation and have attained accuracy of 78.3493%
![plot](https://user-images.githubusercontent.com/91511739/137575181-12308d35-2fb9-483f-aec7-0bf08cb1dcf6.png)

### Conclusion:

Algorithm Performance:
- The algorithm is performing better than expected. Out of the 8 tested dog images, 6 were predicted correctly with confidence rate of more than 70%. 
- One of the dog image is Shih Tzu (not in the trained classes) and is misintepreted as a Maltese. Also a chihuahua is misclassified as a German pinscher.
- For the case of misclassification of shih tzu, it would improve the algorithm with more training data of more dog breeds.
- For the case of misclassification of pinscher could be due to dogs which looks similarbut of different size. Would improve the algorithm if there is a way to intepret dog size from the image. 
- The algorithm can be improved by differentiate predicted dog breed of low confidence rate that is probably misclassified by setting a threshold to the confidence rate of prediction.

### Flask Web App:

- For web app, run main.py
- Load in local browser of url: http://127.0.0.1:5000/

Extract of screenshots of Web App as follows:

Home page.
![WebApp_home](https://user-images.githubusercontent.com/91511739/137574827-b49b77a2-b35e-4b45-9a27-a82f0ea4b09e.jpg)
Load image.
![WebApp_load](https://user-images.githubusercontent.com/91511739/137574833-24564962-6891-4688-befe-043696297be4.jpg)
Predict image.
![WebApp_predict](https://user-images.githubusercontent.com/91511739/137574844-f3f8e3d0-9cbe-4379-b397-e82cf88fc6c8.jpg)
Predict wrong image.
![WebApp_predictwrong](https://user-images.githubusercontent.com/91511739/137574848-88089a55-83d9-4507-a713-196298316a13.jpg)
