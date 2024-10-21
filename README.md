Math Symbols Recognition
This project is a deep learning-powered web application designed to recognize handwritten mathematical symbols. Users can input simple math expressions via a canvas, and the app predicts the digits and operations (addition, subtraction, multiplication, and division) using a convolutional neural network (CNN).

Features:
Handwritten digit and symbol recognition
Real-time prediction through a web interface
Built using TensorFlow and Streamlit


Requirements:
Python 3.x
TensorFlow
Streamlit
OpenCV
Pillow


I. Installation:
1. Clone the repository:

git clone https://github.com/MinhNgNg18/Math-Symbols-Recognition.git
cd Math-Symbols-Recognition

2. Install dependencies:

pip install -r requirements.txt

II. Running the Project
1. Train the Model:
To train the model, run:

python model.py

2. Start the Streamlit App
Once the model is trained, start the app:

streamlit run app.py

III. Usage:
Draw a mathematical expression using your mouse on the canvas, and the model will predict the digits and operations.

Model:
A CNN is used for multiclass classification of digits (0-9) and operations ( +, -, *, /).
Data augmentation techniques are used to improve model generalization.

Future Improvements:
Expand symbol recognition to include more complex mathematical operations.
Optimize the model for improved accuracy.

Author
Ngoc Minh Nguyen
