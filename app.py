import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from google import genai # This is the new way
from flask import Flask, render_template, request, make_response, redirect, url_for, flash # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from dotenv import load_dotenv # type: ignore
import os
import datetime
from PIL import Image # type: ignore
import io
from werkzeug.security import generate_password_hash, check_password_hash # type: ignore
from functools import wraps
import sqlite3
import pickle
import numpy as np # type: ignore
import joblib # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from keras.preprocessing import image  # type: ignore
import tensorflow as tf # type: ignore
import torch # type: ignore
from torchvision import transforms # type: ignore
from model import load_model1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model (Ensure that the model is loaded only once when the app starts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_brain = load_model1(device=device)
model_brain.to(device)

# Transform for the uploaded image (including contrast adjustment)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ColorJitter(contrast=0.5),  # Adjust contrast by a factor of 0.5
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to remove old image if it exists
def remove_old_image():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the old image
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")

# Predict function
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model_brain(image)
        _, predicted = torch.max(output.data, 1)
    
    return 'Tumor' if predicted.item() == 1 else 'No Tumor'

# Load trained model
MODEL_PATH = 'Models/trained_model.h5'
model_pne = load_model(MODEL_PATH, compile=False)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image_pne(img_path):
    img = preprocess_image(img_path)
    prediction = model_pne.predict(img)
    return 'Pneumonia' if prediction > 0.5 else 'Normal'
load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# Load the Random Forest CLassifier model
filename = 'Models/diabetes-model.pkl'
filename1 = 'Models/cancer-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
rf = pickle.load(open(filename1, 'rb'))


# --- Gemini API Setup ---
client = genai.Client(
    api_key=GOOGLE_API_KEY,
)

# Define the model ID you want to use
model_id = 'gemini-2.5-flash'

# --- Create directories for saving interactions ---
interactions_dir_diagnosis = "diagnosis_database"
if not os.path.exists(interactions_dir_diagnosis):
    os.makedirs(interactions_dir_diagnosis)

interactions_dir_medicine = "medicine_database"
if not os.path.exists(interactions_dir_medicine):
    os.makedirs(interactions_dir_medicine)

interactions_dir_image_analysis = "image_analysis"
if not os.path.exists(interactions_dir_image_analysis):
    os.makedirs(interactions_dir_image_analysis)

interactions_dir_chat_history = "chat_history"
if not os.path.exists(interactions_dir_chat_history):
    os.makedirs(interactions_dir_chat_history)

# --- User Database (SQLite) ---
# Initialize the database connection
db = sqlite3.connect('users.db')
cursor = db.cursor()

# Create the users table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Commit the changes to the database
db.commit()

# Close the database connection
db.close()

# --- Utility Functions ---
def get_gemini_response(prompt):
    """Gets a response from the Gemini API using the new SDK."""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

def save_interaction(user_input, insights, directory):
    """Saves user input and Gemini's response to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"interaction_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("User Input:\n")
        f.write(user_input + "\n\n")
        f.write("Response:\n")
        f.write(insights)

# --- Function to read interactions from files ---
def read_interactions_from_file(directory):
    """Reads interaction files from a directory."""
    interactions = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                interactions.append({"filename": filename, "content": content})
    return interactions

# --- Diagnosis App Functions ---
def get_user_input_from_form():
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    diabetes = request.form.get('diabetes')
    previous_diseases = request.form.get('previous_diseases')
    choice = request.form.get('choice')
    return gender, age, diabetes, previous_diseases, choice

def get_medical_insights(symptoms=None, disease=None, gender=None, age=None,
                         diabetes=None, previous_diseases=None):
    """Provides medical insights from Gemini."""
    if symptoms:
        prompt = (
            f"A {gender} patient aged {age} presents with these symptoms: {symptoms}. "
            f"They have {'diabetes' if diabetes == 'yes' else 'no diabetes'}"
            f"{' and a history of ' + previous_diseases if previous_diseases != 'none' else ''}. "
            "Provide the following information:\n"
            "1. **Symptom Descriptions:** Describe the provided symptoms in detail.\n"
            "2. **Possible Diseases:** List potential diseases or conditions.\n"
            "3. **Disease Descriptions:** Provide detailed descriptions of the possible diseases.\n"
            "4. **Precautions:** Suggest precautions to take for each possible disease.\n"
            "5. **Medications:** List common medications used to treat each possible disease.\n"
            "6. **Workout:** Recommend suitable workout regimens for each possible disease (if any).\n"
            "7. **Diet:** Provide dietary advice for each possible disease.\n"
            "8. **Suggested Medical Diagnosis:** Recommend relevant diagnostic tests (e.g., blood tests, X-rays)."
        )
    elif disease:
        prompt = (
            f"Provide comprehensive information about {disease}, specifically for a "
            f"{gender} patient aged {age} with "
            f"{'diabetes' if diabetes == 'yes' else 'no diabetes'}"
            f"{' and a history of ' + previous_diseases if previous_diseases != 'none' else ''}."
            "\nInclude the following:\n"
            "1. **Disease Description:** \n"
            "2. **Precautions:** \n"
            "3. **Medications:** \n"
            "4. **Workout:** \n"
            "5. **Diet:** \n"
            "6. **Suggested Medical Diagnosis:** "
        )
    else:
        return "Please provide either symptoms or a disease."

    response = get_gemini_response(prompt)
    return response

# --- Medicine App Functions ---
def get_medicine_info(name, search_by="generic", country="US"):
    """Gets information about a medicine from Gemini."""
    if search_by == "generic":
        prompt = (
            f"Provide information about the medicine '{name}' "
            f"({search_by.upper()} name) as available in {country.upper()}.\n\n"
            "Include the following details, clearly separated:\n\n"
            "1. **Indications:** What is this medicine used for?\n"
            "2. **Pharmacology:** How does this medicine work in the body?\n"
            "3. **Dosage & Administration:** What are the usual dosages for adults and children?\n"
            "4. **Interaction:** What other medications or substances does this interact with?\n"
            "5. **Side Effects:** What are the common and serious side effects?\n"
            "6. **Pregnancy & Lactation:** Can it be used during pregnancy or breastfeeding?\n"
            "7. **Precautions & Warnings:** Are there any special warnings or groups it shouldn't be used for?\n"
            "8. **Overdose Effects:** What happens in case of an overdose?\n"
            "9. **Storage Conditions:** How should this medicine be stored?\n"
            "10. **Medicine Brand Names:** What are some brand names for this medicine?\n" 
        )
    else:  # search_by == "brand":
        prompt = (
            f"Provide information about the medicine '{name}' "
            f"({search_by.upper()} name), including its country of origin.\n\n"
            "Include the following details, clearly separated:\n\n"
            "1. **Indications:** What is this medicine used for?\n"
            "2. **Pharmacology:** How does this medicine work in the body?\n"
            "3. **Dosage & Administration:** What are the usual dosages for adults and children?\n"
            "4. **Interaction:** What other medications or substances does this interact with?\n"
            "5. **Side Effects:** What are the common and serious side effects?\n"
            "6. **Pregnancy & Lactation:** Can it be used during pregnancy or breastfeeding?\n"
            "7. **Precautions & Warnings:** Are there any special warnings or groups it shouldn't be used for?\n"
            "8. **Overdose Effects:** What happens in case of an overdose?\n"
            "9. **Storage Conditions:** How should this medicine be stored?\n"
            "10. **Generic Name:** What is the generic name for this brand?\n"
        )

    response = get_gemini_response(prompt)
    return response

# --- Image Analysis Functions ---
def process_image(image_data, analysis_type):
    """Processes the uploaded image based on the analysis type."""
    try:
        image = Image.open(io.BytesIO(image_data))

        if analysis_type == "xray_description":
            prompt = """
            You are a radiologist. Analyze this X-ray image and provide a detailed report with the following sections:

            **Findings:**
              * 1. Describe the image. (e.g., This is an X-ray image of the chest, taken from a posterior-anterior view.)
              * 2. Point out any abnormalities or significant features. Be specific and use medical terminology where appropriate.

            **Interpretation:**
              * 1. Provide a possible explanation for each finding.
              * 2. Include potential causes, likely symptoms the patient might experience, and typical treatment approaches.
            """
        elif analysis_type == 'skin_cancer':
            prompt = """
            You are a dermatologist. 
            Analyze this skin image and provide a comprehensive report:

            **Detailed Findings:**
              * Describe the image, highlighting any abnormalities or key features. Be as specific as possible, using medical terminology to describe size, shape, location, color, borders, and any other relevant characteristics. 

            **Possible Conditions:**
              * 1. List potential medical conditions that could be indicated by the findings. 
              * 2. Provide a concise explanation of each condition, including its typical causes, symptoms, and prognosis.

            **Recommendations:**
              * 1. Give clear and actionable recommendations based on the analysis.
              * 2. Include suggestions for further tests, potential treatment options, or lifestyle changes that might be beneficial. 
              * 3. Explain the reasoning behind each recommendation. 
            """
        elif analysis_type == 'tumor_detection': 
            prompt = """
            You are a medical imaging expert.
            Analyze this image and provide a comprehensive report:

            **Detailed Findings:**
              * 1. Describe the image and identify the body part or region being imaged.
              * 2. Point out any suspicious masses or lesions that could indicate a tumor. Be specific about their size, shape, location, and any other relevant characteristics.

            **Possible Conditions:**
              * 1. List potential types of tumors that could be present based on the findings. 
              * 2. Provide a brief overview of each possible condition, including its typical presentation, behavior, and potential implications.

            **Recommendations:**
              * 1. Recommend further investigations or tests necessary to confirm or rule out the presence of a tumor. 
              * 2. Explain the importance and purpose of each recommended test. 
            """

        elif analysis_type == 'pregnancy_detection':
            prompt = """
            You are a medical professional specializing in pregnancy detection. 
            Analyze this image and provide a detailed report focusing on the possibility of pregnancy:

            **Findings:**
              * 1. Describe the image and any visual cues related to pregnancy.
              * 2. If there are signs of pregnancy, mention them specifically.

            **Interpretation:**
              * 1. Provide an interpretation of the findings, addressing whether the image suggests pregnancy. 
              * 2. If possible, provide a cautious estimate of gestational age based on any observable features.

            **Recommendations:**
              * 1. Recommend appropriate medical tests or consultations for confirmation of pregnancy. 
              * 2. Emphasize that this analysis is not a definitive diagnosis and a qualified healthcare professional should be consulted.
            """
        elif analysis_type == 'medical_image':
            prompt = """
            You are a medical image analysis AI.
            Analyze this image and provide information based on these points:

            **Image Description:**
              * 1. Describe the image. What does the image appear to show?
              * 2. Are there any specific objects or structures that stand out?

            **Possible Interpretations:**
              * 1. What could this image potentially represent in a medical context?
              * 2. Are there any visual cues that suggest a possible medical condition or procedure?

            **Cautions:**
              * 1. Note that this is just an AI analysis and not a medical diagnosis. 
              * 2. It is essential to consult with a qualified healthcare professional for accurate interpretation and diagnosis.
            """
        elif analysis_type == 'lesion_detection':
            prompt = """
            You are a medical image analysis AI.
            Analyze this image and provide a comprehensive report:

            **Image Description:**
              * 1. Describe the image. What does the image appear to show?
              * 2. Are there any specific objects or structures that stand out?

            **Detailed Findings:**
              * 1. Describe the image and identify the body part or region being imaged.
              * 2. Describe any visible lesions, sores, wounds, rashes, or marks that resemble bite marks. 
              * 3. Be specific about their size, shape, color, location, and any other relevant characteristics (e.g., raised, flat, crusty, oozing).

            **Possible Causes:**
              * 1. List potential causes for the observed lesions or bite marks.
              * 2. Consider various possibilities, including skin conditions, infections, insect bites, allergic reactions, or injuries. 

            **Recommendations:**
              * 1. Recommend appropriate actions based on the findings. 
              * 2. This may include seeking medical advice, home care strategies, over-the-counter remedies, or observation.
              * 3. If the image suggests a potentially serious condition, strongly recommend consulting a healthcare professional.
            """
        else:
            return "Image analysis type not supported."

        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, image]
        )
        return response.text

    except Exception as e:
        return str(e)

def process_custom_image(image_data, custom_prompt):
    """Processes the image using a custom prompt from the user."""
    try:
        image = Image.open(io.BytesIO(image_data))
        response = client.models.generate_content(
            model=model_id,
            contents=[custom_prompt, image]
        )
        return response.text
    except Exception as e:
        return str(e)

def generate_response(prompt):
    """Generates a text response from the chatbot model."""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Authentication Decorator (No Session, Basic Redirect) ---
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'username' in request.cookies:  # Check for username cookie
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrapper

# --- Routes ---

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Hash the password for security
        hashed_password = generate_password_hash(password)

        # Connect to the database
        db = sqlite3.connect('users.db')
        cursor = db.cursor()

        try:
            # Insert the new user into the database
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                           (username, hashed_password))
            db.commit()
            return "<script>alert('User created successfully! You can now log in.'); window.location.href='/login';</script>"  # Redirect using JavaScript
        except sqlite3.IntegrityError:
            # Handle username already exists error
            return "<script>alert('Username already exists'); window.location.href='/signup';</script>"
        finally:
            db.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Connect to the database
        db = sqlite3.connect('users.db')
        cursor = db.cursor()

        try:
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result:
                hashed_password = result[0]
                if check_password_hash(hashed_password, password):
                    response = redirect(url_for('index'))
                    response.set_cookie('username', username)  # Set username cookie
                    return response
                else:
                    return "<script>alert('Invalid password'); window.location.href='/login';</script>"
            else:
                return "<script>alert('User not found'); window.location.href='/login';</script>"
        finally:
            db.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    response.set_cookie('username', '', expires=0)  # Clear cookie
    return response
@app.route('/disease')
def home():
    return render_template('index1.html')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('d_result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('diabetes'))


@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        rad = float(request.form['Radius_mean'])
        tex = float(request.form['Texture_mean'])
        par = float(request.form['Perimeter_mean'])
        area = float(request.form['Area_mean'])
        smooth = float(request.form['Smoothness_mean'])
        compact = float(request.form['Compactness_mean'])
        con = float(request.form['Concavity_mean'])
        concave = float(request.form['concave points_mean'])
        sym = float(request.form['symmetry_mean'])
        frac = float(request.form['fractal_dimension_mean'])
        rad_se = float(request.form['radius_se'])
        tex_se = float(request.form['texture_se'])
        par_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smooth_se = float(request.form['smoothness_se'])
        compact_se = float(request.form['compactness_se'])
        con_se = float(request.form['concavity_se'])
        concave_se = float(request.form['concave points_se'])
        sym_se = float(request.form['symmetry_se'])
        frac_se = float(request.form['fractal_dimension_se'])
        rad_worst = float(request.form['radius_worst'])
        tex_worst = float(request.form['texture_worst'])
        par_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smooth_worst = float(request.form['smoothness_worst'])
        compact_worst = float(request.form['compactness_worst'])
        con_worst = float(request.form['concavity_worst'])
        concave_worst = float(request.form['concave points_worst'])
        sym_worst = float(request.form['symmetry_worst'])
        frac_worst = float(request.form['fractal_dimension_worst'])

        data = np.array([[rad, tex, par, area, smooth, compact, con, concave, sym, frac, rad_se, tex_se, par_se, area_se, smooth_se, compact_se, con_se, concave_se,
                          sym_se, frac_se, rad_worst, tex_worst, par_worst, area_worst, smooth_worst, compact_worst, con_worst, concave_worst, sym_worst, frac_worst]])
        my_prediction = rf.predict(data)

        return render_template('c_result.html', prediction=my_prediction)


def ValuePredictor(to_predict_list, size):
    loaded_model = joblib.load('models/heart_model')
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/predict_heart', methods=['POST'])
def predict_heart():

    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list, 11)

            if(int(result) == 1):
                prediction = 1
            else:
                prediction = 0

            return render_template('h_result.html', prediction=prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('heart'))


# this function use to predict the output for Fetal Health from given data
def fetal_health_value_predictor(data):
    try:
        # after get the data from html form then we collect the values and
        # converts into 2D numpy array for prediction
        data = list(data.values())
        data = list(map(float, data))
        data = np.array(data).reshape(1, -1)
        # load the saved pre-trained model for new prediction
        model_path = 'Models/fetal-health-model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict(data)
        result = int(result[0])
        status = True
        # returns the predicted output value
        return (result, status)
    except Exception as e:
        result = str(e)
        status = False
        return (result, status)


# this route for prediction of Fetal Health
@app.route('/fetal_health', methods=['GET', 'POST'])
def fetal_health_prediction():
    if request.method == 'POST':
        # geting the form data by POST method
        data = request.form.to_dict()
        # passing form data to castome predict method to get the result
        result, status = fetal_health_value_predictor(data)
        if status:
            # if prediction happens successfully status=True and then pass uotput to html page
            return render_template('fetal_health.html', result=result)
        else:
            # if any error occured during prediction then the error msg will be displayed
            return f'<h2>Error : {result}</h2>'

    # if the user send a GET request to '/fetal_health' route then we just render the html page
    # which contains a form for prediction
    return render_template('fetal_health.html', result=None)


def strokeValuePredictor(s_predict_list):
    '''function to predict the output by data we get
    from the route'''

    model = joblib.load('Models/stroke_model.pkl')
    data = np.array(s_predict_list).reshape(1, -1)
    result = model.predict(data)
    return result[0]


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/predict_stroke', methods=['POST'])
# this route for predicting chances of stroke
def predict_stroke():

    if request.method == 'POST':
        s_predict_list = request.form.to_dict()
        s_predict_list = list(s_predict_list.values())
        # list to keep the values of the dictionary items of request.form field
        s_predict_list = list(map(float, s_predict_list))
        result = strokeValuePredictor(s_predict_list)

        if(int(result) == 1):
            prediction = 1
        else:
            prediction = 0
        return render_template('st_result.html', prediction=prediction)


def liverprediction(final_features):
    # Loading the pickle file
    model_path = 'Models/liver-disease_model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    result = model.predict(final_features)
    return result[0]


@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/predict_liver', methods=['POST'])
# predicting
def predict_liver_disease():

    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = liverprediction(final_features)
        pred = int(output)

        return render_template('liver_result.html', prediction=pred)


@app.route("/malaria", methods=['GET', 'POST'])
def malaria():
    return render_template('malaria.html')


@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredict():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((50, 50))
                img = np.asarray(img)
                img = img.reshape((1, 50, 50, 3))
                img = img.astype(np.float64)

                model_path = "Models/malaria-model.h5"
                model = tf.keras.models.load_model(model_path)
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
@login_required
def diagnosis_index():
    if request.method == 'POST':
        gender, age, diabetes, previous_diseases, choice = get_user_input_from_form()

        if choice == '1':
            symptoms = request.form.get('symptoms')
            insights = get_medical_insights(symptoms=symptoms, gender=gender,
                                             age=age, diabetes=diabetes,
                                             previous_diseases=previous_diseases)
            save_interaction(f"Symptoms: {symptoms}, Gender: {gender}, Age: {age}, "
                             f"Diabetes: {diabetes}, Previous Diseases: {previous_diseases}",
                             insights, interactions_dir_diagnosis)
            return render_template('diagnosis_results.html', insights=insights, symptoms=symptoms,
                                    gender=gender, age=age, diabetes=diabetes,
                                    previous_diseases=previous_diseases) 

        elif choice == '2':
            disease = request.form.get('disease')
            gender = request.form.get('gender')
            age = int(request.form.get('age') or 0)  
            diabetes = request.form.get('diabetes')
            previous_diseases = request.form.get('previous_diseases')

            insights = get_medical_insights(disease=disease, gender=gender,
                                             age=age, diabetes=diabetes,
                                             previous_diseases=previous_diseases)
            save_interaction(f"Disease: {disease}, Gender: {gender}, Age: {age}, "
                             f"Diabetes: {diabetes}, Previous Diseases: {previous_diseases}",
                             insights, interactions_dir_diagnosis)
            return render_template('diagnosis_results.html', insights=insights,
                                   gender=gender, age=age, diabetes=diabetes,
                                   previous_diseases=previous_diseases) 

    return render_template('diagnosis_index.html')

@app.route("/medicine", methods=["GET", "POST"])
@login_required
def medicine_index():
    if request.method == "POST":
        search_by = request.form.get("search_by")
        name = request.form.get("name")
        country = request.form.get("country") if search_by == "generic" else "N/A"

        info = get_medicine_info(name, search_by, country)
        save_interaction(f"Search By: {search_by}, Name: {name}, Country: {country}",
                         info, interactions_dir_medicine)
        return render_template("medicine_results.html", info=info)
    return render_template("medicine_index.html")

@app.route('/image', methods=['GET', 'POST'])
@login_required
def image_analysis():
    """Handles requests to the main page and processes image uploads."""
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type')
        custom_prompt = request.form.get('custom_prompt')

        try:
            image_data = request.files['image'].read()

            if analysis_type == 'custom_image':
                result = process_custom_image(image_data, custom_prompt)
            else:
                result = process_image(image_data, analysis_type)

            save_interaction(f"Analysis Type: {analysis_type}\nCustom Prompt: {custom_prompt}", result, interactions_dir_image_analysis)
            return render_template('image_results.html', analysis_result=result)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('image_index.html')

@app.route('/chat', methods=["GET", "POST"])
@login_required
def chatbot():
    """Handles chatbot interactions."""
    if request.method == "POST":
        user_input = request.form['prompt']
        selected_topic = request.form.get('topic')
        num_molecules = request.form.get('num-molecules')

        # Get molecule inputs if available
        molecules_data = []
        if num_molecules:
            for i in range(1, int(num_molecules) + 1):
                molecule_name = request.form.get(f'molecule-{i}')
                molecule_property = request.form.get(f'property-{i}')
                molecules_data.append((molecule_name, molecule_property))
				
        # Construct prompt incorporating molecule details
        if selected_topic:
            prompt = f"Considering the topic of {selected_topic}, "
            if molecules_data:
                prompt += "for the following molecules and their properties:\n"
                for i, (name, prop) in enumerate(molecules_data):
                    prompt += f"Molecule {i + 1}: {name}, Property: {prop}\n"
            prompt += user_input  # Add user input to the prompt
        else:
            prompt = user_input

        response_text = generate_response(prompt)
        save_interaction(user_input, response_text, interactions_dir_chat_history)
        return render_template("chat_results.html", user_input=user_input, response=response_text)
    else:
        return render_template("chat_index.html")

@app.route('/history')
@login_required
def history():
    """Renders the history page."""
    diagnosis_history = read_interactions_from_file(interactions_dir_diagnosis)
    medicine_history = read_interactions_from_file(interactions_dir_medicine)
    image_history = read_interactions_from_file(interactions_dir_image_analysis)
    chatbot_history = read_interactions_from_file(interactions_dir_chat_history)
    return render_template('history.html', 
                           diagnosis_history=diagnosis_history,
                           medicine_history=medicine_history, 
                           image_history=image_history, 
                           chatbot_history=chatbot_history)

@app.route('/history/<interaction_type>/<filename>')
@login_required
def download_interaction(interaction_type, filename):
    """Downloads a specific interaction as a text file."""
    if interaction_type == 'diagnosis':
        directory = interactions_dir_diagnosis
    elif interaction_type == 'medicine':
        directory = interactions_dir_medicine
    elif interaction_type == 'image':
        directory = interactions_dir_image_analysis
    elif interaction_type == 'chat':
        directory = interactions_dir_chat_history
    else:
        return "Invalid interaction type", 404  

    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return "File not found", 404 

    with open(filepath, 'r', encoding="utf-8") as f:
        interaction_content = f.read()

    response = make_response(interaction_content)
    response.headers['Content-Type'] = 'text/plain'  
    response.headers['Content-Disposition'] = f'attachment; filename={filename}' 
    return response
@app.route('/image1')
@login_required
def imagedet():
    return render_template('image_analysis.html')
# Route for the home page
@app.route('/brain')
@login_required
def brain_tu():
    return render_template('brain.html')

# Route for handling image uploads and predictions
@app.route('/predict_brain', methods=['POST'])
@login_required
def brain_predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Remove old image before saving the new one
        remove_old_image()

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict the uploaded image
        prediction = predict_image(filepath)
        return render_template('brain.html', prediction=prediction, image_path=filename)
    
    return redirect(request.url)
@app.route('/pneumonia')
@login_required
def pneumonia():
    return render_template('pneumonia.html')
@app.route('/predict_pne', methods=['POST'])
@login_required
def predict_pne():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_image_pne(filepath)
        return render_template('pneumonia.html', prediction=prediction, image_path=filename)
    
    return redirect(request.url)
if __name__ == '__main__':
    app.run(debug=True)
