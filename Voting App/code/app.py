from flask import Flask, render_template, request, session, flash, redirect, url_for, make_response,Response
import pymysql
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from datetime import datetime
from functools import wraps
import base64
import pdfkit
import time
import torch
import pickle
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import dlib
from scipy.spatial import distance
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import smtplib
import json
from flask_mail import Mail, Message
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from flask_cors import CORS 
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Text
from flask import jsonify
from dotenv import load_dotenv
import urllib.request 

# Load environment variables from a .env file (for local development)
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Set secret key
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# Enable CORS for all routes
CORS(app)

# Email configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'sandbox.smtp.mailtrap.io')
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT', 2525)
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', '48fd56811e318b')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '5a22a51459efe6')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

# SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://default:1IFCXU3etObd@ep-royal-block-a4v3zczb.us-east-1.aws.neon.tech:5432/verceldb?sslmode=require')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Create SQLAlchemy engine for pandas
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

# Face detection using MTCNN
mtcnn = MTCNN()

# Face recognition using FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Download and load model for dlib
model_url = os.getenv('MODEL_URL', 'https://your-local-development-url.com/shape_predictor_68_face_landmarks.dat')
model_local_path = 'models/shape_predictor_68_face_landmarks.dat'

if not os.path.exists(model_local_path):
    try:
        print(f"Downloading shape predictor model from {model_url}...")
        os.makedirs(os.path.dirname(model_local_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_local_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        raise

# Load the face detector and shape predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_local_path)
    print("Shape predictor loaded successfully.")
except Exception as e:
    print(f"Error loading shape predictor: {e}")
    raise

# Constants for eye aspect ratio (EAR) threshold and blinks
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0

CAMERA_WIDTH = 400
CAMERA_HEIGHT = 400
FRAME_SKIP_RATE = 2

required_consistent_matches = 3
blinks_required = 3
blink_total = 0

# Database models
class ElectionSchedule(db.Model):
    __tablename__ = 'election_schedule'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    election_title = db.Column(db.String(100), nullable=False)

class Student(db.Model):
    __tablename__ = 'student'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    first_name = db.Column(db.String(100))
    middle_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    cnic = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(100))
    phone_number = db.Column(db.String(15))
    department = db.Column(db.String(50))
    semester = db.Column(db.String(10))
    photo = db.Column(db.LargeBinary)  # Image stored as binary (BLOB)
    face_embedding = db.Column(db.LargeBinary)  # Embedding for face recognition

class Candidate(db.Model):
    __tablename__ = 'candidates'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    position = db.Column(db.String(100), nullable=False)
    member_name = db.Column(db.String(100), nullable=False)
    party_name = db.Column(db.String(100), nullable=True)
    picture = db.Column(db.LargeBinary, nullable=True)

class Vote(db.Model):
    __tablename__ = 'vote'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    position = db.Column(db.String(100), nullable=False)
    vote = db.Column(db.String(100), nullable=False)
    cnic = db.Column(db.String(100), nullable=False)

class ElectionRecord(db.Model):
    __tablename__ = 'election_records'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    election_date = db.Column(db.DateTime, nullable=False)
    election_title = db.Column(db.String(100), nullable=False)
    total_voters = db.Column(db.Integer, nullable=False)
    total_participants = db.Column(db.Integer, nullable=False)
    results = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'election_date': self.election_date,
            'election_title': self.election_title,
            'total_voters': self.total_voters,
            'total_participants': self.total_participants,
            'results': self.results
        }

# Decorator to ensure admin access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('IsAdmin'):
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin'))
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def initialize():
    session.setdefault('IsAdmin', False)
    session.setdefault('User', None)

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blinks(frame, gray_frame):
    global blink_counter, blink_total

    faces = detector(gray_frame, 0)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                blink_total += 1
                blink_counter = 0

        cv2.rectangle(frame, (left_eye[0][0], left_eye[0][1]), (left_eye[3][0], left_eye[3][1]), (0, 255, 0), 1)
        cv2.rectangle(frame, (right_eye[0][0], right_eye[0][1]), (right_eye[3][0], right_eye[3][1]), (0, 255, 0), 1)

    return blink_total >= blinks_required

def verify_face(cnic, frame):
    global blink_total
    blink_total = 0  # Reset blink counter

    consistent_matches = 0
    face_verified = False

    # Load student data from the database
    student = Student.query.filter_by(cnic=cnic).first()

    if not student:
        flash("No student found with the provided CNIC.", "danger")
        return False

    stored_embedding = pickle.loads(student.face_embedding)

    # Resize the input frame
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert frame to grayscale and RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using MTCNN
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = rgb_frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Resize and prepare the face for embedding extraction
            face_resized = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
            face_tensor.div_(255.0)

            # Extract the face embedding using a pre-trained model
            with torch.no_grad():
                face_embedding = resnet(face_tensor).numpy()

            # Calculate similarity between stored and captured face embeddings
            similarity = np.dot(stored_embedding, face_embedding.T) / (
                np.linalg.norm(stored_embedding) * np.linalg.norm(face_embedding)
            )

            if similarity > 0.7:  # Adjust similarity threshold as needed
                consistent_matches += 1
                if consistent_matches >= required_consistent_matches:
                    face_verified = True
                    break
            else:
                consistent_matches = 0

    if face_verified:
        # Perform blink detection for liveness check
        blink_check_passed = detect_blinks(frame, gray)

        if blink_check_passed:
            return True

    return False

@app.route('/confirm_voter', methods=['POST'])
def confirm_voter():
    cnic = session.get('voter_cnic')

    if not cnic:
        flash("No CNIC provided.", "danger")
        return redirect(url_for('registration'))

    # Start video capture to get a frame
    cap = cv2.VideoCapture(0)

    # Wait for the camera to warm up (1 second delay)
    time.sleep(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        flash("Failed to capture video frame.", "danger")
        return redirect(url_for('registration'))

    # Verify the voter's face using the captured frame
    if not verify_face(cnic, frame):
        flash("Face not verified. Please try again.", "danger")
        return redirect(url_for('registration'))

    # Check if the election is still active
    schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()

    if not schedule or datetime.now() > schedule.end_time:
        flash("Election time is over or not active.", "danger")
        return redirect(url_for('home'))

    return redirect(url_for('select_candidate'))

@app.route('/add_candidate', methods=['GET', 'POST'])
def add_candidate():
    if request.method == 'POST':
        position = request.form.get('position')
        member_name = request.form.get('member_name')
        party_name = request.form.get('party_name')
        picture = request.files.get('picture')  # Upload image

        if not position or not member_name or not party_name or not picture:
            flash('All fields must be filled.', 'danger')
            return redirect(url_for('add_candidate'))

        picture_data = picture.read()  # Read image as binary data (BLOB)
        new_candidate = Candidate(
            position=position,
            member_name=member_name,
            party_name=party_name,
            picture=picture_data
        )
        db.session.add(new_candidate)
        db.session.commit()
        flash('Candidate added successfully!', 'success')
    return render_template('add_candidate.html')

@app.route('/candidate_picture/<int:candidate_id>')
def candidate_picture(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    if candidate and candidate.picture:
        return Response(candidate.picture, mimetype='image/jpg')
    return "No picture available", 404

@app.route('/student_photo/<cnic>')
def student_photo(cnic):
    student = Student.query.filter_by(cnic=cnic).first()
    if student and student.photo:
        return Response(student.photo, mimetype='image/jpg')
    return "No photo available", 404

@app.route('/voter-details/<cnic>', methods=['GET'])
def voter_details(cnic):
    student = db.session.query(Student).filter_by(cnic=cnic).first()
    
    if student:
        # Convert photo to Base64 only if it exists and is not empty
        if student.photo:
            student.photo = base64.b64encode(student.photo).decode('utf-8')
        return render_template('voter_details.html', student=student)
    else:
        return "Student not found", 404

@app.route('/set_schedule', methods=['POST', 'GET'])
@admin_required
def set_schedule():
    if request.method == 'POST':
        action = request.form['action']
        start_time_str = request.form['start_time']
        end_time_str = request.form['end_time']
        election_title = request.form['election_title']

        try:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
            current_time = datetime.now()
        except ValueError:
            flash('Invalid date/time format.', 'danger')
            return redirect(url_for('set_schedule'))

        if start_time < current_time or end_time < current_time or end_time < start_time:
            flash('Invalid time range.', 'danger')
            return redirect(url_for('set_schedule'))

        if action == "schedule_new":
            clear_previous_results()
            new_schedule = ElectionSchedule(start_time=start_time, end_time=end_time, election_title=election_title)
            db.session.add(new_schedule)
            db.session.commit()
            flash('New election schedule set successfully!', 'success')
        elif action == "extend":
            schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()
            schedule.start_time = start_time
            schedule.end_time = end_time
            schedule.election_title = election_title
            db.session.commit()
            flash('Election time extended successfully!', 'success')
    return render_template('set_schedule.html')

def clear_previous_results():
    Vote.query.delete()
    db.session.commit()

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        cnic = request.form['cnic']
        student = Student.query.filter_by(cnic=cnic).first()

        if not student:
            flash("No student found with the provided CNIC.", "danger")
            return redirect(url_for('registration'))

        session['voter_cnic'] = cnic
        student_info = {
            'first_name': student.first_name,
            'middle_name': student.middle_name,
            'last_name': student.last_name
        }

        return render_template('confirm_voter.html', student=student_info)

    return render_template('registration.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/admin', methods=['POST', 'GET'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@voting.com' and password == 'admin':
            session['IsAdmin'] = True
            session['User'] = 'admin'
            flash('Admin login successful', 'success')
            return render_template('admin_dashboard.html', admin=session.get('IsAdmin', False))
    return render_template('admin.html', admin=session.get('IsAdmin', False))

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/')
@app.route('/home')
def home():
    schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()
    remaining_seconds = None
    if schedule:
        end_time = schedule.end_time
        current_time = datetime.now()
        if current_time < end_time:
            remaining_time = end_time - current_time
            remaining_seconds = int(remaining_time.total_seconds())
    return render_template('index.html', remaining_seconds=remaining_seconds)

# Add a new student
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        first_name = request.form['first_name']
        middle_name = request.form.get('middle_name', '')
        last_name = request.form['last_name']
        cnic = request.form['cnic']
        email = request.form['email']
        phone_number = request.form['phone_number']
        voter_id = request.form['voter_id']
        department = request.form['department']
        semester = request.form['semester']
        photo = request.files['photo']  # Photo as binary (BLOB)

        if not first_name or not last_name or not cnic or not photo:
            flash('All required fields must be filled.', 'danger')
            return redirect(url_for('add_student'))

        photo_data = photo.read()  # Read photo as binary data
        face_embedding = extract_face_embedding(photo_data)

        if face_embedding is None:
            flash("Face embedding extraction failed. Please try with a clearer image.", "danger")
            return redirect(url_for('add_student'))

        new_student = Student(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            cnic=cnic,
            email=email,
            phone_number=phone_number,
            voter_id=voter_id,
            department=department,
            semester=semester,
            photo=photo_data,  # Store photo as binary (BLOB)
            face_embedding=face_embedding
        )
        db.session.add(new_student)
        db.session.commit()
        flash('Student added successfully!', 'success')
    return render_template('add_student.html')

# Extract face embedding
def extract_face_embedding(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    face = mtcnn(img)
    if face is not None:
        face_embedding = resnet(face.unsqueeze(0)).detach().numpy()
        return pickle.dumps(face_embedding)
    else:
        return None


@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    cnic = session.get('voter_cnic')

    if not cnic:
        flash("No CNIC found in session. Please complete the registration process.", "error")
        return redirect(url_for('registration'))

    try:
        # Fetch candidates from the database using SQLAlchemy
        candidates = Candidate.query.all()
        if not candidates:
            flash('No candidates found.', 'warning')
            return redirect(url_for('home'))

        # Get all unique positions
        all_positions = {candidate.position for candidate in candidates}

        # Create a dictionary of positions and candidates
        position_nominees = {
            pos: [(cand.id, cand.member_name) for cand in candidates if cand.position == pos]
            for pos in all_positions
        }

        # Check if the voter has already voted using SQLAlchemy
        existing_vote = Vote.query.filter_by(cnic=cnic).first()

        if existing_vote:
            flash("You have already voted", "warning")
            return redirect(url_for('home'))

        if request.method == 'POST':
            votes = {position: request.form.get(position) for position in all_positions}

            # Insert votes into the database
            for position, vote in votes.items():
                new_vote = Vote(position=position, vote=vote, cnic=cnic)
                db.session.add(new_vote)
            db.session.commit()

            # Fetch the voter's email
            student = Student.query.filter_by(cnic=cnic).first()
            email = student.email if student else None

            if email:
                vote_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                send_vote_confirmation_email(email, votes, candidates, vote_time)

            return redirect(url_for('home'))

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return redirect(url_for('home'))

    # Render the page to select candidates
    return render_template('select_candidate.html', position_nominees=position_nominees)



# Function to attach candidate images to the email
def attach_candidate_images(votes, df_nom, msg):
    """
    Attaches candidate images to the email message.
    
    Args:
        votes (dict): The dictionary containing positions and selected candidates.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        msg (MIMEMultipart): The email message object where the images will be attached.
    """
    for position, candidate_id in votes.items():
        # Fetch the candidate's details based on their ID
        candidate = Candidate.query.get(candidate_id)

        if candidate and candidate.picture:
            # Convert candidate picture (BLOB) to base64
            picture_base64 = base64.b64encode(candidate.picture).decode('utf-8')

            # Attach the candidate image
            image = MIMEImage(base64.b64decode(picture_base64))
            image.add_header('Content-ID', f"<candidate_{candidate_id}>")
            msg.attach(image)
        else:
            print(f"Error attaching image for candidate {candidate_id}")


def create_vote_email_html(votes, df_nom, vote_time):
    """
    Creates the HTML email body with candidate images embedded.

    Args:
        votes (dict): The dictionary containing positions and selected candidates.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        vote_time (str): The timestamp when the vote was cast.

    Returns:
        str: The HTML content of the email.
    """
    vote_details_html = ""
    added_positions = set()

    for position, candidate_id in votes.items():
        if position in added_positions:
            continue
        added_positions.add(position)

        # Fetch the candidate's details from the database
        candidate = Candidate.query.get(candidate_id)

        # Build HTML block for each candidate
        if candidate and candidate.picture:
            vote_details_html += f"""
            <p><strong>{position}</strong>: {candidate.member_name}<br>
            <img src="cid:candidate_{candidate_id}" style="width: 100px; height: auto;"></p>
            """
        else:
            vote_details_html += f"""
            <p><strong>{position}</strong>: {candidate.member_name} (No image available)</p>
            """

    message_body_html = f"""
    <html>
    <head></head>
    <body>
        <p>Dear voter,</p>
        <p>You have successfully cast your vote.</p>
        <p>Details:</p>
        {vote_details_html}
        <p>Time: {vote_time}</p>
        <p>Thank you for participating in the election!</p>
    </body>
    </html>
    """
    
    return message_body_html


# Revised function to send the confirmation email with candidate images
def send_vote_confirmation_email(email, votes, df_nom, vote_time):
    """
    Sends the voting confirmation email with attached candidate images.
    
    Args:
        email (str): The recipient's email address.
        votes (dict): The dictionary containing positions and selected candidates.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        vote_time (str): The timestamp when the vote was cast.
    """
    # Email subject
    subject = "Voting Confirmation for UOBS E-Voting"
    
    # Create the email message object
    msg = MIMEMultipart()
    msg['From'] = "UOBS E-Voting-System <info@scholarlink.biz>"
    msg['To'] = email
    msg['Subject'] = subject

    # Attach the HTML version with embedded images
    message_body_html = create_vote_email_html(votes, df_nom, vote_time)
    msg.attach(MIMEText(message_body_html, 'html'))

    # Attach candidate images
    attach_candidate_images(votes, df_nom, msg)

    # Send the email using SMTP
    try:
        sender_address = "info@scholarlink.biz"
        password = "B@ltist@n941"
        with smtplib.SMTP_SSL("smtp.stackmail.com", 465) as server:
            server.login(sender_address, password)
            server.sendmail(sender_address, email, msg.as_string())
        
        flash('Voted successfully! Confirmation email sent.', 'success')
    except smtplib.SMTPException as e:
        flash(f"An error occurred while sending the confirmation email: {e}", "error")


@app.route('/chart')
def chart():
    # Fetch the live vote count using SQLAlchemy
    votes = db.session.query(Vote.vote, db.func.count(Vote.vote).label('count')).group_by(Vote.vote).all()

    # Prepare labels and data for the chart
    labels = [vote.vote for vote in votes]
    data = [vote.count for vote in votes]

    return render_template('chart.html', labels=labels, data=data)

@app.route('/chart_data')
def chart_data():
    # Fetch the vote data using SQLAlchemy
    votes = db.session.query(Vote.vote, db.func.count(Vote.vote).label('count')).group_by(Vote.vote).all()

    labels = [vote.vote for vote in votes]
    data = [vote.count for vote in votes]

    return jsonify({'labels': labels, 'data': data})

@app.route('/voting_res')
def voting_res():
    # Get the election schedule and check the end time using SQLAlchemy
    schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()

    if not schedule:
        flash('Election schedule is not set.', 'warning')
        return redirect(url_for('home'))

    end_time = schedule.end_time
    election_title = schedule.election_title
    current_time = datetime.now()

    if current_time < end_time:
        return redirect(url_for('chart'))

    # Get total voters and participants using SQLAlchemy
    total_voters = db.session.query(db.func.count(Student.id)).scalar()
    total_participants = db.session.query(db.func.count(Vote.cnic.distinct())).scalar()

    # Fetch vote counts for each candidate using SQLAlchemy
    votes = db.session.query(
        Candidate.position, Candidate.member_name, Candidate.picture, 
        db.func.coalesce(db.func.count(Vote.vote), 0).label('count')
    ).outerjoin(Vote, Candidate.id == Vote.vote)\
     .group_by(Candidate.position, Candidate.member_name, Candidate.picture)\
     .order_by(Candidate.position, 'count').all()

    vote_results = {}
    for vote in votes:
        position, name, picture, count = vote

        # Convert candidate picture (BLOB) to base64 string
        if picture:
            picture_base64 = base64.b64encode(picture).decode('utf-8')
            picture_data = f"data:image/jpg;base64,{picture_base64}"
        else:
            picture_data = None  # Placeholder or default image can be used here

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append({'name': name, 'count': count, 'picture': picture_data})

    # Insert election record into the database
    election_record = {
        'election_date': current_time,
        'election_title': election_title,
        'total_voters': total_voters,
        'total_participants': total_participants,
        'results': json.dumps(vote_results)
    }

    new_record = ElectionRecord(
        election_date=election_record['election_date'],
        election_title=election_record['election_title'],
        total_voters=election_record['total_voters'],
        total_participants=election_record['total_participants'],
        results=election_record['results']
    )
    db.session.add(new_record)
    db.session.commit()

    return render_template('voting_res.html', 
                           vote_results=vote_results, 
                           total_voters=total_voters, 
                           total_participants=total_participants,
                           election_title=election_title,
                           chart_url=url_for('chart'))

import pdfkit

@app.route('/pdf_results')
def pdf_results():
    # Fetch election schedule using SQLAlchemy
    schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()

    if not schedule:
        flash('Election schedule is not set.', 'warning')
        return redirect(url_for('home'))

    end_time = schedule.end_time
    current_time = datetime.now()

    if current_time < end_time:
        flash('Election results are not available until the election ends.', 'warning')
        return redirect(url_for('home'))

    # Get total voters and participants
    total_voters = db.session.query(db.func.count(Student.id)).scalar()
    total_participants = db.session.query(db.func.count(Vote.cnic.distinct())).scalar()

    # Get voting results
    votes = db.session.query(
        Candidate.position, Candidate.member_name, Candidate.picture, 
        db.func.coalesce(db.func.count(Vote.vote), 0).label('count')
    ).outerjoin(Vote, Candidate.id == Vote.vote)\
     .group_by(Candidate.position, Candidate.member_name, Candidate.picture)\
     .order_by(Candidate.position, 'count').all()

    vote_results = {}
    for vote in votes:
        position, name, picture, count = vote

        # Convert candidate picture (BLOB) to base64 string for embedding in PDF
        if picture:
            picture_base64 = base64.b64encode(picture).decode('utf-8')
            picture_data = f"data:image/jpg;base64,{picture_base64}"
        else:
            picture_data = None  # Placeholder or default image

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append({'name': name, 'count': count, 'picture': picture_data})

    # Render the HTML template with embedded base64 images for PDF generation
    rendered_html = render_template('pdf_results.html', 
                                    vote_results=vote_results, 
                                    total_voters=total_voters, 
                                    total_participants=total_participants)

    # Generate PDF using pdfkit
    options = {
        'enable-local-file-access': ''
    }
    pdf = pdfkit.from_string(rendered_html, False, options=options)

    # Return the PDF as a downloadable response
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=election_results.pdf'

    return response

@app.route('/pdf_voters')
@admin_required
def pdf_voters():
    # Fetch all voters using SQLAlchemy
    voters = Student.query.all()

    rendered = render_template('pdf_voters.html', voters=voters)
    pdf = pdfkit.from_string(rendered, False)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=voters.pdf'
    return response


@app.route('/voters/<int:candidate_id>')
@admin_required
def voters_for_candidate(candidate_id):
    # Fetch candidate details including position using SQLAlchemy
    candidate = db.session.query(
        Candidate.member_name, Candidate.position, 
        db.func.count(Vote.vote).label('total_votes')
    ).outerjoin(Vote, Candidate.id == Vote.vote)\
     .filter(Candidate.id == candidate_id)\
     .group_by(Candidate.id, Candidate.member_name, Candidate.position)\
     .first()

    # Fetch voter details for the given candidate using SQLAlchemy
    voters = db.session.query(
        Student.first_name, Student.last_name, Student.cnic
    ).join(Vote, Vote.cnic == Student.cnic)\
     .filter(Vote.vote == candidate_id)\
     .all()

    return render_template('voters_for_candidate.html', candidate=candidate, voters=voters)


@app.route('/admin/end_election', methods=['POST'])
@admin_required
def end_election():
    # Update the election schedule's end time using SQLAlchemy
    schedule = ElectionSchedule.query.order_by(ElectionSchedule.id.desc()).first()
    if schedule:
        schedule.end_time = datetime.now()
        db.session.commit()

    # Fetch vote results using SQLAlchemy
    total_voters = db.session.query(db.func.count(Student.id)).scalar()
    total_participants = db.session.query(db.func.count(Vote.cnic.distinct())).scalar()

    # Get voting results, including candidates who received zero votes
    votes = db.session.query(
        Candidate.position, Candidate.member_name, 
        db.func.coalesce(db.func.count(Vote.vote), 0).label('count')
    ).outerjoin(Vote, Candidate.id == Vote.vote)\
     .group_by(Candidate.position, Candidate.member_name)\
     .order_by(Candidate.position, 'count').all()

    vote_results = {}
    result_data = ""
    for vote in votes:
        position, name, count = vote
        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append((name, count))
        result_data += f"Position: {position}, Candidate: {name}, Votes: {count}\n"

    # Insert election results into election_records
    new_record = ElectionRecord(
        election_date=datetime.now(),
        election_title=schedule.election_title,
        total_voters=total_voters,
        total_participants=total_participants,
        results=result_data
    )
    db.session.add(new_record)
    db.session.commit()

    flash('Election ended successfully!', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/election_records')
@admin_required
def election_records():
    records = ElectionRecord.query.all()
    records_list = [record.to_dict() for record in records]
    return render_template('election_records.html', records=records_list)


@app.route('/election_record_details/<int:record_id>')
@admin_required
def election_record_details(record_id):
    record = ElectionRecord.query.get(record_id)

    if not record:
        flash("No record found with the provided ID.", "danger")
        return redirect(url_for('election_records'))

    vote_results = json.loads(record.results)

    return render_template('election_record_details.html', record=record, vote_results=vote_results)


@app.route('/delete_student', methods=['GET', 'POST'])
def delete_student():
    if request.method == 'POST':
        cnic = request.form['cnic']

        # Check if student exists using SQLAlchemy
        student = Student.query.filter_by(cnic=cnic).first()

        if not student:
            flash("No student found with the provided CNIC.", "danger")
            return redirect(url_for('delete_student'))

        # Delete student record using SQLAlchemy
        db.session.delete(student)
        db.session.commit()

        flash(f'Student with CNIC {cnic} has been deleted successfully!', 'success')
        return redirect(url_for('delete_student'))

    return render_template('delete_student.html')


@app.route('/update_student', methods=['GET', 'POST'])
def update_student():
    if request.method == 'POST':
        cnic = request.form['cnic']

        # Fetch student details using SQLAlchemy
        student = Student.query.filter_by(cnic=cnic).first()

        if not student:
            flash("No student found with the provided CNIC.", "danger")
            return redirect(url_for('update_student'))

        # Update student details if fields are provided
        if request.form['email']:
            student.email = request.form['email']
        if request.form['phone_number']:
            student.phone_number = request.form['phone_number']

        # Handle photo update
        if 'photo' in request.files and request.files['photo'].filename != '':
            photo = request.files['photo']
            photo_data = photo.read()  # Store new photo as BLOB
            student.photo = photo_data

        db.session.commit()
        flash('Student details updated successfully!', 'success')

        return redirect(url_for('update_student'))

    return render_template('update_student.html')

app.config['TEMPLATES_AUTO_RELOAD'] = True
