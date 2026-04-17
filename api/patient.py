from flask import Blueprint, request, jsonify, session, Response
import pandas as pd
import json
import bcrypt
from datetime import datetime
from config.database_config import get_db_connection
from models.sepsis_predictor import SepsisPredictor
from utils.helpers import Helpers
from chatbot.nlp_processor import NLPProcessor
from chatbot.response_generator import ResponseGenerator
from chatbot.sepsis_doc_agent import SepsisDocAgent
from utils.constants import CHATBOT_INTENTS
import hashlib
import base64
response_gen = ResponseGenerator()
doc_agent = SepsisDocAgent()

nlp_processor = NLPProcessor()
from werkzeug.security import check_password_hash, generate_password_hash


patient_bp = Blueprint("patient", __name__)

sepsis_predictor = SepsisPredictor()
helpers = Helpers()

nlp_loaded = False
model_loaded = False
feature_names = []



from flask import request

@patient_bp.route("/patient/submit_vitals", methods=["POST"])
def submit_vitals():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    data = request.json

    # 🔹 Extract vitals from request
    temperature = data.get("temperature")
    heart_rate = data.get("heart_rate")
    respiratory_rate = data.get("respiratory_rate")
    o2_saturation = data.get("o2_saturation")

    if not all([temperature, heart_rate, respiratory_rate, o2_saturation]):
        return jsonify({"error": "Please provide all vitals"}), 400

    # 🔹 Save vitals in DB
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO vitals (user_id, temperature, heart_rate, respiratory_rate, o2_saturation, source, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            temperature,
            heart_rate,
            respiratory_rate,
            o2_saturation,
            "manual",
            datetime.now()
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Error saving vitals:", str(e))
        return jsonify({"error": "Could not save vitals"}), 500

    return jsonify({"message": "Vitals submitted successfully", "vitals": data})

@patient_bp.route("/patient/predict", methods=["POST"])
def patient_predict():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    try:
        global model_loaded, feature_names
        if not model_loaded:
            sepsis_predictor.load_model("models/saved_models/sepsis_model.pkl")
            with open("models/saved_models/feature_names.json") as f:
                feature_names = json.load(f)
            model_loaded = True

        data = request.json
        patient_features = helpers.prepare_patient_features(data)
        fixed = {f: patient_features.get(f, 0) for f in feature_names}
        X = pd.DataFrame([fixed])

        result = sepsis_predictor.predict_single(X.iloc[0].to_dict(), feature_names, threshold=0.5)

        # ⭐ SAVE TO DATABASE
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
    INSERT INTO prediction_history (
        user_id,
        prediction,
        probability,
        risk_level,
        heart_rate,
        respiratory_rate,
        blood_pressure,
        temperature,
        notes,
        timestamp
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", (
    session["user_id"],
    result["prediction"],
    float(result["probability"]),
    result["risk_level"],
    data.get("heart_rate"),
    data.get("respiratory_rate"),
    f"{data.get('systolic_bp')}/{data.get('diastolic_bp')}",
    data.get("temperature"),
    "AI Sepsis Risk Assessment",
    datetime.now()
))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "probability": float(result["probability"]),
            "risk_level": result["risk_level"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@patient_bp.route("/patient/profile", methods=["POST"])
def create_or_update_patient_profile():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    user_id = session["user_id"]
    data = request.json

    conn = get_db_connection()
    cur = conn.cursor()

    # 🔎 Check existing profile
    cur.execute(
        "SELECT id FROM patient_profiles WHERE user_id=%s",
        (user_id,)
    )
    existing = cur.fetchone()

    if existing:
        # 🔄 UPDATE
        cur.execute("""
            UPDATE patient_profiles SET
                full_name=%s,
                gender=%s,
                date_of_birth=%s,
                age=%s,
                blood_group=%s,
                contact_country_code=%s,
                contact_number=%s,
                admission_type=%s,
                address=%s,
                updated_at=%s
            WHERE user_id=%s
        """, (
            data.get("full_name"),
            data.get("gender"),
            data.get("dob"),
            data.get("age"),
            data.get("blood_group"),
            data.get("contact_country_code"),
            data.get("contact_number"),
            data.get("admission_type"),
            data.get("address"),
            datetime.now(),
            user_id
        ))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "message": "Profile updated"})

    else:
        # ➕ CREATE
        cur.execute("""
            INSERT INTO patient_profiles
            (user_id, full_name, gender, date_of_birth, age, blood_group,
             contact_country_code, contact_number, admission_type, address,
             created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id,
            data.get("full_name"),
            data.get("gender"),
            data.get("dob"),
            data.get("age"),
            data.get("blood_group"),
            data.get("contact_country_code"),
            data.get("contact_number"),
            data.get("admission_type"),
            data.get("address"),
            datetime.now(),
            datetime.now()
        ))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "message": "Profile created"})
    
@patient_bp.route("/patient/profile", methods=["GET"])
def get_patient_profile():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if session.get("role") != "patient":
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT full_name, gender, date_of_birth, age, blood_group,
               contact_country_code, contact_number, admission_type, address
        FROM patient_profiles
        WHERE user_id=%s
    """, (session["user_id"],))

    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return jsonify({"exists": False})

    return jsonify({
        "exists": True,
        "profile": {
            "full_name": row[0],
            "gender": row[1],
            "dob": row[2],
            "age": row[3],
            "blood_group": row[4],
            "contact_country_code": row[5],
            "contact_number": row[6],
            "admission_type": row[7],
            "address": row[8],
        }
    })


@patient_bp.route("/patient/history")
def patient_history():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            heart_rate,
            respiratory_rate,
            blood_pressure,
            temperature,
            risk_level,
            notes,
            timestamp
        FROM prediction_history 
        WHERE user_id=%s 
        ORDER BY timestamp DESC
    """, (session["user_id"],))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = [
        {
            "heart_rate": r[0],
            "respiratory_rate": r[1],
            "blood_pressure": r[2],
            "temperature": r[3],
            "risk_score": r[4],
            "notes": r[5],
            "timestamp": r[6].strftime("%Y-%m-%d %H:%M")
        }
        for r in rows
    ]

    return jsonify(data)


@patient_bp.route("/patient/report")
def patient_report():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    conn = get_db_connection()
    cur = conn.cursor()

    # 1️⃣ Latest prediction for this user
    cur.execute("""
        SELECT prediction, probability, risk_level, timestamp
        FROM prediction_history
        WHERE user_id=%s
        ORDER BY timestamp DESC LIMIT 1
    """, (user_id,))
    row = cur.fetchone()

    latest_prediction = {
        "prediction": row[0] if row else None,
        "probability": float(row[1]) if row and row[1] is not None else 0,
        "risk_level": row[2] if row else "None",
        "timestamp": row[3].isoformat() if row and row[3] else None
    }

    # 2️⃣ Aggregate stats for this user
    cur.execute("""
        SELECT COUNT(*), AVG(probability), MAX(risk_level)
        FROM prediction_history
        WHERE user_id=%s
    """, (user_id,))
    total, avg_prob, high_risk = cur.fetchone()
    
    # 3️⃣ Latest vitals from patient_vitals table
    cur.execute("""
        SELECT temperature, heart_rate, respiratory_rate, o2_saturation, timestamp
        FROM patient_vitals
        WHERE user_id=%s
        ORDER BY timestamp DESC LIMIT 1
    """, (user_id,))
    vitals_row = cur.fetchone()

    latest_vitals = {
        "temperature": float(vitals_row[0]) if vitals_row and vitals_row[0] is not None else None,
        "heart_rate": vitals_row[1] if vitals_row and vitals_row[1] is not None else None,
        "respiratory_rate": vitals_row[2] if vitals_row and vitals_row[2] is not None else None,
        "o2_saturation": vitals_row[3] if vitals_row and vitals_row[3] is not None else None,
        "timestamp": vitals_row[4].isoformat() if vitals_row and vitals_row[4] else None
    }

    cur.close()
    conn.close()

    return jsonify({
        "total_predictions": total or 0,
        "average_probability": round((avg_prob or 0) * 100, 2),
        "highest_risk": high_risk or "None",
        "latest_prediction": latest_prediction,
        "latest_vitals": latest_vitals
    })



@patient_bp.route("/patient/simulate")
def simulate():
    import random
    return jsonify({
        "temperature": round(random.uniform(36, 40), 1),
        "heart_rate": random.randint(60, 140),
        "respiratory_rate": random.randint(10, 32),
        "o2_saturation": random.randint(80, 100)
    })

def detect_intent(message: str):
    msg = message.lower().strip()

    # Greeting
    if any(w in msg for w in ["hi", "hello", "hey"]):
        return CHATBOT_INTENTS['GREETING']

    # Direct definition question
    if "what is sepsis" in msg or "define sepsis" in msg:
        return "DOC_QUERY"

    # Symptoms
    if any(w in msg for w in ["symptom", "sign", "feel", "fever", "pain"]):
        return CHATBOT_INTENTS['SYMPTOMS']

    # Prevention
    if any(w in msg for w in ["prevent", "prevention", "avoid", "protection", "stop sepsis"]):
        return CHATBOT_INTENTS['PREVENTION']

    # Causes  (keep BEFORE generic why/reason logic)
    if any(w in msg for w in ["cause of sepsis", "causes of sepsis", "why does sepsis happen", "how sepsis happens"]):
        return "CAUSES"

    # Treatment
    if any(w in msg for w in ["treat", "treatment", "cure", "medicine", "antibiotic"]):
        return CHATBOT_INTENTS['TREATMENT']

    # Risk prediction
    if any(w in msg for w in ["risk", "risks", "chance", "probability"]):
        return CHATBOT_INTENTS['SEPSIS_RISK']

    # If message is long → send to document QA
    if len(msg.split()) > 4:
        return "DOC_QUERY"

    # Otherwise fallback
    return CHATBOT_INTENTS['HELP']



@patient_bp.route("/patient/assistant", methods=["POST"])
def assistant():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    text = request.json.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Please enter a question."})

    # Detect user intent
    intent = detect_intent(text)

    # 🟢 Direct response intents (no document search)
    if intent in [
        CHATBOT_INTENTS['GREETING'],
        CHATBOT_INTENTS['SYMPTOMS'],
        CHATBOT_INTENTS['PREVENTION'],
        CHATBOT_INTENTS['TREATMENT'],
        CHATBOT_INTENTS['HELP'],
        CHATBOT_INTENTS['GOODBYE'],
        "CAUSES"
    ]:
        response = response_gen.generate_response(
            intent=intent,
            original_message=text
        )
        return jsonify({
            "reply": response["response"],
            "intent": intent,
            "suggestions": response["suggestions"]
        })

    # 🔵 Document-based answers (only DOC_QUERY)
    if intent == "DOC_QUERY":
        try:
            doc_answer = doc_agent.answer(text)
        except Exception as e:
            print("❌ Document agent error:", str(e))
            doc_answer = None

        # Fallback if document agent fails
        if not doc_answer or "could not find information" in doc_answer.lower():
            fallback = response_gen.generate_response(
                intent=CHATBOT_INTENTS['HELP'],
                original_message=text
            )
            return jsonify({
                "reply": fallback["response"],
                "intent": "fallback",
                "suggestions": fallback["suggestions"]
            })

        return jsonify({
            "reply": doc_answer,
            "intent": "DOC_QUERY",
            "suggestions": response_gen._generate_suggestions(
                CHATBOT_INTENTS['SYMPTOMS']
            )
        })

    # 🔁 Catch-all fallback for any other cases
    fallback = response_gen.generate_fallback_response(text)
    return jsonify({
        "reply": fallback["response"],
        "intent": fallback["data"]["intent"],
        "suggestions": fallback["suggestions"]
    })


@patient_bp.route("/patient/settings", methods=["POST"])
def settings_update():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE users
        SET email_alerts=%s,
            sms_alerts=%s,
            weekly_report=%s
        WHERE id=%s
    """, (
        data.get("email_alerts"),
        data.get("sms_alerts"),
        data.get("weekly_report"),
        session["user_id"]
    ))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"saved": True})


@patient_bp.route("/patient/update-password", methods=["POST"])
def update_password():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_password = data.get("current_password")
    new_password = data.get("new_password")

    if not current_password or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT password_hash FROM users WHERE id=%s", (session["user_id"],))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    stored_hash = row[0]

    # ✅ CORRECT CHECK
    if not check_password_hash(stored_hash, current_password):
        cur.close()
        conn.close()
        return jsonify({"error": "Current password is incorrect"}), 403

    # ✅ CORRECT HASHING
    new_hash = generate_password_hash(new_password)

    cur.execute(
        "UPDATE users SET password_hash=%s WHERE id=%s",
        (new_hash, session["user_id"])
    )
    conn.commit()

    cur.close()
    conn.close()

    return jsonify({"success": True, "message": "Password updated successfully"})

@patient_bp.route("/patient/update-email", methods=["POST"])
def update_email():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_email = data.get("current_email", "").strip()
    new_email = data.get("new_email", "").strip()
    confirm_new_email = data.get("confirm_new_email", "").strip()

    if not current_email or not new_email or not confirm_new_email:
        return jsonify({"error": "Current email, new email, and confirmation are required"}), 400

    if new_email != confirm_new_email:
        return jsonify({"error": "New email and confirmation do not match"}), 400

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1️⃣ Get email from DB
            cur.execute("SELECT email FROM users WHERE id=%s", (session["user_id"],))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "User not found"}), 404

            stored_email = row[0]

            # 2️⃣ Verify current email
            if current_email != stored_email:
                return jsonify({"error": "Current email does not match"}), 403

            # 3️⃣ Check if new email is already used by someone else
            cur.execute("SELECT id FROM users WHERE email=%s", (new_email,))
            if cur.fetchone():
                return jsonify({"error": "New email already in use"}), 409

            # 4️⃣ Update email
            cur.execute(
                "UPDATE users SET email=%s WHERE id=%s",
                (new_email, session["user_id"])
            )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Email updated successfully"})



@patient_bp.route("/patient/update-username", methods=["POST"])
def update_username():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    current_username = data.get("current_username")
    new_username = data.get("new_username")

    if not current_username or not new_username:
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # 🔎 Verify current username
    cur.execute(
        "SELECT username FROM users WHERE id=%s",
        (session["user_id"],)
    )
    stored_username = cur.fetchone()[0]

    if current_username != stored_username:
        cur.close()
        conn.close()
        return jsonify({"error": "Current username is incorrect"}), 403

    # 🔎 Check duplicate username
    cur.execute(
        "SELECT id FROM users WHERE username=%s",
        (new_username,)
    )
    if cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"error": "Username already taken"}), 409

    # 🔄 Update username
    cur.execute(
        "UPDATE users SET username=%s WHERE id=%s",
        (new_username, session["user_id"])
    )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"success": True})

@patient_bp.route("/patient/export", methods=["GET"])
def export_data():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, heart_rate, bp, created_at
    FROM vitals
    WHERE user_id=%s
    """, (session["user_id"],))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","heart_rate","bp","created_at"])
    writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=data.csv"}
    )
