from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from cohort import score_roster

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")
st.write("Add students to compare their estimated final exam scores.")
st.info("This starter project uses simulated practice data. Its predictions are for learning and are not validated for real student decisions.")

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.warning("The model has not been trained yet. In the project folder, run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


model = load_model(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
roster = st.session_state.setdefault("roster", {})
info = json.loads(INFO_PATH.read_text())
with st.form("prediction_form"):
    name = st.text_input("Student name")
    usn = st.text_input("University Seat Number (USN)").strip().upper()
    left, right = st.columns(2)
    with left:
        study = st.number_input("Study hours per day", min_value=0.0, max_value=16.0, value=3.0, step=0.5)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        previous = st.number_input("Previous exam score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with right:
        sleep = st.number_input("Sleep hours per night", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
        assignments = st.number_input("Assignments completed (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    submitted = st.form_submit_button("Predict and add student", type="primary", use_container_width=True)

if submitted:
    if not name.strip() or not usn:
        st.error("Enter both the student name and USN.")
    elif usn in roster:
        st.error("This USN is already in the class. To correct the entry, remove it below and add it again.")
    else:
        roster[usn] = {
            "Student name": name.strip(), "USN": usn, "study_hours": study,
            "attendance_percent": attendance, "previous_score": previous,
            "sleep_hours": sleep, "assignments_completed_percent": assignments,
        }
        st.success(f"Added {name.strip()} ({usn}).")

st.caption("The class list lasts for this browser session. Refreshing the page clears it.")
if roster:
    st.header("Class comparison")
    results, average = score_roster(roster.values(), model, info["features"])
    latest = next(reversed(roster))
    latest_score = results.loc[results["USN"] == latest, "Predicted score"].iloc[0]
    left, right = st.columns(2)
    left.metric(f"Estimate for {roster[latest]['Student name']}", f"{latest_score:.1f} / 100")
    right.metric(f"Class average · {len(roster)} student(s)", f"{average:.1f} / 100")
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.subheader("Academic check-in list")
    if len(roster) < 2:
        st.info("Add a second student to compare scores.")
    else:
        rule = st.radio("Select students by predicted score", [
            "Below class average", "Within 5 marks of class average", "Below a target score"
        ], horizontal=True)
        if rule == "Below class average":
            selected = results[results["Predicted score"] < average]
        elif rule == "Within 5 marks of class average":
            selected = results[results["Marks from average"].abs() <= 5]
        else:
            target = st.number_input("Target score out of 100", 0.0, 100.0, 50.0, step=1.0)
            selected = results[results["Predicted score"] < target]
        st.caption("Use these estimates to start a conversation. Check actual coursework and ask each student what support would help.")
        if selected.empty:
            st.info("No students meet the selected rule.")
        else:
            st.dataframe(selected, hide_index=True, use_container_width=True)
    with st.expander("Remove a student"):
        remove = st.selectbox("Select USN", list(roster))
        if st.button("Remove selected student"):
            del roster[remove]
            st.rerun()

with st.expander("Model comparison"):
    st.write(f"Selected model: {info['model_name']}")
    st.dataframe(pd.DataFrame(info["metrics"]).T, use_container_width=True)
