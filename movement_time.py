import cv2
import streamlit as st
import time
import numpy as np

st.title("Event-based Movement Time Counter")

if "running" not in st.session_state:
    st.session_state.running = False
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0
if "last_movement_time" not in st.session_state:
    st.session_state.last_movement_time = None
if "movement_active" not in st.session_state:
    st.session_state.movement_active = False

start = st.button("▶️ Start")
stop = st.button("⏹️ Stop")

# Toggle for output mode
view_mode = st.selectbox("View mode", ["Normal", "Event-based"])

frame_placeholder = st.empty()
time_placeholder = st.empty()

if start:
    st.session_state.running = True
    st.session_state.total_time = 0.0
    st.session_state.last_movement_time = None
    st.session_state.movement_active = False

if stop:
    st.session_state.running = False
    st.write(f"✅ Total movement time: **{st.session_state.total_time:.2f} seconds**")

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prev_frame = None

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # smaller threshold values → more sensitive
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
        movement_detected = np.sum(thresh) > 15000  # lower sensitivity threshold

        now = time.time()

        if movement_detected and not st.session_state.movement_active:
            st.session_state.movement_active = True
            st.session_state.last_movement_time = now

        elif not movement_detected and st.session_state.movement_active:
            st.session_state.movement_active = False
            st.session_state.total_time += now - st.session_state.last_movement_time
            st.session_state.last_movement_time = None

        if st.session_state.movement_active:
            st.session_state.total_time += now - st.session_state.last_movement_time
            st.session_state.last_movement_time = now

        # Display depending on view mode
        if view_mode == "Normal":
            frame_to_show = frame
        else:  # Event-based view
            frame_to_show = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        frame_placeholder.image(frame_to_show, channels="BGR")
        time_placeholder.write(f"⏱ Movement time: **{st.session_state.total_time:.2f} seconds**")

        prev_frame = gray
        cv2.waitKey(30)

    cap.release()
