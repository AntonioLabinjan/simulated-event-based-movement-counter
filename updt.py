import cv2
import streamlit as st
import time
import numpy as np

st.set_page_config(layout="wide")
st.title("⚡ Event-based Movement + Simulated Spiking CNN (LIF-style)")

# --- Session state init ---
if "running" not in st.session_state:
    st.session_state.running = False
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0
if "last_movement_time" not in st.session_state:
    st.session_state.last_movement_time = None
if "movement_active" not in st.session_state:
    st.session_state.movement_active = False
if "triggers" not in st.session_state:
    st.session_state.triggers = 0
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- Controls ---
col_left, col_right = st.columns([1, 1])

with col_left:
    start = st.button("▶️ Start", key="start")
    stop = st.button("⏹️ Stop", key="stop")
    reset_triggers = st.button("🔄 Reset triggers", key="reset_triggers")

with col_right:
    view_mode = st.selectbox("View mode", ["Normal", "Event-based", "Spiking (overlay)"])
    st.markdown("**Status:**")
    status_text = st.empty()

# Sidebar for fine tuning
st.sidebar.header("⚙️ Sensitivity + Spiking Settings")
pixel_diff_threshold = st.sidebar.slider("Pixel Intensity Threshold", 1, 50, 15)
area_threshold = st.sidebar.slider("Pixel Change Area (sum of pixels)", 1000, 100000, 12000, step=1000)
resize_width = st.sidebar.selectbox("Processing width (px)", [160, 240, 320, 480], index=2)

# Spiking sim params
st.sidebar.markdown("**Spiking CNN (LIF) params**")
decay = st.sidebar.slider("Membrane decay (alpha)", 0.80, 0.999, 0.92)
layer1_thr = st.sidebar.slider("Layer1 spike threshold", 5.0, 100.0, 30.0)
layer2_thr = st.sidebar.slider("Layer2 spike threshold", 5.0, 500.0, 120.0)
refractory_time = st.sidebar.slider("Refractory frames (per neuron)", 0, 10, 2)
trigger_spike_count = st.sidebar.slider("Trigger when final spikes >", 1, 500, 60)

# kernels for "convolution"
k1 = st.sidebar.selectbox("Layer1 kernel", ["sobel", "gaussian", "box"], index=0)
k2 = st.sidebar.selectbox("Layer2 kernel", ["box", "gaussian"], index=0)

# placeholders
frame_col1, frame_col2 = st.columns(2)
frame_placeholder = frame_col1.empty()
spike_placeholder = frame_col2.empty()
time_placeholder = st.empty()
trigger_placeholder = st.empty()
log_placeholder = st.empty()

# handle start/stop/reset
if start:
    st.session_state.running = True
    st.session_state.total_time = 0.0
    st.session_state.last_movement_time = None
    st.session_state.movement_active = False
    st.session_state.triggers = 0
    st.session_state.logs = []
    # will (re)init mem potentials on first frame

if stop:
    st.session_state.running = False
    st.session_state.logs.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "movement_time_s": round(st.session_state.total_time, 3),
        "triggers": st.session_state.triggers
    })

if reset_triggers:
    st.session_state.triggers = 0

# helper kernels
def get_kernel(name, ksize=3):
    if name == "sobel":
        # simple gradient magnitude kernel (approx)
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        ky = kx.T
        return kx, ky
    elif name == "gaussian":
        k = cv2.getGaussianKernel(ksize, -1)
        return (k @ k.T).astype(np.float32)
    else:
        return np.ones((ksize, ksize), dtype=np.float32) / (ksize*ksize)

# main loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prev_frame = None

    # initialize membrane potentials and refractory maps after we know frame size
    mem1 = None
    mem2 = None
    refr1 = None
    refr2 = None

    # fps limiter
    last_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_text.info("Camera not available.")
            break

        # resize for faster processing
        h0, w0 = frame.shape[:2]
        scale = resize_width / w0
        frame_small = cv2.resize(frame, (resize_width, int(h0*scale)))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if prev_frame is None:
            prev_frame = gray.copy()
            # init mem potentials/resets
            mem1 = np.zeros_like(gray, dtype=np.float32)
            mem2 = np.zeros_like(gray, dtype=np.float32)
            refr1 = np.zeros_like(gray, dtype=np.int16)
            refr2 = np.zeros_like(gray, dtype=np.int16)
            continue

        # event mask (pixel difference)
        frame_delta = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_delta, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
        movement_strength = np.sum(thresh)

        # movement detected boolean
        movement_detected = movement_strength > area_threshold

        now = time.time()

        # movement time accumulation (same logic)
        if movement_detected and not st.session_state.movement_active:
            st.session_state.movement_active = True
            st.session_state.last_movement_time = now

        elif (not movement_detected) and st.session_state.movement_active:
            st.session_state.movement_active = False
            st.session_state.total_time += now - st.session_state.last_movement_time
            st.session_state.last_movement_time = None

        if st.session_state.movement_active:
            st.session_state.total_time += now - st.session_state.last_movement_time
            st.session_state.last_movement_time = now

        # --- Simulated spiking pipeline ---
        # Layer 1: convolve event mask (or gradient)
        # pick kernel
        if isinstance(get_kernel(k1), tuple) and k1 == "sobel":
            kx, ky = get_kernel("sobel")
            gradx = cv2.filter2D(thresh.astype(np.float32), -1, kx)
            grady = cv2.filter2D(thresh.astype(np.float32), -1, ky)
            conv1 = np.sqrt(np.square(gradx) + np.square(grady))
        else:
            kern1 = get_kernel(k1, ksize=3)
            conv1 = cv2.filter2D(thresh.astype(np.float32), -1, kern1)

        # normalize conv for stability
        conv1 = conv1 / (conv1.max()+1e-6) * 50.0  # scale to reasonable input

        # LIF update layer1
        # refractory decrement
        refr1 = np.maximum(0, refr1 - 1)
        mem1 = mem1 * decay + conv1
        # neurons that are in refractory can't spike; mask them
        can_spike1 = (refr1 == 0)
        spikes1 = (mem1 >= layer1_thr) & can_spike1
        spikes1 = spikes1.astype(np.float32)
        # reset membrane where spike occurred & set refractory
        mem1[spikes1 == 1] = 0.0
        refr1[spikes1 == 1] = refractory_time

        # Layer2: pool/blur + conv
        if k2 == "gaussian":
            kern2 = get_kernel("gaussian", ksize=5)
            conv2 = cv2.filter2D(spikes1, -1, kern2)
        else:
            conv2 = cv2.blur(spikes1, (3,3))

        conv2 = conv2 * 200.0  # scale up for layer2 drive

        # LIF update layer2
        refr2 = np.maximum(0, refr2 - 1)
        mem2 = mem2 * decay + conv2
        can_spike2 = (refr2 == 0)
        spikes2 = (mem2 >= layer2_thr) & can_spike2
        spikes2 = spikes2.astype(np.float32)
        mem2[spikes2 == 1] = 0.0
        refr2[spikes2 == 1] = refractory_time

        # final spike count
        final_spike_count = int(np.sum(spikes2))

        # Trigger logic
        triggered_now = False
        if final_spike_count >= trigger_spike_count:
            st.session_state.triggers += 1
            triggered_now = True

        # --- Visualization building ---
        # upscale visuals to original small frame size (for display)
        vis_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # colorize spikes1 and spikes2 for overlay
        s1_vis = (spikes1 * 255).astype(np.uint8)
        s1_vis = cv2.cvtColor(s1_vis, cv2.COLOR_GRAY2BGR)
        s2_vis = (spikes2 * 255).astype(np.uint8)
        s2_vis = cv2.cvtColor(s2_vis, cv2.COLOR_GRAY2BGR)

        # color overlays: ticks where spikes happened
        overlay = frame_small.copy()
        if view_mode == "Spiking (overlay)":
            # overlay spikes1 in green, spikes2 in red (alpha)
            alpha = 0.6
            overlay = cv2.addWeighted(overlay.astype(np.float32), 1.0, s1_vis.astype(np.float32), alpha, 0).astype(np.uint8)
            overlay = cv2.addWeighted(overlay.astype(np.float32), 1.0, s2_vis.astype(np.float32), alpha, 0).astype(np.uint8)

        # prepare side-by-side: original, thresh, spikes1, spikes2 (all same height)
        small_orig = cv2.resize(frame_small, (frame_small.shape[1], frame_small.shape[0]))
        comb1 = cv2.resize(small_orig, (0,0), fx=1.0, fy=1.0)
        comb2 = cv2.resize(vis_thresh, (comb1.shape[1], comb1.shape[0]))
        comb3 = cv2.resize((np.clip(mem1,0,255)).astype(np.uint8), (comb1.shape[1], comb1.shape[0]))
        comb3 = cv2.cvtColor(comb3, cv2.COLOR_GRAY2BGR)
        comb4 = cv2.resize((np.clip(mem2/np.max(mem2+1e-6)*255,0,255)).astype(np.uint8), (comb1.shape[1], comb1.shape[0]))
        comb4 = cv2.cvtColor(comb4, cv2.COLOR_GRAY2BGR)

        left_vis = cv2.hconcat([comb1, comb2])
        right_vis = cv2.hconcat([comb3, comb4])
        final_vis = cv2.hconcat([left_vis, right_vis])

        # If view mode normal or event-based, show appropriate image in main placeholder
        if view_mode == "Normal":
            display_img = cv2.resize(frame_small, (640, 480))
        elif view_mode == "Event-based":
            display_img = cv2.resize(vis_thresh, (640, 480))
        else:  # Spiking overlay
            display_img = cv2.resize(overlay, (640, 480))

        # Draw status boxes
        if triggered_now:
            cv2.putText(display_img, "TRIGGERED!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        cv2.putText(display_img, f"Movement: {int(movement_strength)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display_img, f"Final spikes: {final_spike_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(display_img, f"Triggers total: {st.session_state.triggers}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # show combined grid (small) in the second column
        grid_small = cv2.resize(final_vis, (640, 240))

        # streamlit display
        frame_placeholder.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), channels="RGB")
        spike_placeholder.image(cv2.cvtColor(grid_small, cv2.COLOR_BGR2RGB), channels="RGB")
        time_placeholder.markdown(f"**Movement time:** {st.session_state.total_time:.2f} s  •  **Triggered:** {st.session_state.triggers}")
        if triggered_now:
            trigger_placeholder.success(f" Triggered at {time.strftime('%H:%M:%S')}  • Final spikes: {final_spike_count}")

        # logs table
        if len(st.session_state.logs) > 0:
            log_placeholder.table(st.session_state.logs[-10:])

        prev_frame = gray.copy()

        # small sleep for stability
        key = cv2.waitKey(1)
        # limit loop to ~30-60 fps
        time.sleep(max(0.001, 1/30 - (time.time() - last_time)))
        last_time = time.time()

    cap.release()
    status_text.info("Stopped.")
else:
    status_text.info("Idle. Press ▶️ Start to run.")
    if len(st.session_state.logs) > 0:
        log_placeholder.table(st.session_state.logs[-10:])
