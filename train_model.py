import os
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


DATA_ROOT = r"C:\Users\Ayush Aggarwal\Desktop\SignLang_Converter\dataset"                              
METADATA_PATH = os.path.join(DATA_ROOT, "English_ISLgloss1.xlsx")
SEQUENCE_LENGTH = 30                              
TARGET_ACTIONS = None                             
MODEL_DIR = "model"
BATCH_SIZE = 8
EPOCHS = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42


os.makedirs(MODEL_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic

def read_metadata(metadata_path):
    df = pd.read_excel(metadata_path, header=None)
   
    header_row = None
    for i in range(min(6, len(df))):
        row = df.iloc[i].astype(str).str.lower().tolist()
        if any("sr.no" in str(v) for v in row) or any("english sentence" in str(v) for v in row):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not detect header row in the metadata Excel. Please check file.")
    df = pd.read_excel(metadata_path, header=header_row)
    
    cols = [c.strip().lower() for c in df.columns.tolist()]
    df.columns = cols
    
    if 'indian sign language gloss' in df.columns:
        df = df.dropna(subset=['indian sign language gloss'])
    else:
       
        possible = [c for c in df.columns if 'gloss' in c or 'sign' in c]
        if possible:
            df = df.dropna(subset=[possible[0]])
            df = df.rename(columns={possible[0]: 'indian sign language gloss'})
        else:
            raise ValueError("Could not find 'Indian sign language gloss' column in metadata.")
    df = df.reset_index(drop=True)
    return df

def normalize_gloss(text):
   
    t = str(text).strip().lower()
    t = t.replace('?', '').replace(',', '').replace('.', '').replace('-', ' ')
    t = '_'.join(t.split())
    return t

def get_video_frames(video_path, target_len):
    """Return exactly target_len frames (BGR) from the video path.
       If video shorter, will pad (repeat last frame). If longer, sample evenly."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise IOError("Video has 0 frames: " + video_path)
    
    if total >= target_len:
       
        indices = np.linspace(0, total - 1, target_len).astype(int).tolist()
    else:
       
        indices = list(range(total))
   
    idx_set = set(indices)
    cur = 0
    picked = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idx_set:
            picked.append(frame.copy())
        cur += 1
    cap.release()
    if len(picked) == 0:
        raise IOError("No frames picked from: " + video_path)
    
    while len(picked) < target_len:
        picked.append(picked[-1].copy())
    
    return picked[:target_len]

def extract_keypoints_from_frames(frames, holistic_model):
    """Given list of BGR frames, use holistic to extract keypoints vector per frame.
       Returns numpy array shape (frames_count, features)."""
    all_kps = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True
       
        if results.pose_landmarks:
            pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 3)
        
        if results.left_hand_landmarks:
            lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3)
       
        if results.right_hand_landmarks:
            rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3)
        keypoints = np.concatenate([pose, lh, rh])
        all_kps.append(keypoints)
    return np.array(all_kps)  

def build_dataset(metadata_df):
    actions = []
   
    mapping = {}
    for idx, row in metadata_df.iterrows():
        gloss_raw = row['indian sign language gloss']
        gloss = normalize_gloss(gloss_raw)
        if TARGET_ACTIONS and gloss not in TARGET_ACTIONS:
            continue
        
        video_file = None
        if 'file name' in metadata_df.columns:
            video_file = row.get('file name')
        if not video_file and 'sr.no' in metadata_df.columns:
            
            video_file = None
       
        mapping.setdefault(gloss, [])
    
    for entry in os.listdir(DATA_ROOT):
        entry_path = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(entry_path) and entry != os.path.basename(METADATA_PATH):
            
            mapping_key = entry.lower()
            if mapping_key not in mapping:
                
                mapping[normalize_gloss(entry)] = [os.path.join(entry_path, f) for f in os.listdir(entry_path) if f.lower().endswith(('.mp4','.mov','.avi','.mkv'))]
            else:
                mapping[mapping_key] = [os.path.join(entry_path, f) for f in os.listdir(entry_path) if f.lower().endswith(('.mp4','.mov','.avi','.mkv'))]
   
    all_files = [f for f in os.listdir(DATA_ROOT) if f.lower().endswith(('.mp4','.mov','.avi','.mkv'))]
    for gloss in list(mapping.keys()):
        if len(mapping.get(gloss, [])) == 0:
            
            gloss_tokens = gloss.replace('_', ' ').split()
            matched = []
            for fname in all_files:
                name = fname.lower()
                if all(tok in name for tok in gloss_tokens[:3]):
                    matched.append(os.path.join(DATA_ROOT, fname))
            if matched:
                mapping[gloss] = matched
   
    mapping = {k: v for k, v in mapping.items() if len(v) > 0}
    actions = sorted(list(mapping.keys()))
    print(f"Detected {len(actions)} actions (labels) with videos.")
    for a in actions:
        print(f" - {a}: {len(mapping[a])} video(s)")
    return mapping, actions

def create_sequences_and_labels(mapping, seq_len):
    X = []
    y = []
    total_v = sum(len(v) for v in mapping.values())
    print(f"Processing {total_v} videos to create sequences (this may take time)...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for label_idx, (gloss, videos) in enumerate(mapping.items()):
            for video_path in tqdm(videos, desc=f"Label {gloss}"):
                try:
                    frames = get_video_frames(video_path, seq_len)
                    keypoints_seq = extract_keypoints_from_frames(frames, holistic) 
                    if keypoints_seq.shape[0] != seq_len:
                       
                        pad_count = seq_len - keypoints_seq.shape[0]
                        last = keypoints_seq[-1]
                        pad = np.vstack([last]*pad_count)
                        keypoints_seq = np.vstack([keypoints_seq, pad])
                    X.append(keypoints_seq)
                    y.append(label_idx)
                except Exception as e:
                    print(f"Skipping {video_path}: {e}")
    X = np.array(X)  
    y = np.array(y)
    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y

def build_and_train(X, y, actions):
    # 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train_cat = to_categorical(y_train, num_classes=len(actions))
    y_test_cat = to_categorical(y_test, num_classes=len(actions))

    timesteps = X.shape[1]
    features = X.shape[2]


    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=(timesteps, features)),
        Dropout(0.4),
        LSTM(128, return_sequences=False, activation='tanh'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

   
    chk_path = os.path.join(MODEL_DIR, "sign_model_best.h5")
    checkpoint = ModelCheckpoint(chk_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early]
    )

    final_model_path = os.path.join(MODEL_DIR, "sign_model_final.h5")
    model.save(final_model_path)
   
    np.save(os.path.join(MODEL_DIR, "labels.npy"), np.array(actions))
    print("Model saved to:", final_model_path)
    return model, history

def main():
    print("Reading metadata...")
    metadata_df = read_metadata(METADATA_PATH)
  
    if TARGET_ACTIONS:
        metadata_df = metadata_df[metadata_df['indian sign language gloss'].apply(lambda x: normalize_gloss(x) in TARGET_ACTIONS)]
    
    mapping, actions = build_dataset(metadata_df)
    if len(mapping) == 0:
        print("No videos found in dataset directory. Make sure videos are in dataset/ or in subfolders.")
        return
   
    X, y = create_sequences_and_labels(mapping, SEQUENCE_LENGTH)
    if X.size == 0:
        print("No valid sequences created. Exiting.")
        return
   
    model, history = build_and_train(X, y, actions)

if __name__ == "__main__":
    main()
