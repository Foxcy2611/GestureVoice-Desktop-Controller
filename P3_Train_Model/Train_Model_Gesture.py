"""
====================
Tên script: Train_Model_Gesture.py
Tác dụng: Huấn luyện, đánh giá và lưu mô hình MLP phân loại 10 gesture.
====================
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers, callbacks, models
import matplotlib.pyplot as plt

# ========= LINK =========
X_PATH = "2_Data_Preprocessing/Output_npy/X_gesture.npy"
Y_PATH = "2_Data_Preprocessing/Output_npy/Y_gesture.npy"
Z_PATH = "2_Data_Preprocessing/Output_npy/Z_classes.npy"

OUT_DIR   = "3_Train_Model/Output_Gesture"
MODEL_OUT = os.path.join(OUT_DIR, "Gesture_MLP_Model.keras") 
CKPT_OUT  = os.path.join(OUT_DIR, "Gesture_MLP_Best.keras") 
LOG_OUT   = os.path.join(OUT_DIR, "Gestures_Train_Log.csv")

NUM_CLASSES = 10 
RAMDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 32

def Load_Data():
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    Z = np.load(Z_PATH, allow_pickle=True)
    return X, Y, Z

def Spilit_Data(X, Y):
    X_train_np, X_temp, Y_train_np, Y_temp = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=RAMDOM_STATE, 
        stratify=Y
    )
    
    X_valid_np, X_test_np, Y_valid_np, Y_test_np = train_test_split(
        X_temp, 
        Y_temp, 
        test_size=0.5, 
        random_state=RAMDOM_STATE, 
        stratify=Y_temp
    )

    X_train = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train_np, dtype=tf.int64)
    X_valid = tf.convert_to_tensor(X_valid_np, dtype=tf.float32)
    Y_valid = tf.convert_to_tensor(Y_valid_np, dtype=tf.int64)
    X_test = tf.convert_to_tensor(X_test_np, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_test_np, dtype=tf.int64)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def Build_MLP_Model():
    model = models.Sequential([
        layers.Input(shape=(63, )),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        layers.Dense(10),
        layers.Softmax(),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

def Build_Callbacks():
    os.makedirs(OUT_DIR, exist_ok=True)

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            mode="min",
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ModelCheckpoint(
            CKPT_OUT,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),

        callbacks.CSVLogger(LOG_OUT),

        callbacks.TerminateOnNaN(),
    ]

def Plot_Training_History(history):
    plt.figure(figsize=(12, 5))
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Độ chính xác (Accuracy)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green', linewidth=2)
    plt.title('Độ suy hao (Loss)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Training_History_1.png'), dpi=150)
    print(">> Đã lưu biểu đồ huấn luyện tại: Training_History.png")
    plt.show()

def main():
    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH MLP GESTURE")
    print("=" * 70)

    # BƯỚC 1: LOAD DATA
    print("\n[BƯỚC 1] ĐANG TẢI DỮ LIỆU...")
    X, Y, Z = Load_Data()
    print(f"  + Kích thước X (Features): {X.shape}")
    print(f"  + Kích thước Y (Labels)  : {Y.shape}")
    print(f"  + Kích thước Z (Classes) : {Z.shape} -> Danh sách: {list(Z)}")

    # BƯỚC 2: CHIA TẬP DATA
    print("\n[BƯỚC 2] ĐANG PHÂN CHIA TẬP DỮ LIỆU (Train/Val/Test)...")
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = Spilit_Data(X, Y)
    print(f"  + Tập Train (Huấn luyện): {X_train.shape[0]:>5} mẫu")
    print(f"  + Tập Valid (Tinh chỉnh): {X_valid.shape[0]:>5} mẫu")
    print(f"  + Tập Test  (Đánh giá)  : {X_test.shape[0]:>5} mẫu")
    print(f"  + Phân bổ số mẫu/lớp   : {np.bincount(Y)}")

    # BƯỚC 3: BUILD MÔ HÌNH
    print("\n[BƯỚC 3] KIẾN TRÚC MÔ HÌNH MLP:")
    model = Build_MLP_Model()
    model.summary()

    # BƯỚC 4: HUẤN LUYỆN
    print("\n[BƯỚC 4] BẮT ĐẦU TRAIN MÔ HÌNH...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=Build_Callbacks(),
        verbose=1 
    ) 

    Plot_Training_History(history)

    # BƯỚC 5: ĐÁNH GIÁ & KẾT QUẢ
    print("\n" + "=" * 70)
    print("🎯 [BƯỚC 5] KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 70)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"  > Test Loss     : {loss:.4f}")
    print(f"  > Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    
    print("\n--- BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    print(classification_report(Y_test, y_pred, target_names=[str(c) for c in Z], digits=4))
    
    print("\n--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    print("Thực tế (hàng) x Dự đoán (cột):")
    print(confusion_matrix(Y_test, y_pred))

    # BƯỚC 6: LƯU FILE
    print("\n" + "=" * 70)
    print("💾 [BƯỚC 6] HOÀN TẤT VÀ LƯU TRỮ")
    print("=" * 70)
    model.save(MODEL_OUT)
    print(f"  [+] Model cuối cùng : {MODEL_OUT}")
    print(f"  [+] Model tốt nhất  : {CKPT_OUT}")
    print(f"  [+] File Log CSV    : {LOG_OUT}\n")

if __name__ == "__main__":
    main()
