from codecs import BOM32_BE
from ctypes import alignment
from unittest import result
from xml.dom.expatbuilder import parseString
import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import requests
import json
import threading
import time

from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox,ttk
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import filedialog

from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

class LCD_CNN:
    def __init__(self,root):
        self.root = root

        # Set these before using them
        self.size = 10
        self.NoSlices = 5

        # Center the window
        window_width = 1006
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        img4=Image.open("gui_images/Lung-Cancer-Detection.jpg")
        img4=img4.resize((1006,500),Image.Resampling.LANCZOS)
        #Antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution.
        self.photoimg4=ImageTk.PhotoImage(img4)

        bg_img=Label(self.root,image=self.photoimg4)
        bg_img.place(x=0,y=50,width=1006,height=500)

        # title Label
        title_lbl=Label(text="Lung Cancer Detection",font=("Bradley Hand ITC",30,"bold"),bg="black",fg="white",)
        title_lbl.place(x=0,y=0,width=1006,height=50)

        #button 1
        self.b1=Button(text="Import Data",cursor="hand2",command=self.import_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b1.place(x=80,y=130,width=180,height=30)

        #button 2
        self.b2=Button(text="Pre-Process Data",cursor="hand2",command=self.preprocess_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b2.place(x=80,y=180,width=180,height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")

        #button 3
        self.b3=Button(text="Train Data",cursor="hand2",command=self.train_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b3.place(x=80,y=230,width=180,height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

        #button 4
        self.b4 = Button(text="Test Image", cursor="hand2", command=self.test_image, font=("Times New Roman",15,"bold"), bg="white", fg="black")
        self.b4.place(x=80, y=280, width=180, height=30)
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

        self.initial_weights = None  # Store initial weights for reuse
        input_shape = (self.NoSlices, self.size, self.size, 1)
        self.model = models.Sequential([
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])

#Data Import lets you upload data from external sources and combine it with data you collect via Analytics.
    def import_data(self):
        ##Data directory
        self.dataDirectory = 'sample_images/'
        self.lungPatients = os.listdir(self.dataDirectory)

        ##Read labels csv 
        self.labels = pd.read_csv('stage1_labels.csv', index_col=0)

        ##Setting x*y size to 10
        self.size = 10

        ## Setting z-dimension (number of slices to 5)
        self.NoSlices = 5

        messagebox.showinfo("Import Data" , "Data Imported Successfully!") 

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow") 
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")   

# Data preprocessing is the process of transforming raw data into an understandable format.
    def preprocess_data(self):

        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if (count < self.NoSlices):
                    yield l[i:i + n]
                    count = count + 1

        def mean(l):
            return sum(l) / len(l)
        #Average

        def dataProcessing(patient_folder, size=10, noslices=5, visualize=False):
            # Determine label from folder name
            if patient_folder.lower().startswith("cancer"):
                label = np.array([0, 1])  # Cancer
            elif patient_folder.lower().startswith("no_cancer"):
                label = np.array([1, 0])  # No Cancer
            else:
                raise ValueError(f"Unknown label for folder: {patient_folder}")

            path = os.path.join(self.dataDirectory, patient_folder)
            slices = [dicom.dcmread(os.path.join(path, s)) for s in os.listdir(path) if s.endswith('.dcm')]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

            chunk_sizes = max(1, math.floor(len(slices) / noslices))
            def chunks(l, n):
                count = 0
                for i in range(0, len(l), n):
                    if (count < noslices):
                        yield l[i:i + n]
                        count = count + 1

            def mean(l):
                return sum(l) / len(l)

            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list(map(mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            return np.array(new_slices), label

        imageData = []
        #Check if Data Labels is available in CSV or not
        for num, patient in enumerate(self.lungPatients):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                img_data, label = dataProcessing(patient, size=self.size, noslices=self.NoSlices)
                imageData.append([img_data, label, patient])
            except KeyError as e:
                print('Data is unlabeled')

        ##Results= Image Data and lable.
        np.save('processedData.npy', np.array(imageData, dtype=object))

        messagebox.showinfo("Pre-Process Data" , "Data Pre-Processing Done Successfully!") 

        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow") 
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

# Data training is the process of training the model based on the dataset and then predict on new data.
    def send_update_to_server(self, weights, num_samples):
        # Convert weights to lists for JSON serialization
        weights_list = [w.tolist() for w in weights]
        data = {
            "weights": weights_list,
            "num_samples": num_samples
        }
        try:
            response = requests.post("http://localhost:5000/upload", json=data)
            resp_json = response.json()
            print(resp_json)
            # Show popup on success
            if resp_json.get("status") in ("success", "queued"):
                messagebox.showinfo("Federated Update", "Model update sent to server!")
            else:
                messagebox.showwarning("Federated Update", "Server did not acknowledge the update.")
        except Exception as e:
            messagebox.showerror("Federated Update", f"Failed to send update to server:\n{e}")

    def wait_for_global_weights(self, poll_interval=10, max_errors=3):
        """
        Poll the server for new global weights, waiting until available.
        If the server is unreachable for max_errors times, ask the user if they want to keep waiting.
        Returns the weights when available, or None if the user chooses not to wait.
        """
        consecutive_errors = 0
        while True:
            try:
                response = requests.get("http://localhost:5000/download")
                if response.status_code == 200:
                    data = response.json()
                    if data["weights"] is not None:
                        return [np.array(w) for w in data["weights"]]
                    else:
                        print("Waiting for server to broadcast global weights...")
                    consecutive_errors = 0  # Reset error count on success
                else:
                    print("Server responded with error, retrying...")
                    consecutive_errors += 1
            except Exception as e:
                print(f"Error while polling for global weights: {e}")
                consecutive_errors += 1

            if consecutive_errors >= max_errors:
                # Ask the user if they want to keep waiting
                keep_waiting = messagebox.askyesno(
                    "Server Down",
                    "The server is down, do you want to keep waiting for the server or just train and use the local model?\n\nYes: Keep waiting\nNo: Use local model"
                )
                if not keep_waiting:
                    return None
                consecutive_errors = 0  # Reset error count if user wants to keep waiting

            time.sleep(poll_interval)

    def train_data(self):    
        imageData = np.load('processedData.npy', allow_pickle=True)
        trainingData = imageData[0:45]
        validationData = imageData[45:50]

        X_train = np.array([i[0] for i in trainingData])
        y_train = np.array([i[1] for i in trainingData])
        X_val = np.array([i[0] for i in validationData])
        y_val = np.array([i[1] for i in validationData])

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]

        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # --- Keep initial weights for every training ---
        if self.initial_weights is None:
            self.initial_weights = self.model.get_weights()
        else:
            self.model.set_weights(self.initial_weights)

        # --- Save the best model during training ---
        checkpoint_path = "best_model.weights.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max', verbose=0
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=4,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[checkpoint]
        )

        # Load the best weights before evaluation and sending to server
        self.model.load_weights(checkpoint_path)

        val_loss, val_acc = self.model.evaluate(X_val, y_val)
        print(f'Validation accuracy (best): {val_acc}')

        predictions = self.model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(y_val, axis=1)

        patients = [i[2] for i in validationData]
        predicted = ["Cancer" if c == 1 else "No Cancer" for c in predicted_classes]
        actual = ["Cancer" if c == 1 else "No Cancer" for c in actual_classes]

        for i in range(len(patients)):
            print("----------------------------------------------------")
            print("Patient: ", patients[i])
            print("Actual: ", actual[i])
            print("Predicted: ", predicted[i])
            print("----------------------------------------------------")

        final_accuracy=Label(text="Final Accuracy: " + str(val_acc),font=("Times New Roman",13,"bold"),bg="black", fg="white",)
        final_accuracy.place(x=750,y=230,width=200,height=18)  

        # Confusion Matrix
        y_actual = pd.Series(actual_classes, name='Actual')
        y_predicted = pd.Series(predicted_classes, name='Predicted')
        df_confusion = pd.crosstab(y_actual, y_predicted).reindex(columns=[0,1],index=[0,1], fill_value=0)
        print('Confusion Matrix:\n')
        print(df_confusion)

        prediction_label=Label(text=">>>>    P R E D I C T I O N    <<<<",font=("Times New Roman",14,"bold"),bg="#778899", fg="black",)
        prediction_label.place(x=0,y=458,width=1006,height=20)   

        result1 = []
        for i in range(len(patients)):
            result1.append(patients[i])
            result1.append(actual[i])
            result1.append(predicted[i])

        total_rows = int(len(patients))
        total_columns = 3

        heading = ["Patient: ", "Actual: ", "Predicted: "]

        self.root.geometry("1006x"+str(500+(len(patients)*20)-20)+"+0+0") 
        self.root.resizable(False, False)

        for i in range(total_rows):
            for j in range(total_columns):
                self.e = Entry(self.root, width=42, fg='black', font=('Times New Roman',12,'bold')) 
                self.e.grid(row=i, column=j) 
                self.e.place(x=(j*335),y=(478+i*20))
                self.e.insert(END, heading[j] + result1[j + i*3]) 
                self.e["state"] = "disabled"
                self.e.config(cursor="arrow")                     

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow") 

        # Enable the test button after training
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

        messagebox.showinfo("Train Data" , "Model Trained Successfully!")

        ## Federated learning: send best weights and sample count to server
        weights = self.model.get_weights()
        num_samples = X_train.shape[0]
        self.send_update_to_server(weights, num_samples)

        # --- Wait for global weights from server and set them ---
        self.root.after(0, lambda: messagebox.showinfo("Global Model", "Waiting for server to broadcast global weights..."))
        global_weights = self.wait_for_global_weights()
        if global_weights is not None:
            self.model.set_weights(global_weights)
            val_loss, val_acc = self.model.evaluate(X_val, y_val)
            print(f'Validation accuracy from global weights: {val_acc}')
        else:
            messagebox.showwarning("Global Model", "Using local model weights as server is unavailable.")
            # Continue using local model

        ## Function to plot confusion matrix
        def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
            plt.matshow(df_confusion, cmap=cmap)  # imshow  
            plt.colorbar()
            tick_marks = np.arange(len(df_confusion.columns))
            plt.title(title)
            plt.xticks(tick_marks, df_confusion.columns, rotation=45)
            plt.yticks(tick_marks, df_confusion.index)
            plt.ylabel(df_confusion.index.name)
            plt.xlabel(df_confusion.columns.name)
            plt.show()
        plot_confusion_matrix(df_confusion)

    def test_image(self):
        def run_prediction():
            # Let user select a DICOM file
            file_path = filedialog.askopenfilename(title="Select DICOM file", filetypes=[("DICOM files", "*.dcm")])
            if not file_path:
                return

            # Preprocess the image (resize, normalize, etc.)
            dcm = dicom.dcmread(file_path)
            img = cv2.resize(np.array(dcm.pixel_array), (self.size, self.size))
            img = img.astype(np.float32)
            img = img / np.max(img)  # normalize if needed

            # If your model expects a stack of slices, you may need to duplicate or pad
            slices = np.stack([img]*self.NoSlices, axis=0)
            slices = slices[np.newaxis, ..., np.newaxis]  # shape: (1, NoSlices, size, size, 1)

            # Predict
            prediction = self.model.predict(slices)
            pred_class = np.argmax(prediction, axis=1)[0]
            result = "Cancer" if pred_class == 1 else "No Cancer"

            self.root.after(0, lambda: messagebox.showinfo("Prediction Result", f"The model predicts: {result}"))

        # Run prediction in a separate thread
        threading.Thread(target=run_prediction).start()

# For GUI
if __name__ == "__main__":
    root=Tk()
    obj=LCD_CNN(root)
    root.mainloop()
