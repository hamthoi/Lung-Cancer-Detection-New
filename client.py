import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import requests
import threading
import time

from tkinter import *
from tkinter import messagebox, filedialog

from tensorflow.keras import layers, models  # type: ignore

class LCD_CNN:
    def __init__(self,root):
        self.root = root

        # Set these before using them
        self.size = 10
        self.NoSlices = 5
        
        # Server configuration
        self.server_ip = "localhost"
        self.server_port = 5000

        # Center the window
        window_width = 1200
        window_height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection - Federated Learning Client")

        # Configure colors
        self.bg_color = "#2C3E50"  # Dark blue-gray
        self.accent_color = "#3498DB"  # Blue
        self.success_color = "#27AE60"  # Green
        self.warning_color = "#F39C12"  # Orange
        self.danger_color = "#E74C3C"  # Red
        self.light_bg = "#ECF0F1"  # Light gray
        self.text_color = "#2C3E50"  # Dark text

        # Set background
        self.root.configure(bg=self.bg_color)

        # Main title with gradient effect
        title_frame = Frame(self.root, bg=self.bg_color, height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        
        title_lbl = Label(title_frame, text="Lung Cancer Detection", 
                         font=("Arial", 28, "bold"), 
                         bg=self.bg_color, fg="white")
        title_lbl.pack(pady=20)
        
        subtitle_lbl = Label(title_frame, text="Federated Learning Client", 
                            font=("Arial", 12), 
                            bg=self.bg_color, fg="#BDC3C7")
        subtitle_lbl.pack()

        # Main content frame
        main_frame = Frame(self.root, bg=self.light_bg)
        main_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Left panel - Server Configuration
        left_panel = Frame(main_frame, bg="white", relief="solid", bd=1)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Server configuration section
        server_label = Label(left_panel, text="🔗 Server Configuration", 
                           font=("Arial", 14, "bold"), 
                           bg="white", fg=self.text_color)
        server_label.pack(pady=(20, 15))
        
        # IP Configuration
        ip_frame = Frame(left_panel, bg="white")
        ip_frame.pack(fill='x', padx=20, pady=5)
        
        ip_label = Label(ip_frame, text="Server IP:", 
                        font=("Arial", 10, "bold"), 
                        bg="white", fg=self.text_color)
        ip_label.pack(anchor='w')
        
        self.ip_entry = Entry(ip_frame, font=("Arial", 11), 
                             width=20, relief="solid", bd=1)
        self.ip_entry.insert(0, "localhost")
        self.ip_entry.pack(fill='x', pady=(5, 10))
        
        # Port Configuration
        port_frame = Frame(left_panel, bg="white")
        port_frame.pack(fill='x', padx=20, pady=5)
        
        port_label = Label(port_frame, text="Port:", 
                          font=("Arial", 10, "bold"), 
                          bg="white", fg=self.text_color)
        port_label.pack(anchor='w')
        
        self.port_entry = Entry(port_frame, font=("Arial", 11), 
                               width=10, relief="solid", bd=1)
        self.port_entry.insert(0, "5000")
        self.port_entry.pack(fill='x', pady=(5, 15))
        
        # Connect button
        connect_btn = Button(left_panel, text="🔌 Connect to Server", 
                           command=self.update_server_config,
                           font=("Arial", 11, "bold"), 
                           bg=self.accent_color, fg="white",
                           relief="flat", bd=0,
                           width=20, height=2,
                           cursor="hand2")
        connect_btn.pack(pady=(0, 20))
        
        # Status indicator
        self.status_label = Label(left_panel, text="● Disconnected", 
                                 font=("Arial", 10), 
                                 bg="white", fg=self.danger_color)
        self.status_label.pack(pady=(0, 20))

        # Right panel - Main Operations
        right_panel = Frame(main_frame, bg="white", relief="solid", bd=1)
        right_panel.pack(side='right', fill='both', expand=True)
        
        operations_label = Label(right_panel, text="⚙️ Operations", 
                               font=("Arial", 14, "bold"), 
                               bg="white", fg=self.text_color)
        operations_label.pack(pady=(20, 15))

        # Operations frame
        operations_frame = Frame(right_panel, bg="white")
        operations_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Button 1 - Import Data
        self.b1 = Button(operations_frame, text="📁 Import Data", 
                        cursor="hand2", command=self.import_data,
                        font=("Arial", 12, "bold"), 
                        bg=self.accent_color, fg="white",
                        relief="flat", bd=0,
                        width=20, height=2,
                        activebackground="#2980B9",
                        activeforeground="white")
        self.b1.pack(pady=10)

        # Button 2 - Pre-Process Data
        self.b2 = Button(operations_frame, text="🔧 Pre-Process Data", 
                        cursor="hand2", command=self.preprocess_data,
                        font=("Arial", 12, "bold"), 
                        bg="#95A5A6", fg="white",
                        relief="flat", bd=0,
                        width=20, height=2,
                        state="disabled")
        self.b2.pack(pady=10)

        # Button 3 - Train Data
        self.b3 = Button(operations_frame, text="🎯 Train Model", 
                        cursor="hand2", command=self.train_data,
                        font=("Arial", 12, "bold"), 
                        bg="#95A5A6", fg="white",
                        relief="flat", bd=0,
                        width=20, height=2,
                        state="disabled")
        self.b3.pack(pady=10)

        # Button 4 - Diagnose Patient
        self.b4 = Button(operations_frame, text="🏥 Diagnose Patient", 
                        cursor="hand2", command=self.diagnose_patient,
                        font=("Arial", 12, "bold"), 
                        bg=self.success_color, fg="white",
                        relief="flat", bd=0,
                        width=20, height=2,
                        activebackground="#229954",
                        activeforeground="white")
        self.b4.pack(pady=10)

        # Progress frame
        progress_frame = Frame(right_panel, bg="white")
        progress_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.progress_label = Label(progress_frame, text="Ready to start federated learning", 
                                  font=("Arial", 10), 
                                  bg="white", fg="#7F8C8D")
        self.progress_label.pack()

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

        # Update progress
        self.progress_label.config(text="Data imported successfully!")
        
        # Update button states
        self.b1.config(state="disabled", bg="#95A5A6", cursor="arrow")
        self.b2.config(state="normal", bg=self.accent_color, cursor="hand2")

        messagebox.showinfo("Import Data" , "Data Imported Successfully!")

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



        # --- Data balancing: oversample cancer folders to match no_cancer count ---
        cancer_folders = [f for f in self.lungPatients if f.lower().startswith('cancer')]
        no_cancer_folders = [f for f in self.lungPatients if f.lower().startswith('no_cancer')]

        n_cancer = len(cancer_folders)
        n_no_cancer = len(no_cancer_folders)

        # If cancer folders are fewer, oversample (with replacement)
        if n_cancer < n_no_cancer:
            # Randomly sample cancer folders to match no_cancer count
            extra_cancer = np.random.choice(cancer_folders, n_no_cancer - n_cancer, replace=True)
            balanced_cancer_folders = cancer_folders + list(extra_cancer)
        else:
            balanced_cancer_folders = cancer_folders

        # Combine and shuffle all folders
        all_folders = balanced_cancer_folders + no_cancer_folders
        np.random.shuffle(all_folders)

        imageData = []
        for num, patient in enumerate(all_folders):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                img_data, label = dataProcessing(patient, size=self.size, noslices=self.NoSlices)
                imageData.append([img_data, label, patient])
            except KeyError as e:
                print('Data is unlabeled')

        ##Results= Image Data and lable.
        np.save('processedData.npy', np.array(imageData, dtype=object))

        # Update progress
        self.progress_label.config(text="Data pre-processing completed!")
        
        # Update button states
        self.b2.config(state="disabled", bg="#95A5A6", cursor="arrow")
        self.b3.config(state="normal", bg=self.accent_color, cursor="hand2")

        messagebox.showinfo("Pre-Process Data" , "Data Pre-Processing Done Successfully!")

# Data training is the process of training the model based on the dataset and then predict on new data.
    def send_update_to_server(self, weights, num_samples):
        # Convert weights to lists for JSON serialization
        weights_list = [w.tolist() for w in weights]
        data = {
            "weights": weights_list,
            "num_samples": num_samples
        }
        try:
            response = requests.post(self.get_server_url("upload"), json=data)
            resp_json = response.json()
            print(resp_json)
            # Show popup on success
            if resp_json.get("status") in ("success", "queued", "accepted"):
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
                response = requests.get(self.get_server_url("download"))
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
        # --- Download and set global weights before training ---
        global_weights = self.wait_for_global_weights()
        if global_weights is not None:
            self.model.set_weights(global_weights)
            print("Loaded global weights from server before training.")
        else:
            print("No global weights available, training from scratch.")
            # Reinitialize model weights (from scratch)
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

        imageData = np.load('processedData.npy', allow_pickle=True)
        # Separate cancer and no_cancer patients
        cancer_data = [item for item in imageData if np.array_equal(item[1], [0, 1])]
        no_cancer_data = [item for item in imageData if np.array_equal(item[1], [1, 0])]

        np.random.shuffle(cancer_data)
        np.random.shuffle(no_cancer_data)

        # Number of validation samples (test set size)
        val_size = 10
        val_cancer = val_size // 2
        val_no_cancer = val_size - val_cancer

        # If not enough cancer/no_cancer samples, adjust
        val_cancer = min(val_cancer, len(cancer_data))
        val_no_cancer = min(val_no_cancer, len(no_cancer_data))
        val_size = val_cancer + val_no_cancer

        validationData = cancer_data[:val_cancer] + no_cancer_data[:val_no_cancer]
        # Remove validation samples from training
        train_cancer = cancer_data[val_cancer:]
        train_no_cancer = no_cancer_data[val_no_cancer:]
        trainingData = train_cancer + train_no_cancer
        np.random.shuffle(trainingData)

        X_train = np.array([i[0] for i in trainingData])
        y_train = np.array([i[1] for i in trainingData])
        X_val = np.array([i[0] for i in validationData])
        y_val = np.array([i[1] for i in validationData])

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # --- Train the model (no ModelCheckpoint) ---
        history = self.model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=4,
            validation_data=(X_val, y_val),
            verbose=1
        )

        val_loss, val_acc = self.model.evaluate(X_val, y_val)
        print(f'Validation accuracy (final epoch): {val_acc}')

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

        # Confusion Matrix
        y_actual = pd.Series(actual_classes, name='Actual')
        y_predicted = pd.Series(predicted_classes, name='Predicted')
        df_confusion = pd.crosstab(y_actual, y_predicted).reindex(columns=[0,1],index=[0,1], fill_value=0)
        print('Confusion Matrix:\n')
        print(df_confusion)

        # Create a separate results window instead of resizing main window
        self.show_training_results(patients, actual, predicted, val_acc)

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow") 

        # Enable the test button after training
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

        # Update progress
        self.progress_label.config(text="Model training completed! Ready for diagnosis.")

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

    def show_training_results(self, patients, actual, predicted, accuracy):
        """Show training results in a separate window"""
        # Create results window
        results_window = Toplevel(self.root)
        results_window.title("Training Results")
        results_window.geometry("800x600")
        results_window.configure(bg=self.light_bg)
        
        # Center the window
        results_window.transient(self.root)
        results_window.grab_set()
        
        # Title
        title_label = Label(results_window, text="Training Results", 
                           font=("Arial", 18, "bold"), 
                           bg=self.light_bg, fg=self.text_color)
        title_label.pack(pady=20)
        
        # Accuracy display
        accuracy_frame = Frame(results_window, bg="white", relief="solid", bd=1)
        accuracy_frame.pack(fill='x', padx=20, pady=10)
        
        accuracy_label = Label(accuracy_frame, text=f"Final Accuracy: {accuracy:.4f}", 
                              font=("Arial", 14, "bold"), 
                              bg="white", fg=self.success_color)
        accuracy_label.pack(pady=10)
        
        # Results table
        table_frame = Frame(results_window, bg="white", relief="solid", bd=1)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Table headers
        headers = ["Patient", "Actual", "Predicted", "Status"]
        for i, header in enumerate(headers):
            header_label = Label(table_frame, text=header, 
                               font=("Arial", 12, "bold"), 
                               bg=self.accent_color, fg="white",
                               relief="solid", bd=1)
            header_label.grid(row=0, column=i, sticky='ew', padx=1, pady=1)
        
        # Table data
        for i, (patient, act, pred) in enumerate(zip(patients, actual, predicted)):
            # Patient name
            patient_label = Label(table_frame, text=patient, 
                                font=("Arial", 10), 
                                bg="white", fg=self.text_color,
                                relief="solid", bd=1)
            patient_label.grid(row=i+1, column=0, sticky='ew', padx=1, pady=1)
            
            # Actual
            actual_label = Label(table_frame, text=act, 
                               font=("Arial", 10), 
                               bg="white", fg=self.text_color,
                               relief="solid", bd=1)
            actual_label.grid(row=i+1, column=1, sticky='ew', padx=1, pady=1)
            
            # Predicted
            predicted_label = Label(table_frame, text=pred, 
                                  font=("Arial", 10), 
                                  bg="white", fg=self.text_color,
                                  relief="solid", bd=1)
            predicted_label.grid(row=i+1, column=2, sticky='ew', padx=1, pady=1)
            
            # Status (correct/incorrect)
            status = "✓ Correct" if act == pred else "✗ Incorrect"
            status_color = self.success_color if act == pred else self.danger_color
            status_label = Label(table_frame, text=status, 
                               font=("Arial", 10, "bold"), 
                               bg="white", fg=status_color,
                               relief="solid", bd=1)
            status_label.grid(row=i+1, column=3, sticky='ew', padx=1, pady=1)
        
        # Configure grid weights
        for i in range(4):
            table_frame.grid_columnconfigure(i, weight=1)
        
        # Close button
        close_button = Button(results_window, text="Close", 
                            command=results_window.destroy,
                            font=("Arial", 12, "bold"), 
                            bg=self.accent_color, fg="white",
                            relief="flat", bd=0,
                            width=15, height=2)
        close_button.pack(pady=20)

    def diagnose_patient(self):
        def run_prediction():
            # Let user select a patient folder
            folder_path = filedialog.askdirectory(title="Select Patient Folder")
            if not folder_path:
                return

            # Load and preprocess all DICOM slices in the folder
            slices = [dicom.dcmread(os.path.join(folder_path, s)) for s in os.listdir(folder_path) if s.endswith('.dcm')]
            if not slices:
                self.root.after(0, lambda: messagebox.showerror("Error", "No DICOM files found in the selected folder."))
                return
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            slices = [cv2.resize(np.array(each_slice.pixel_array), (self.size, self.size)) for each_slice in slices]

            # Chunk and average as in dataProcessing
            chunk_sizes = max(1, math.floor(len(slices) / self.NoSlices))
            def chunks(l, n):
                count = 0
                for i in range(0, len(l), n):
                    if (count < self.NoSlices):
                        yield l[i:i + n]
                        count = count + 1
            def mean(l):
                return sum(l) / len(l)
            new_slices = []
            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list(map(mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)
            patient_stack = np.array(new_slices)
            patient_stack = patient_stack[np.newaxis, ..., np.newaxis]  # shape: (1, NoSlices, size, size, 1)

            # Try to get global weights from server
            used_global = False
            try:
                response = requests.get(self.get_server_url("download"), timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data["weights"] is not None:
                        global_weights = [np.array(w) for w in data["weights"]]
                        self.model.set_weights(global_weights)
                        used_global = True
            except Exception as e:
                print(f"Could not fetch global weights: {e}")

            # Predict
            prediction = self.model.predict(patient_stack)
            pred_class = np.argmax(prediction, axis=1)[0]
            result = "Cancer" if pred_class == 1 else "No Cancer"

            msg = f"The model predicts: {result}\n"
            if used_global:
                msg += "(Used global model from server)"
            else:
                msg += "(Used latest local model weights)"

            self.root.after(0, lambda: messagebox.showinfo("Diagnosis Result", msg))

        # Run prediction in a separate thread
        threading.Thread(target=run_prediction).start()

    def update_server_config(self):
        """Update server configuration from GUI inputs"""
        try:
            self.server_ip = self.ip_entry.get().strip()
            self.server_port = int(self.port_entry.get().strip())
            
            # Update status to connecting
            self.status_label.config(text="● Connecting...", fg=self.warning_color)
            self.progress_label.config(text="Testing connection to server...")
            self.root.update()
            
            # Test connection
            test_url = f"http://{self.server_ip}:{self.server_port}/download"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                self.status_label.config(text="● Connected", fg=self.success_color)
                self.progress_label.config(text=f"Successfully connected to server at {self.server_ip}:{self.server_port}")
                messagebox.showinfo("Connection", f"Successfully connected to server at {self.server_ip}:{self.server_port}")
            else:
                self.status_label.config(text="● Error", fg=self.danger_color)
                self.progress_label.config(text=f"Server responded with status {response.status_code}")
                messagebox.showwarning("Connection", f"Server responded with status {response.status_code}")
        except ValueError:
            self.status_label.config(text="● Error", fg=self.danger_color)
            self.progress_label.config(text="Invalid port number")
            messagebox.showerror("Error", "Invalid port number. Please enter a valid integer.")
        except requests.exceptions.ConnectionError:
            self.status_label.config(text="● Disconnected", fg=self.danger_color)
            self.progress_label.config(text="Cannot connect to server")
            messagebox.showerror("Connection Error", f"Cannot connect to server at {self.server_ip}:{self.server_port}\n\nPlease check:\n- Server is running\n- IP address is correct\n- Port is correct\n- Network connectivity")
        except Exception as e:
            self.status_label.config(text="● Error", fg=self.danger_color)
            self.progress_label.config(text=f"Connection error: {str(e)}")
            messagebox.showerror("Error", f"Connection error: {str(e)}")

    def get_server_url(self, endpoint):
        """Get the full server URL for a given endpoint"""
        return f"http://{self.server_ip}:{self.server_port}/{endpoint}"

# For GUI
if __name__ == "__main__":
    root=Tk()
    obj=LCD_CNN(root)
    root.mainloop()
