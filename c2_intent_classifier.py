import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from ttkthemes import ThemedTk
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import traceback

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    protocol_encoder = joblib.load('protocol_encoder.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")
    traceback.print_exc()
    exit(1)

app = ThemedTk(theme="arc")
app.title("🌐 C2 Intent Classifier")
app.geometry("1000x700")

df = None
output_df = None

header = tk.Label(app, text="C2 INTENT CLASSIFIER DASHBOARD", font=("Segoe UI", 18, "bold"),
                  bg="#1f6aa5", fg="white", pady=10)
header.pack(fill="x")

main_frame = ttk.Frame(app, padding=10)
main_frame.pack(fill="both", expand=True)

top_frame = ttk.Frame(main_frame)
top_frame.pack(fill="x", pady=5)

middle_frame = ttk.Frame(main_frame)
middle_frame.pack(fill="both", expand=True, pady=10)

bottom_frame = ttk.Frame(main_frame)
bottom_frame.pack(fill="both", expand=True, pady=10)

report_label = ttk.Label(bottom_frame, text="Classification Report", font=("Segoe UI", 12, "bold"))
report_label.pack(anchor="w", pady=5)

def has_intent_column(data):
    return 'intent' in data.columns

def select_file():
    global df
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        try:
            df = pd.read_csv(filepath)
            
            print(f"Loaded CSV with columns: {df.columns.tolist()}")
            
            preview_data(df)
            predict_button.config(state="normal")
            report_area.delete(1.0, tk.END)
            
            if has_intent_column(df):
                report_area.insert(tk.END, "Intent column found. Classification report will be generated after prediction.")
            else:
                report_area.insert(tk.END, "No intent column found. Only predictions will be made without evaluation.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            traceback.print_exc()

def preview_data(data):
    preview.delete(*preview.get_children())
    
    columns_preview = list(data.columns)
    preview["columns"] = columns_preview
    
    for col in columns_preview:
        preview.heading(col, text=col)
        preview.column(col, width=100, anchor="center")
    
    for i, (_, row) in enumerate(data.head(20).iterrows()):
        preview.insert("", "end", values=list(row))

def preprocess_protocol(df):
    processed_df = df.copy()
    
    if 'protocol' in processed_df.columns and processed_df['protocol'].dtype == object:
        try:
            unseen_protocols = processed_df['protocol'][~processed_df['protocol'].isin(protocol_encoder.classes_)]
            if not unseen_protocols.empty:
                fallback_protocol = protocol_encoder.classes_[0]
                processed_df['protocol'] = processed_df['protocol'].apply(
                    lambda x: fallback_protocol if x not in protocol_encoder.classes_ else x)
        except Exception as e:
            print(f"Error handling protocols: {e}")
            traceback.print_exc()
    
    return processed_df

def predict_intent():
    global df, output_df

    if df is None:
        messagebox.showinfo("Info", "Please upload a CSV file first.")
        return

    try:
        processed_df = df.copy()

        has_intent = 'intent' in processed_df.columns
        if has_intent:
            original_intent = processed_df['intent'].copy()
            print("Intent column found and preserved")

        processed_df = preprocess_protocol(processed_df)

        exclude_cols = ["src_ip", "dst_ip", "predicted_intent"]
        if has_intent:
            exclude_cols.append('intent')

        feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
        input_data = processed_df[feature_cols].copy()
        print(f"Input columns for prediction: {input_data.columns.tolist()}")

        if 'protocol' in input_data.columns and input_data['protocol'].dtype == object:
            try:
                input_data['protocol'] = protocol_encoder.transform(input_data['protocol'])
                print("Protocol encoding completed")
            except Exception as e:
                print(f"Error encoding protocol: {e}")
                traceback.print_exc()

        try:
            scaled = scaler.transform(input_data)
            print("Data scaling completed")
        except Exception as e:
            print(f"Error scaling data: {e}")
            print(f"Input data shape: {input_data.shape}")
            traceback.print_exc()
            raise

        predictions = model.predict(scaled)
        processed_df['predicted_intent'] = predictions  

        if has_intent:
            processed_df['intent'] = original_intent

        output_df = processed_df
        print(f"Final dataframe columns: {output_df.columns.tolist()}")

        preview_data(processed_df)
        save_button.config(state="normal")

        if has_intent:
            print("Generating classification report...")
            y_true = processed_df['intent']
            y_pred = processed_df['predicted_intent']

            report = classification_report(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            report_area.delete(1.0, tk.END)
            report_area.insert(tk.END, f"Classification Report:\n\n{report}\n\n")
            report_area.insert(tk.END, f"Confusion Matrix:\n\n{cm}\n")

            print("\nClassification Report:")
            print(report)
            print("\nConfusion Matrix:")
            print(cm)

            messagebox.showinfo("Success", "Prediction completed! See the report below.")
        else:
            report_area.delete(1.0, tk.END)
            report_area.insert(tk.END, "Predictions completed successfully.\n\n")
            report_area.insert(tk.END, "Note: No 'intent' column found in the data, so no evaluation metrics could be generated.\n")
            report_area.insert(tk.END, "To see a classification report, your input data must contain an 'intent' column with ground truth values.")
            messagebox.showinfo("Done", "Prediction completed! No evaluation metrics available.")

    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        traceback.print_exc()

def save_output():
    if output_df is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path:
            output_df.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Output saved to:\n{save_path}")
    else:
        messagebox.showinfo("Info", "No prediction results to save.")

upload_button = ttk.Button(top_frame, text="Upload CSV", command=select_file)
upload_button.pack(side="left", padx=5)

predict_button = ttk.Button(top_frame, text="Predict Intent", command=predict_intent, state="disabled")
predict_button.pack(side="left", padx=5)

save_button = ttk.Button(top_frame, text="Save Results", command=save_output, state="disabled")
save_button.pack(side="left", padx=5)

def show_help():
    help_text = """
    HOW TO USE THIS APPLICATION:
    
    1. Click "Upload CSV" to load your data file
    2. To generate a classification report, your CSV must contain:
       - An "intent" column with ground truth labels
       - All the feature columns used during model training
    3. Click "Predict Intent" to classify the data
    4. The results will be displayed below
    5. Use "Save Results" to export the predictions
    
    If you only see predicted_intent without a classification report,
    your CSV may be missing the "intent" column with ground truth values.
    """
    messagebox.showinfo("Help", help_text)

help_button = ttk.Button(top_frame, text="Help", command=show_help)
help_button.pack(side="left", padx=5)

preview_label = ttk.Label(middle_frame, text="Data Preview", font=("Segoe UI", 12, "bold"))
preview_label.pack(anchor="w", pady=5)

preview_frame = ttk.Frame(middle_frame)
preview_frame.pack(fill="both", expand=True)

preview = ttk.Treeview(preview_frame, show="headings", height=10)
preview.pack(side="left", fill="both", expand=True)

v_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview.yview)
preview.configure(yscrollcommand=v_scroll.set)
v_scroll.pack(side="right", fill="y")

h_scroll = ttk.Scrollbar(middle_frame, orient="horizontal", command=preview.xview)
preview.configure(xscrollcommand=h_scroll.set)
h_scroll.pack(fill="x")

report_area = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=12, 
                                       font=("Courier New", 10))
report_area.pack(fill="both", expand=True)
report_area.insert(tk.END, "Classification report will appear here after prediction...")

app.mainloop()
