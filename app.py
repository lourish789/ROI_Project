import gradio as gr
import pickle
import pandas as pd

# Load model
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)

# Define categorical mappings (must match training!)
label_encoders = {
    "School_Type": {"Public": 0, "Private": 1},
    "School_Location": {"Urban": 0, "Rural": 1},
    "Extra_Tutorials": {"No": 0, "Yes": 1},
    "Access_To_Learning_Materials": {"No": 0, "Yes": 1},
    "Parent_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "IT_Knowledge": {"Low": 0, "Medium": 1, "High": 2},
    "Gender": {"Male": 0, "Female": 1},
    "Socioeconomic_Status": {"Low": 0, "Medium": 1, "High": 2},
    "Parent_Education_Level": {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
}

# Prediction function
def predict_jamb(study_hours, attendance, teacher_quality, distance,
                 school_type, school_location, extra_tutorials, learning_materials,
                 parent_involvement, it_knowledge, age, gender, socioeconomic_status,
                 parent_education, assignments_completed):

    # Encode categorical inputs
    input_dict = {
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Rate": attendance,
        "Teacher_Quality": teacher_quality,
        "Distance_To_School": distance,
        "School_Type": label_encoders["School_Type"][school_type],
        "School_Location": label_encoders["School_Location"][school_location],
        "Extra_Tutorials": label_encoders["Extra_Tutorials"][extra_tutorials],
        "Access_To_Learning_Materials": label_encoders["Access_To_Learning_Materials"][learning_materials],
        "Parent_Involvement": label_encoders["Parent_Involvement"][parent_involvement],
        "IT_Knowledge": label_encoders["IT_Knowledge"][it_knowledge],
        "Age": age,
        "Gender": label_encoders["Gender"][gender],
        "Socioeconomic_Status": label_encoders["Socioeconomic_Status"][socioeconomic_status],
        "Parent_Education_Level": label_encoders["Parent_Education_Level"][parent_education],
        "Assignments_Completed": assignments_completed
    }

    # Convert to dataframe
    input_data = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        return "📉 Below Average"
    #elif prediction == 1:
      #  return "⚖️ Average"
    else:
        return "✅ Pass"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 JAMB Score Prediction App")
    gr.Markdown("Predict student's JAMB performance (Below Average, Pass).")
    gr.Markdown("Fill in the details below to predict!")

    with gr.Row():
        study_hours = gr.Slider(0, 40, value=10, label="Study Hours per Week")
        attendance = gr.Slider(0, 100, value=80, label="Attendance Rate (%)")
        teacher_quality = gr.Dropdown([1,2,3,4,5], label="Teacher Quality")
        distance = gr.Number(value=5, label="Distance to School (km)")

    with gr.Row():
        school_type = gr.Radio(["Public", "Private"], label="School Type")
        school_location = gr.Radio(["Urban", "Rural"], label="School Location")
        extra_tutorials = gr.Radio(["Yes", "No"], label="Extra Tutorials")
        learning_materials = gr.Radio(["Yes", "No"], label="Access to Learning Materials")

    with gr.Row():
        parent_involvement = gr.Dropdown(["Low", "Medium", "High"], label="Parent Involvement")
        it_knowledge = gr.Dropdown(["Low", "Medium", "High"], label="IT Knowledge")
        age = gr.Slider(10, 30, value=18, label="Age")
        gender = gr.Radio(["Male", "Female"], label="Gender")

    with gr.Row():
        socioeconomic_status = gr.Dropdown(["Low", "Medium", "High"], label="Socioeconomic Status")
        parent_education = gr.Dropdown(["None", "Primary", "Secondary", "Tertiary"], label="Parent Education")
        assignments_completed = gr.Slider(0, 10, value=2, label="Assignments Completed")

    output = gr.Textbox(label="Prediction")

    submit_btn = gr.Button("Predict")
    submit_btn.click(
        predict_jamb,
        inputs=[study_hours, attendance, teacher_quality, distance,
                school_type, school_location, extra_tutorials, learning_materials,
                parent_involvement, it_knowledge, age, gender, socioeconomic_status,
                parent_education, assignments_completed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
