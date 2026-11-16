import os
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. FILE AND DATASET SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "student_data.csv")

COLUMNS = [
    "Student_Name",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Final_Score"
]

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed"
]


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        for col in COLUMNS:
            if col not in df.columns:
                df[col] = "" if col == "Student_Name" else 0

        df = df[COLUMNS]

    else:
        df = pd.DataFrame(columns=COLUMNS)

    df["Student_Name"] = (
        df["Student_Name"].fillna("").astype(str).str.strip()
    )

    numeric_columns = COLUMNS[1:]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    return df


data = load_data()


# ==========================================
# 3. DASHBOARD OVERVIEW
# ==========================================

def get_dashboard():
    total_students = len(data)

    if total_students > 0:
        average_score = data["Final_Score"].mean()
    else:
        average_score = 0

    return f"""
    <div class="stats">
        <div class="stat-card">
            <h3>Total Students</h3>
            <h1>{total_students}</h1>
        </div>

        <div class="stat-card">
            <h3>Average Final Score</h3>
            <h1>{average_score:.2f}</h1>
        </div>

        <div class="stat-card">
            <h3>Project</h3>
            <h2>StudentInsight AI</h2>
        </div>
    </div>
    """


# ==========================================
# 4. PERFORMANCE GRAPHS
# ==========================================

def create_graphs():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if data.empty:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No student data available",
                ha="center", va="center"
            )
            ax.set_axis_off()

        fig.tight_layout()
        return fig

    axes[0].scatter(
        data["Study_Hours"],
        data["Final_Score"]
    )
    axes[0].set_title("Study Hours vs Final Score")
    axes[0].set_xlabel("Study Hours")
    axes[0].set_ylabel("Final Score")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        data["Attendance"],
        data["Final_Score"]
    )
    axes[1].set_title("Attendance vs Final Score")
    axes[1].set_xlabel("Attendance (%)")
    axes[1].set_ylabel("Final Score")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        data["Final_Score"],
        bins=8,
        edgecolor="black"
    )
    axes[2].set_title("Final Score Distribution")
    axes[2].set_xlabel("Final Score")
    axes[2].set_ylabel("Number of Students")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ==========================================
# 5. ADD NEW STUDENT
# ==========================================

def add_student(name, hours, attendance, previous,
                assignments, score):

    global data

    if name is None or not str(name).strip():
        return (
            "Please enter a student name.",
            data.copy(),
            get_dashboard(),
            create_graphs()
        )

    try:
        hours = float(hours)
        attendance = float(attendance)
        previous = float(previous)
        assignments = float(assignments)
        score = float(score)
    except (TypeError, ValueError):
        return (
            "Please enter all student details correctly.",
            data.copy(),
            get_dashboard(),
            create_graphs()
        )

    if hours < 0 or assignments < 0:
        return (
            "Study hours and assignments cannot be negative.",
            data.copy(),
            get_dashboard(),
            create_graphs()
        )

    if not 0 <= attendance <= 100:
        return (
            "Attendance must be between 0 and 100.",
            data.copy(),
            get_dashboard(),
            create_graphs()
        )

    if not 0 <= previous <= 100 or not 0 <= score <= 100:
        return (
            "Scores must be between 0 and 100.",
            data.copy(),
            get_dashboard(),
            create_graphs()
        )

    new_student = {
        "Student_Name": str(name).strip(),
        "Study_Hours": hours,
        "Attendance": attendance,
        "Previous_Score": previous,
        "Assignments_Completed": assignments,
        "Final_Score": score
    }

    new_row = pd.DataFrame([new_student], columns=COLUMNS)

    data = pd.concat(
        [data, new_row],
        ignore_index=True
    )

    data.to_csv(DATA_FILE, index=False)

    return (
        f"Student {str(name).strip()} added successfully!",
        data.copy(),
        get_dashboard(),
        create_graphs()
    )


# ==========================================
# 6. SEARCH STUDENT
# ==========================================

def search_student(name):

    if name is None or not str(name).strip():
        return data.copy()

    result = data[
        data["Student_Name"].fillna("").str.contains(
            str(name).strip(),
            case=False,
            na=False
        )
    ]

    return result.copy()


# ==========================================
# 7. AI PERFORMANCE PREDICTION
# ==========================================

def predict_performance(hours, attendance,
                        previous, assignments):

    try:
        hours = float(hours)
        attendance = float(attendance)
        previous = float(previous)
        assignments = float(assignments)

    except (TypeError, ValueError):
        return "Please enter all prediction details."

    if hours < 0 or assignments < 0:
        return "Study hours and assignments cannot be negative."

    if not 0 <= attendance <= 100:
        return "Attendance must be between 0 and 100."

    if not 0 <= previous <= 100:
        return "Previous score must be between 0 and 100."

    if len(data) < 2:
        return "Please add at least two student records first."

    X = data[FEATURES]
    y = data["Final_Score"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    student_input = pd.DataFrame(
        [[hours, attendance, previous, assignments]],
        columns=FEATURES
    )

    prediction = model.predict(student_input)[0]

    prediction = max(0, min(100, prediction))

    return (
        f"Predicted Final Score: {prediction:.2f} / 100\n\n"
        "This is an estimated result based on the available "
        "sample dataset, not a guaranteed score."
    )


# ==========================================
# 8. WEBSITE DESIGN
# ==========================================

CSS = """
.gradio-container {
    background: linear-gradient(135deg, #080d20, #17102e);
    color: white;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.stat-card {
    background: linear-gradient(135deg, #24164b, #172b50);
    border: 1px solid #7654c8;
    border-radius: 15px;
    padding: 22px;
    margin: 8px;
    text-align: center;
}

.stat-card h3 {
    color: #c7b6ff !important;
    font-size: 17px;
}

.stat-card h1 {
    color: #ffffff !important;
    font-size: 34px;
}

.stat-card h2 {
    color: #ffffff !important;
    font-size: 22px;
}

button {
    border-radius: 10px !important;
}
"""


# ==========================================
# 9. GRADIO WEBSITE
# ==========================================

with gr.Blocks(
    title="StudentInsight AI",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue"
    )
) as demo:

    gr.Markdown(
        """
        # StudentInsight AI

        ### Your Intelligent Student Performance Dashboard

        **Predict. Understand. Improve.**
        """
    )

    gr.Markdown("## Dashboard Overview")

    dashboard_stats = gr.HTML(
        value=get_dashboard()
    )

    gr.Markdown("## Performance Analytics")

    performance_plot = gr.Plot(
        value=create_graphs()
    )

    # STUDENT RECORDS

    with gr.Tab("Student Records"):

        gr.Markdown("### All Student Records")

        student_table = gr.Dataframe(
            value=data,
            headers=COLUMNS,
            interactive=False,
            label="All Student Records"
        )

    # ADD STUDENT

    with gr.Tab("Add Student"):

        gr.Markdown("### Add a New Student")

        student_name = gr.Textbox(
            label="Student Name",
            placeholder="Enter any student name"
        )

        study_hours = gr.Number(
            label="Study Hours"
        )

        attendance = gr.Number(
            label="Attendance (%)"
        )

        previous_score = gr.Number(
            label="Previous Score"
        )

        assignments = gr.Number(
            label="Assignments Completed"
        )

        final_score = gr.Number(
            label="Final Score"
        )

        add_button = gr.Button(
            "Add Student",
            variant="primary"
        )

        add_status = gr.Textbox(
            label="Status",
            interactive=False
        )

        add_button.click(
            fn=add_student,
            inputs=[
                student_name,
                study_hours,
                attendance,
                previous_score,
                assignments,
                final_score
            ],
            outputs=[
                add_status,
                student_table,
                dashboard_stats,
                performance_plot
            ]
        )

    # SEARCH STUDENT

    with gr.Tab("Search Student"):

        gr.Markdown("### Find a Student")

        search_name = gr.Textbox(
            label="Enter Student Name",
            placeholder="Type a name or leave blank to show all"
        )

        search_button = gr.Button(
            "Search Student",
            variant="primary"
        )

        search_results = gr.Dataframe(
            headers=COLUMNS,
            interactive=False,
            label="Search Results"
        )

        search_button.click(
            fn=search_student,
            inputs=search_name,
            outputs=search_results
        )

    # AI PREDICTION

    with gr.Tab("AI Performance Prediction"):

        gr.Markdown(
            """
            ### Predict Student Performance

            Enter student details to get an estimated final score.
            """
        )

        pred_hours = gr.Number(
            label="Study Hours"
        )

        pred_attendance = gr.Number(
            label="Attendance (%)"
        )

        pred_previous = gr.Number(
            label="Previous Score"
        )

        pred_assignments = gr.Number(
            label="Assignments Completed"
        )

        predict_button = gr.Button(
            "Predict Performance",
            variant="primary"
        )

        prediction_result = gr.Textbox(
            label="AI Prediction Result",
            lines=3,
            interactive=False
        )

        predict_button.click(
            fn=predict_performance,
            inputs=[
                pred_hours,
                pred_attendance,
                pred_previous,
                pred_assignments
            ],
            outputs=prediction_result
        )

    gr.Markdown(
        """
        ---
        **StudentInsight AI** | CAA Project 2026–27
        """
    )


# ==========================================
# 10. LAUNCH WEBSITE
# ==========================================

if __name__ == "__main__":
    demo.launch(css=CSS, share=True, debug=True)