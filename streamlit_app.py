import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "augmented_balanced_dataset_m.csv"

# column name -> (friendly label, help text), grouped for the form
GROUPS = {
    "📅 Attendance": {
        "Days.Logged.In": ("Days logged in", "Number of different days the student logged in."),
    },
    "📖 Learning activity": {
        "Read.Actions": ("Reading actions", "Times the student viewed course materials."),
        "Quiz.Actions": ("Quiz actions", "Quiz attempts and quiz-related clicks."),
        "Forum.Actions": ("Forum actions", "Posts, replies and views in discussion forums."),
    },
    "📝 Coursework": {
        "Submission.Actions": ("Submissions", "Assignments submitted."),
        "Create.Actions": ("Items created", "New content the student created."),
        "Update.Actions": ("Items edited", "Edits to existing content."),
        "Delete.Actions": ("Items deleted", "Content the student deleted."),
    },
}
FEATURES = [f for g in GROUPS.values() for f in g]
LABEL = {f: lbl for g in GROUPS.values() for f, (lbl, _) in g.items()}

st.set_page_config(page_title="High or Low?", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Training model (first load only)...")
def load_model():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = joblib.load("StudentPredictorV1.joblib")

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": pd.DataFrame(
            classification_report(y_test, y_pred, labels=["High", "Low"], output_dict=True)
        ).T,
        "cm": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=["High", "Low"]),
            index=["Actual High", "Actual Low"],
            columns=["Predicted High", "Predicted Low"],
        ),
    }
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    # "typical" student for each class and overall (medians), used for presets and comparison
    profiles = pd.DataFrame(
        {
            "High": df[df["rating"] == "High"][FEATURES].median(),
            "Low": df[df["rating"] == "Low"][FEATURES].median(),
            "All": df[FEATURES].median(),
        }
    )
    
    for grp in ["High", "Low"]:
        q = df[df["rating"] == grp][FEATURES].quantile([0.25, 0.75]).round().astype(int)
        profiles[f"{grp} range"] = [f"{q.loc[0.25, f]}–{q.loc[0.75, f]}" for f in FEATURES]
    
    return model, metrics, importances, profiles


model, metrics, importances, profiles = load_model()
classes = list(model.classes_)

# ---------- preset buttons (callbacks run before the widgets are drawn) ----------
for f in FEATURES:
    st.session_state.setdefault(f, int(profiles.loc[f, "All"]))


def apply_preset(name):
    for f in FEATURES:
        st.session_state[f] = int(profiles.loc[f, name])


st.title("🎓 High or Low?: Early Student Engagement Monitor")
st.caption(
    "Enter a student's recent LMS activity to see whether their engagement looks High or Low. "
    "Use it to decide who to check in with, not as a final judgment."
)

tab_single, tab_batch, tab_model = st.tabs(["Single student", "Batch (CSV)", "Model info"])

# ---------- Single prediction ----------
with tab_single:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Student activity")
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.button("Typical High", on_click=apply_preset, args=("High",), width="stretch")
        b2.button("Typical Low", on_click=apply_preset, args=("Low",), width="stretch")
        b3.button("Reset", on_click=apply_preset, args=("All",), width="stretch")
        st.caption("Presets fill the form with the median student from each group in the training data.")

        for group, feats in GROUPS.items():
            with st.container(border=True):
                st.markdown(f"**{group}**")
                cols = st.columns(len(feats))
                for col, (feat, (label, tip)) in zip(cols, feats.items()):
                    col.number_input(label, min_value=0, step=1, help=tip, key=feat)

    with right:
        st.subheader("Prediction")
        row = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])[FEATURES]
        proba = dict(zip(classes, model.predict_proba(row)[0]))
        p_low = proba["Low"]

        with st.container(border=True):
            if p_low >= 0.5:
                st.error("### ⚠️ Low engagement\nThis student may need a check-in.")
            else:
                st.success("### ✅ High engagement\nThis student looks on track.")
            st.progress(float(p_low), text=f"Chance of Low engagement: {p_low:.0%}")
            if 0.35 <= p_low <= 0.65:
                st.caption("⚠️ Low confidence. Check this student's activity and recent trend before deciding on an action.")

        st.markdown("**How this student compares**")
        compare = pd.DataFrame(
                {
                    "This student": row.iloc[0].astype(int).astype(str),
                    "Typical High (middle 50%)": profiles["High range"],
                    "Typical Low (middle 50%)": profiles["Low range"],
                }
        ).loc[FEATURES]
        compare.index = [LABEL[f] for f in compare.index]
        st.dataframe(compare, width="stretch")

# ---------- Batch prediction ----------
with tab_batch:
    st.subheader("Upload weekly activity logs")
    st.write(
        "Upload a CSV with one row per student. Required columns: "
        + ", ".join(f"`{f}`" for f in FEATURES)
        + ". Extra columns (like a student ID) are kept."
    )
    up = st.file_uploader("CSV file", type="csv")
    if up:
        batch = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch["Prediction"] = model.predict(batch[FEATURES])
            batch["P(Low)"] = model.predict_proba(batch[FEATURES])[:, classes.index("Low")]
            batch = batch.sort_values("P(Low)", ascending=False)
            n_low = int((batch["Prediction"] == "Low").sum())
            m1, m2 = st.columns(2)
            m1.metric("Students", len(batch))
            m2.metric("Flagged as Low", n_low)
            st.dataframe(
                batch,
                width="stretch",
                column_config={"P(Low)": st.column_config.ProgressColumn("P(Low)", min_value=0, max_value=1)},
            )
            st.download_button(
                "Download results", batch.to_csv(index=False).encode(), "predictions.csv", "text/csv"
            )

# ---------- Model info ----------
with tab_model:
    st.subheader("How well does the model work?")
    st.metric("Accuracy on held-out test data", f"{metrics['accuracy']:.1%}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Confusion matrix** (rows are the truth, columns are the prediction)")
        st.dataframe(metrics["cm"])
        st.write("**Classification report**")
        st.dataframe(metrics["report"].round(2))
    with c2:
        st.write("**Which activities matter most to the model**")
        imp = importances.copy()
        imp.index = [LABEL[f] for f in imp.index]
        st.bar_chart(imp, horizontal=True)
