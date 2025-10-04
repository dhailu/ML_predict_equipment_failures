import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Assuming PredictPipeline and CustomData are defined in src.pipeline.predict_pipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# =========================================
# Streamlit App
# =========================================
st.title(" Equipment Failure Prediction (Batch Mode)")

# ========== File Upload Section ==========
st.subheader(" Upload Equipment Data (CSV)")
uploaded_file = st.file_uploader("Upload file here", type="csv") = "artifacts/data.csv"


if uploaded_file is not None:
    # Load file into dataframe
    input_df = pd.read_csv(uploaded_file)
    st.write(" Uploaded Data Preview:", input_df.head())

    # ========== Run Prediction ==========
    if st.button(" Predict Failures"):
        try:
            predict_pipeline = PredictPipeline()
            predictions = predict_pipeline.predict(input_df)

            # Add predictions to dataframe
            input_df["failure_within_7_days"] = predictions

            st.success(" Predictions complete!")

            # ====== Summaries ======
            failed_df = input_df[input_df["failure_within_7_days"] == 1]

            if not failed_df.empty:
                st.subheader(" Failure Summary")

                # Count by equipment type
                failure_counts = failed_df["equipment_type"].value_counts()

                st.write("### Failures by Type")
                st.dataframe(
                    failure_counts.reset_index(name="count")
                    .rename(columns={"index": "equipment_type"})
                )

                # Example: "2 HVAC, 1 Boiler will fail within 7 days"
                fail_summary = ", ".join(
                    [f"{count} {etype}" for etype, count in failure_counts.items()]
                )
                st.info(f" {fail_summary} predicted to fail within 7 days.")

                # ====== Visualizations ======
                fig1, ax1 = plt.subplots()
                ax1.pie(failure_counts, labels=failure_counts.index, autopct="%1.1f%%")
                ax1.set_title("Failures by Equipment Type")
                st.pyplot(fig1)

                st.bar_chart(failure_counts)

                # Failed equipment details
                st.subheader(" Failed Equipment Details")
                st.dataframe(failed_df[["equipment_id", "equipment_type", "location"]])

                # ====== Notifications ======
                if st.button(" Send Email Alerts"):
                    st.success(" Email alerts sent to maintenance team!")

            else:
                st.info(" No failures predicted within 7 days.")

        except Exception as e:
            st.error(f" Error during prediction: {str(e)}")

else:
    st.warning("Please upload a CSV file to continue.")

# ========== Notification Button ==========
st.subheader("Send Notifications")
if st.button("Send Email Alerts"):
    # (placeholder for email logic)    
    st.success(" Email alerts sent to site leader for site(site_code) and eq_id!") ##0000FF

# ####################################################SECOND OPTION####################################################
# ## Load your data
# test_df = pd.read_csv("artifacts/data.csv")  # replace with your dataset

# st.title(" Equipment Failure Prediction Dashboard")

# # Filter failed equipment
# failed_df = test_df[test_df['failure_within_7_days'] == 1]

# #========== Visualization ==========
# st.subheader("Failures by Equipment Type")


# # Pie chart
# failure_counts = failed_df['equipment_type'].value_counts()
# fig1, ax1 = plt.subplots()
# ax1.pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%')
# st.pyplot(fig1)

# # Bar chart
# fig2, ax2 = plt.subplots()
# ax2.bar(failure_counts.index, failure_counts.values, color="skyblue")
# ax2.set_xlabel("Equipment Type")
# ax2.set_ylabel("Number of Failures")
# ax2.set_title("Failures by Equipment Type")
# st.pyplot(fig2)

# # ========== Details Table ==========
# st.subheader("Failed Equipment Details")
# st.dataframe(failed_df[['equipment_id', 'equipment_type', 'location']])

# # ========== Details Table ==========
# st.subheader("Failed Equipment count by Type")
# st.dataframe(failed_df[['equipment_type']].value_counts().reset_index(name='count').sort_values(by='count', ascending=False))

# # ========== Notification Button ==========
# st.subheader("Send Notifications")
# if st.button("Send Email Alerts"):
#     # (placeholder for email logic)    
#     st.success(" Email alerts sent to maintenance team!") ##0000FF
