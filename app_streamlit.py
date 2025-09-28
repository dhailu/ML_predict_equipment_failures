import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
test_df = pd.read_csv("artifacts/data.csv")  # replace with your dataset

st.title(" Equipment Failure Prediction Dashboard")

# Filter failed equipment
failed_df = test_df[test_df['failure_within_7_days'] == 1]

# ========== Visualization ==========
st.subheader("Failures by Equipment Type")

# Pie chart
failure_counts = failed_df['equipment_type'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%')
st.pyplot(fig1)

# Bar chart
fig2, ax2 = plt.subplots()
ax2.bar(failure_counts.index, failure_counts.values, color="skyblue")
ax2.set_xlabel("Equipment Type")
ax2.set_ylabel("Number of Failures")
ax2.set_title("Failures by Equipment Type")
st.pyplot(fig2)

# ========== Details Table ==========
st.subheader("Failed Equipment Details")
st.dataframe(failed_df[['equipment_id', 'equipment_type', 'location']])

# ========== Notification Button ==========
st.subheader("Send Notifications")
if st.button("Send Email Alerts"):
    # (placeholder for email logic)
    st.success(" Email alerts sent to maintenance team!")
