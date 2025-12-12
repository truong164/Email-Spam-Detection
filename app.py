import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===== Page Config =====
st.set_page_config(
    page_title="Spam Email Classifier",
    layout="wide"
)

# ===== Load model =====
model = joblib.load("phanloaiemail.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ===== Card Component =====
def card(title, desc, link=None):
    st.markdown(
        f"""
        <div style="
            padding:1rem;
            border-radius:12px;
            background:#f2f2f2;
            margin-bottom:1rem;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);
        ">
            <h3 style="margin:0; color:#2c3e50;">{title}</h3>
            <p style="margin:0.2rem 0 0.6rem 0; color:#444;">{desc}</p>
            {f'<a href="{link}" target="_blank" style="text-decoration:none; color:#0066cc; font-weight:600;">Visit</a>' if link else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Title =====
st.markdown("<h1 style='text-align:center;'>Spam Email Classifier</h1>", unsafe_allow_html=True)

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["Dashboard", "Test Email", "Batch Upload"])


# =======================================================================================
# DASHBOARD
# =======================================================================================
with tab1:
    st.info("Ứng dụng phân loại Spam/Ham sử dụng Logistic Regression + TF-IDF.")

    col1, col2, col3 = st.columns(3)
    with col1:
        card("Model", "Logistic Regression")
    with col2:
        card("Vectorizer", "TF-IDF")
    with col3:
        card("Accuracy", "≈ 96%+")

    with st.expander("Ví dụ phân bố Spam/Ham"):
        sample = pd.DataFrame({"Type": ["Spam", "Ham"], "Count": [60, 40]})
        colA, colB, colC = st.columns([1,2,1])
        with colB:
            fig, ax = plt.subplots()
            sns.barplot(data=sample, x="Type", y="Count", ax=ax, palette="coolwarm")
            st.pyplot(fig)


# =======================================================================================
# TEST EMAIL
# =======================================================================================
with tab2:
    st.subheader("Test Email Realtime")

    review = st.text_area("Nhập nội dung email:", height=150)

    if st.button("Phân loại"):
        if review.strip():
            review_vec = vectorizer.transform([review])
            y_pred = model.predict(review_vec)[0]
            proba = model.predict_proba(review_vec)[0]

            st.write("### Kết quả:")
            if y_pred == 1:
                st.error(f"Spam (Độ tự tin: {proba[1]*100:.2f}%)")
            else:
                st.success(f"Ham (Độ tự tin: {proba[0]*100:.2f}%)")

            # Highlight spam keywords
            if y_pred == 1:
                keywords = ["free", "click", "win", "offer", "money", "credit"]
                highlighted = review
                for k in keywords:
                    highlighted = highlighted.replace(
                        k, f"<mark style='background:red; color:white;'>{k}</mark>"
                    )
                st.markdown("### Nội dung được highlight", unsafe_allow_html=True)
                st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhập nội dung email!")


# =======================================================================================
# BATCH UPLOAD CSV
# =======================================================================================
with tab3:

    with st.expander("Upload file CSV"):
        file_upload = st.file_uploader("Chọn file CSV chứa cột 'Message' & 'Category'", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload).dropna().drop_duplicates()

        if "Message" not in data.columns:
            st.error("File phải có cột `Message`")
        else:
            review_vec = vectorizer.transform(data["Message"])
            y_pred = model.predict(review_vec)
            proba = model.predict_proba(review_vec)

            data["Prediction"] = ["Spam" if p == 1 else "Ham" for p in y_pred]
            data["Confidence"] = proba.max(axis=1)

            if "Category" in data.columns:
                y_test = data["Category"].map({"ham":0, "spam":1})
                cm = confusion_matrix(y_test, y_pred)
            else:
                cm = None

            st.success("Phân loại thành công!")

            with st.expander("Kết quả dự đoán"):
                st.dataframe(data[["Message", "Prediction", "Confidence"]])

            if cm is not None:
                with st.expander("Confusion Matrix"):
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
                    st.pyplot(fig)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download kết quả",
                csv,
                "spam_predictions.csv",
                "text/csv",
                key="download-csv"
            )
