import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

password=os.getenv('password')


# Email Configuration
SENDER_EMAIL = "anandsahu5097@gmail.com"  # Replace with your email address
SENDER_PASSWORD = password  # Replace with your email app-specific password
SMTP_SERVER = "smtp.gmail.com"  # Use your email provider's SMTP server
SMTP_PORT = 587  # Default SMTP port for TLS

def send_email(to_email, subject, message):
    """
    Send an email to the provided address.
    """
    try:
        # Create the email
        email = MIMEMultipart()
        email["From"] = SENDER_EMAIL
        email["To"] = to_email
        email["Subject"] = subject
        email.attach(MIMEText(message, "plain"))

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, email.as_string())

        return True
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False

# Streamlit Form
st.title("Contact Me")
st.write("Fill out the form below, and I'll get back to you as soon as possible!")

with st.form(key="contact_form"):
    email = st.text_input("Your Email Address", placeholder="example@example.com")
    subject = st.text_input("Subject", placeholder="Enter your subject")
    message = st.text_area("Message", placeholder="Enter your message here...")

    submitted = st.form_submit_button("Send Message")

    if submitted:
        if email and subject and message:
            if send_email(
                to_email="anandsahu5097@gmail.com",  # Replace with your email
                subject=f"Contact Form: {subject}",
                message=f"From: {email}\n\n{message}",
            ):
                st.success("Your message has been sent successfully!")
            else:
                st.error("Failed to send your message. Please try again.")
        else:
            st.warning("Please fill in all fields!")
