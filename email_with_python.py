import smtplib
import email_config
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


email_user = email_config.EMAIL_ADDRESS
email_password = email_config.PASSWORD
email_send = email_user

subject = 'Thank you for using Build a Model - Here are your results'

msg = MIMEMultipart()
msg['From'] = email_user
msg['To'] = email_send
msg['Subject'] = subject

body = "This is a test message"
msg.attach(MIMEText(body,'plain'))

text = msg.as_string()
server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login(email_user,email_password)
server.sendmail(email_user,email_send,text)
server.quit()
   
