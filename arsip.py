from flask import Flask, render_template, request, redirect, url_for, session, flash
from textblob import TextBlob
from flask_mysqldb import MySQL
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from functools import wraps

from flask import render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # wajib untuk session

# Konfigurasi koneksi ke MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sentiment_db'
mysql = MySQL(app)

# Decorator supaya route butuh login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash("Silakan login terlebih dahulu.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validasi password dan konfirmasi password
        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok!', 'danger')
            return redirect(url_for('register'))

        # Enkripsi password sebelum disimpan
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Cek apakah username sudah ada di database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cur.fetchone()
        cur.close()

        if existing_user:
            flash('Username sudah digunakan, silakan pilih yang lain.', 'danger')
            return redirect(url_for('register'))

        # Membuat pengguna baru dan menyimpannya ke database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        mysql.connection.commit()
        cur.close()

        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Cek username dan password di database admin
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session['admin_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('index'))
        else:
            flash("Username atau password salah.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.")
    return redirect(url_for('login'))

@app.route('/')
# @login_required
def index():
    return render_template('index.html')

@app.route('/arsip')
#@login_required
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, cam_id, timestamp, plat_number, vehicle_type, violation, vehicle_speed, image_path FROM pelanggaran_kendaraan ORDER BY timestamp DESC")
    logs = cur.fetchall()
    cur.close()
    # Pastikan Anda punya template 'history.html'
    return render_template('arsip.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/logout')
def logout():
    session.clear()  # Hapus semua data session (logout)
    flash("Anda telah logout.")
    return redirect(url_for('login'))  # Kembali ke halaman login