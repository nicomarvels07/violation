from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret-key-kamu'  # wajib untuk session

# Konfigurasi koneksi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sentiment_db'
mysql = MySQL(app)

# Route login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password FROM admin WHERE username = %s", (username,))
        admin = cur.fetchone()
        cur.close()

        if admin and check_password_hash(admin[1], password):
            session['admin_id'] = admin[0]
            session['username'] = username
            flash('Login berhasil!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah', 'danger')

    return render_template('login.html')

# Route logout
@app.route('/logout')
def logout():
    session.clear()
    flash('Anda sudah logout', 'info')
    return redirect(url_for('login'))

# Contoh proteksi halaman index (hanya untuk admin yang login)
@app.route('/')
def index():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])
