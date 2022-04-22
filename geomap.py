#@velsiloh
import sys
import io
import folium # pip install folium
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView # pip install PyQtWebEngine
import sqlite3

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My Details')
        self.window_width, self.window_height = 1600, 1200
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        coordinate = (-0.29602864869324946, 36.0946559805275)
        m = folium.Map(
        	tiles='Stamen Terrain',
        	zoom_start=8,
        	location=coordinate
        )
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM records")

        rows = cur.fetchall()

        for row in rows:
            folium.Marker(location=[row[3],row[4]],popup=row[8], tooltip=row[9], icon=folium.Icon(color='red', icon='cloud')).add_to(m)

        # save map data to data object
        data = io.BytesIO()
        m.save(data, close_file=False)
        m.save('cotrack.html')

        webView = QWebEngineView()
        webView.setHtml(data.getvalue().decode())
        layout.addWidget(webView)
        cur.close()
        conn.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 35px;
        }
    ''')
    
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')