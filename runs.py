from app import create_app
from video_processing.genrate_frame import flies_number


app= create_app()

if __name__ == '__main__':
    app.run(debug=True)
    