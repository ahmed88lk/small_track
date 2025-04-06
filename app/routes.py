from flask import Blueprint, render_template,Response,stream_template,stream_with_context

from video_processing.genrate_frame import  genrate_frames,genrate_frame,flies_number
import time

main = Blueprint('main', __name__)

@main.route('/')
def index():
     number = next(flies_number())
     return render_template('index.html', number=number)


@main.route('/live')
def live():
    return Response(genrate_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/video_feed')
def video_feed():
    return Response(genrate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/number_of_flies')

def number_of_flies():
     number=next(flies_number())
     return Response(number,mimetype='multipart/x-mixed-replace; boundary=frame')
     
    
    
    
