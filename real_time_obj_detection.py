import json

from video.video_stream import VideoStream
import argparse
import os
import datetime
import cv2
from models.sift import sift
from video.json_parser import parse_json, link_constructor, extract_id


def check_folder(input_folder, output_folder):
    processed_file = set()
    while True:
        for filename in os.listdir(input_folder):
            filepath = os.path.join(input_folder, filename)
            if os.path.isfile(filepath) and filename not in processed_file:
                try:
                    if filename.endswith('.json'):
                        data = parse_json(filename)
                        file_id = extract_id(filename)
                        video_stream = VideoStream(link_constructor(data)).real_time_detection()
                    else:
                        file_id = extract_id(filename)
                        video_stream = VideoStream(filepath).real_time_detection()
                    for i, detected_object in enumerate(video_stream):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_folder = os.path.join(output_folder, file_id)
                        if not os.path.exists(image_folder):
                            os.makedirs(image_folder)
                        image_filename = f'{file_id}_{i}.jpeg'
                        cv2.imwrite(f'{image_folder}/' + image_filename, detected_object)
                        #into_gray_scale = cv2.cvtColor(f'{image_folder}/' + image_filename,cv2.COLOR_BGR2GRAY)
                        #if sift.check_and_add_image(image_filename, 'ml_/models/sift/data/images') is not False:
                        cv2.imwrite(f'ml_/models/sift/data/images/' + image_filename, detected_object)
                        message = {"id": file_id, "file": image_filename, "timestamp": timestamp}
                        print(json.dumps(message))
                    processed_file.add(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object detection script')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Input folder path')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder path')
    args = parser.parse_args()
    check_folder(args.input_folder, args.output_folder)
