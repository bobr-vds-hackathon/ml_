from video_stream import VideoStream
import argparse
import os
from json_parser import parse_json, link_constructor, extract_id
import datetime
import cv2
from models.sift.sift import check_and_add_image


def check_folder(input_folder, output_folder):
    processed_file = set()
    while True:
        for filename in os.listdir(input_folder):
            if os.path.isfile(os.path.join(input_folder, filename)) and filename not in processed_file:
                if filename.endswith('.json'):
                    processed_file.add(filename)
                    data = parse_json(filename)
                    file_id = extract_id(filename)
                    for i, detected_object in enumerate(VideoStream(link_constructor(data)).real_time_detection()):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename_image = f'{output_folder}/{file_id}/{file_id}_{i}.jpeg'
                        if check_and_add_image(filename_image, 'ml_/models/sift/data/images') is not False:
                            cv2.imwrite(filename_image, detected_object)
                            cv2.imwrite('ml_/models/sift/data/images/'+filename, detected_object)
                            message = {"id": file_id,
                                       "file": filename_image,
                                       "timestamp": timestamp}
                            print(message)
                        else:
                            pass

                else:
                    processed_file.add(filename)
                    file_id = extract_id(filename)
                    for i, detected_object in enumerate(VideoStream(filename).real_time_detection()):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename_image = f'{output_folder}/{file_id}/{file_id}_{i}.jpeg'
                        if check_and_add_image(filename, 'ml_/models/sift/data/images') is not False:
                            cv2.imwrite(filename, detected_object)
                            cv2.imwrite('ml_/models/sift/data/images/'+filename, detected_object)
                            message = {"id": file_id,
                                       "file": filename_image,
                                       "timestamp": timestamp}
                            print(message)
                        else:
                            pass


if __name__ == "main":
    parser = argparse.ArgumentParser(description='Object detection script')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Input folder path')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder path')
    args = parser.parse_args()
    check_folder(args.input_folder, args.output_folder)
