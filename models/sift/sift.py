import cv2
import numpy as np
import pickle
import os


class Sift:
    def __init__(self):
        self.sift_obj = cv2.SIFT.create()


def compute_sift(sift: Sift, image):
    img = cv2.imread(image)
    computed_sift = sift.sift_obj.detectAndCompute(img, None)
    return computed_sift


def get_descriptor_and_key_point(computed_sift):
    key_points, descriptor = computed_sift
    return key_points, descriptor


def store_key_points(key_points, image):
    data_path = "data/keypoints/" + str(image.split('.')) + ".txt"
    temp = []
    for point in key_points:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
    with open(data_path, 'wb') as fp:
        pickle.dump(temp, fp)


def store_descriptors(descriptor, image):
    descriptor_path = "data/keypoints/" + str(image.split('.')) + ".txt"
    with open(descriptor_path, 'wb') as fp:
        pickle.dump(descriptor, fp)


def fetchKeypointFromFile(image):
    filepath = "data/keypoints/" + str(image.split('.')) + ".txt"
    keypoint = []
    file = open(filepath, 'rb')
    deserialized_key_points = pickle.load(file)
    file.close()
    for point in deserialized_key_points:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint


def fetchDescriptorFromFile(image):
    filepath = "data/descriptors/" + str(image.split('.')) + ".txt"
    file = open(filepath, 'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor


def calculate_results_for_pairs(image_1, image_2):
    keypoint1 = fetchKeypointFromFile(image_1)
    descriptor1 = fetchDescriptorFromFile(image_1)
    keypoint2 = fetchKeypointFromFile(image_2)
    descriptor2 = fetchDescriptorFromFile(image_2)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches), len(keypoint1), len(keypoint2))
    return score


def calculateScore(matches, keypoint1, keypoint2):
    return 100 * (matches/min(keypoint1, keypoint2))


def calculateMatches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    topResults1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)
    topResults2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults


def check_and_add_image(image_path, folder_path, threshold=70):
    sift = Sift()
    data = compute_sift(sift, image_path)
    image_key_points, image_descriptor = get_descriptor_and_key_point(data)
    if len(os.listdir(folder_path)) != 0:
        for filename in os.listdir(folder_path):
            score = calculate_results_for_pairs(image_path, filename)
            if score >= threshold:
                return False
            else:
                store_descriptors(image_descriptor, image_path)
                store_key_points(image_key_points, image_path)
                cv2.imwrite('data/images', image_path)
                return image_path
    else:
        store_descriptors(image_descriptor, image_path)
        store_key_points(image_key_points, image_path)
        return image_path


def image_to_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    sift_ = Sift()
    a = compute_sift(sift=sift_, image='../../output/1/1_0.jpeg')
    c = get_descriptor_and_key_point(a)
    b = check_and_add_image(image_path='1_0.jpeg', folder_path='data/images')
    print(b)
