from recognition import FacialRecognizer

import os

base_dir = os.path.dirname(os.path.realpath(__file__))

recognizer = FacialRecognizer('db.sqlite')

image = recognizer._get_image(os.path.join(base_dir, 'test_img', 'selfie.jpg'))
faces = recognizer.recognize_faces(image)
print(faces)
assert len(faces) == 1

face = faces[0]
recognizer.assign_name_to_image(image[face.top:face.bottom, face.left:face.right], 'Po')

image_same = recognizer._get_image(os.path.join(base_dir, 'test_img', 'selfie.jpg'))
faces = recognizer.recognize_faces(image_same)
print(faces)
assert len(faces) == 1
assert faces[0].name == 'Po'

image_friends = recognizer._get_image(os.path.join(base_dir, 'test_img', 'friends_with_me.jpg'))
faces = recognizer.recognize_faces(image_friends)
print(faces)
assert len(faces) == 3  # could be 4, but...
count_me = 0
for face in faces:
    if face.name == 'Po':
        count_me += 1
assert count_me == 1

image_guys = recognizer._get_image(os.path.join(base_dir, 'test_img', 'both_not_me.jpg'))
faces = recognizer.recognize_faces(image_guys)
print(faces)
assert len(faces) == 2
for face in faces:
    assert face.name != 'Po'
