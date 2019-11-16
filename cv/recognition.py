"""
Tools for facial recognition and classification.
Toolset works on single images (no video streams).
"""

import face_recognition
import keras_facenet
import numpy as np
import PIL

import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')


class FaceObject:
    """
    Simple structure.
    Rectangle with optional name assigned
    """
    def __init__(self, t, r, b, l, name=None):
        self.top = t
        self.bottom = b
        self.right = r
        self.left = l
        self.name = name


class FacialRecognizer:
    MAX_SIMILAR_DISTANCE = .5  # magic number to be adjusted

    def __init__(self, db_name):
        """
        :string db_name: name for sqlite file
        """
        self.embedder = keras_facenet.FaceNet()
        self.db_name = db_name
        self._load_db()

    def __del__(self):
        log.info('Deleting the recognizer instance')
        self._dump_to_db()
    
    def recognize_faces(self, image):
        """
        :np.array image: cropped face-containing region
        :returns: list of FaceObject
        """
        face_objects = []
        for t, r, b, l in self._extract_face_coordinates(image):
            embedding = self._get_embedding(image[t:b, l:r])
            distance, closest_name = self._get_closest_name(embedding)
            face_objects.append(
                FaceObject(
                    t, r, b, l,
                    closest_name if distance < self.MAX_SIMILAR_DISTANCE else None
                )
            )
        return face_objects

    def assign_name_to_image(self, image, name):
        """
        :np.array image: cropped face-containing region
        :str name: non-empty string, may appear more than once
        :returns:
        """
        assert len(name) > 0
        # TODO

    def _get_closest_name(self, embedding):
        """
        :np.array embedding: (128,) array
        :returns: (float distance, str name)
        """
        # TODO

    def _get_image(self, image_path, mode='RGB'):
        """
        Example image reading method, mode in ('RGB', 'L')
        """
        im = PIL.Image.open(image_path)
        if mode:
            im = im.convert(mode)
        return np.array(im)

    def _extract_face_coordinates(self, image):
        """
        :np.array image:
        :returns: frame coordinates list, [(top, right, bottom, left), ...]
        Note: the face is image[t:b, l:r]
        """
        return face_recognition.face_locations(image)

    def _load_db(self):
        log.info(f'Loading embeddings to memory from {self.db_name}...')
        # TODO

    def _dump_to_db(self):
        log.info(f'Unloading embeddings to database {self.db_name}...')
        # TODO
    
    def _get_embedding(self, image):
        """
        :np.array image:
        :returns: np.array of facial embedding
        """
        return self.embedder.embeddings([image])

