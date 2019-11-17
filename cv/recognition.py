"""
Tools for facial recognition and classification.
Toolset works on single images (no video streams).
"""

import keras_facenet
import face_recognition
import numpy as np
from scipy.spatial import cKDTree
import sqlite3
import PIL

import logging
import os

logging.basicConfig()
log = logging.getLogger()
log.setLevel('INFO')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

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

    def __repr__(self):
        return f'[{self.top}: {self.bottom}, {self.left}: {self.right}], name is {self.name}'


class FacialRecognizer:
    MAX_SIMILAR_DISTANCE = 1.35  # magic number to be adjusted
    EMBEDDING_SIZE = 512

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
        self.connection.close()
    
    def recognize_faces(self, image):
        """
        :np.array image: cropped face-containing region
        :returns: list of FaceObject
        """
        face_objects = []
        for t, r, b, l in self._extract_face_coordinates(image):
            embedding = self._get_embedding(image[t:b, l:r])
            distance, closest_name = self._get_closest_face(embedding)
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
        Note: must rebuild a kdtree here!
        """
        assert len(name) > 0
        self.names.append(name)
        self.embeddings = np.vstack((self.embeddings, self._get_embedding(image)))
        self.kdtree = cKDTree(self.embeddings)

    def _get_closest_face(self, embedding):
        """
        :np.array embedding: (EMBEDDING_SIZE,) array
        :returns: (float distance, str name), distance may be infinite if no candidates
        """
        if len(self.embeddings) == 0:
            return float('inf'), None
        distance, closest_index = self.kdtree.query(embedding, k=1)
        log.info(f'{distance}, {closest_index}')
        return distance[0], self.names[closest_index[0]]

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
        """
        Loads database into a matrix of (N, 128) embeddings and aligned names list.
        Builds a cKDTree for a faster search.
        """
        log.info(f'Loading embeddings to memory from {self.db_name}...')
        self.connection = sqlite3.connect(os.path.join(BASE_DIR, self.db_name))
        c = self.connection.cursor()
        self._create_db(c)
        self.names = []
        self.embeddings = np.empty((0, self.EMBEDDING_SIZE), dtype='f')
        for name, embedding_serialized in c.execute('''SELECT name, embedding_serialized FROM embeddings'''):
            self.names.append(name);
            embedding = np.array(eval(embedding_serialized), dtype='f')
            self.embeddings = np.vstack((self.embeddings, embedding))
        log.info(f'{len(self.embeddings)} entries of length {self.embeddings.shape[1]} found in db')
        self.kdtree = cKDTree(self.embeddings)

    def _dump_to_db(self):
        log.info(f'Unloading embeddings to database {self.db_name}...')
        c = self.connection.cursor()
        c.execute('''DROP TABLE IF EXISTS embeddings''')
        self._create_db(c)
        serialized_embeddings = [str(list(emb)) for emb in self.embeddings]
        log.info(f'Have to dump {len(self.embeddings)} entries')
        c.executemany('''INSERT INTO embeddings VALUES (?, ?)''', zip(self.names, serialized_embeddings))
        self.connection.commit()
    
    def _get_embedding(self, image):
        """
        :np.array image:
        :returns: np.array of facial embedding
        """
        return self.embedder.embeddings([image])

    def _create_db(self, cursor):
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                name text NOT NULL,
                embedding_serialized text NOT NULL
            )''') 

