import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import os


def _findNode(parent, name, debug_name=None, parse=None):
	if debug_name is None:
		debug_name = name

	result = parent.find(name)
	if result is None:
		raise ValueError('missing element \'{}\''.format(debug_name))
	if parse is not None:
		try:
			return parse(result.text)
		except ValueError as e:
			raise_from(ValueError(
				'illegal value for \'{}\': {}'.format(debug_name, e)), None)
	return result


class Stanford40:
	def __init__(self, root, transform=None, is_test=False, keep_difficult=False, label_file=None):
		"""Dataset for VOC data.
		Args:
						root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
										Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		if is_test:
			image_sets_file = self.root / "ImageSplits/test.txt"
		else:
			image_sets_file = self.root / "ImageSplits/train.txt"
		self.ids = Stanford40._read_image_ids(image_sets_file)
		self.keep_difficult = keep_difficult
		# if the labels file exists, read in the class names
		label_file_name = self.root / "labels.txt"

		if os.path.isfile(label_file_name):
			class_string = ""
			with open(label_file_name, 'r') as infile:
				for line in infile:
					class_string += line.rstrip()

			# classes should be a comma separated list

			classes = class_string.split(',')
			# prepend BACKGROUND as first class
			# classes.insert(0, 'BACKGROUND')
			classes = [elem.replace(" ", "") for elem in classes]
			self.class_names = tuple(classes)
			logging.info("VOC Labels read from file: " + str(self.class_names))

		else:
			logging.info("No labels file, using default VOC classes.")
			self.class_names = ('running', 'walking')

		self.class_dict = {class_name: i for i,
						   class_name in enumerate(self.class_names)}

		print(self.class_dict)
		# remove ids
		self.remove_ids()

	def remove_ids(self):
		mask = []
		image_names_copy = self.ids.copy()
		for image_name in self.ids:
			filename = os.path.join(image_name + '.xml')
			tree = ET.parse(os.path.join(self.root, 'XMLAnnotations', filename))
			annotations = self.__parse_annotations(tree.getroot())
			mask.append(annotations['labels'].size != 0)

		mask = np.array(mask)
		self.ids = np.array(self.ids)
		self.ids = self.ids[mask]
	

	def __parse_annotation(self, element):
		""" Parse an annotation given an XML element.
		"""
		# truncated = _findNode(element, 'truncated', parse=int)
		# difficult = _findNode(element, 'difficult', parse=int)

		class_name = _findNode(element, 'action').text
		if class_name not in self.class_dict:
			return None, None

		box = []
		label = self.class_dict[class_name]

		bndbox = _findNode(element, 'bndbox')
		box.append(_findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1)
		box.append(_findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1)

		return box, label

	def __parse_annotations(self, xml_root):
		""" Parse all annotations under the xml_root.
		"""
		annotations = {'labels': [], 'bboxes': []}

		for i, element in enumerate(xml_root.iter('object')):
			try:
				box, label = self.__parse_annotation(
					element)
				if label is None:
					continue
			except ValueError as e:
				raise_from(ValueError(
					'could not parse object #{}: {}'.format(i, e)), None)

			annotations['bboxes'].append(np.array(box))
			annotations['labels'].append(label)

		annotations['bboxes'] = np.array(annotations['bboxes'])
		annotations['labels'] = np.array(annotations['labels'])

		return annotations

	def __getitem__(self, index):
		image_id = self.ids[index]
		image = self._read_image(image_id)
		box, label = self._get_annotation(image_id)
		label = label[0]
		box = box[0].astype(int)
		image = image.crop((box[0], box[1], box[2], box[3]))
		if self.transform:
			image = self.transform(image)
		return image, label

	def get_image(self, index):
		image_id = self.ids[index]
		image = self._read_image(image_id)
		print(image_id)
		if self.transform:
    			image, _ = self.transform(image)
		return image

	def get_annotation(self, index):
		image_id = self.ids[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip().split(".")[0])
		return ids

	def _get_annotation(self, image_id):
		annotation_file = self.root / f"XMLAnnotations/{image_id}.xml"
		objects = ET.parse(annotation_file).findall("object")
		boxes = []
		labels = []
		is_difficult = []
		for object in objects:
			class_name = object.find('action').text.strip()
			# we're only concerned with clases in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')
			
				# VOC dataset format follows Matlab, in which indexes start from 0
				x1 = int(bbox.find('xmin').text) - 1
				y1 = int(bbox.find('ymin').text) - 1
				x2 = int(bbox.find('xmax').text) - 1
				y2 = int(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])
		
		return (np.array(boxes, dtype=np.float32),
				np.array(labels, dtype=np.int64))

	def _read_image(self, image_id):
		image_file = self.root / f"JPEGImages/{image_id}.jpg"
		image = Image.open(str(image_file))
		# image = cv2.imread(str(image_file))
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image
