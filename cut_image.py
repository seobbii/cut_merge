#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:13:15 2023

@author: seob
"""

import os
import random
import argparse
from PIL import Image, ImageOps

def cut_image(image_file_name, column_num, row_num, prefix_output_filename):
    # Load the image
    image = Image.open(image_file_name)

    # Adjust the size of the image if necessary
    width, height = image.size
    if width % column_num != 0:
        width = (width // column_num) * column_num
    if height % row_num != 0:
        height = (height // row_num) * row_num
    image = image.resize((width, height))

    # Calculate the size of each subimage
    subimage_width = width // column_num
    subimage_height = height // row_num

    # Cut the image and apply random transformations
    for i in range(column_num):
        for j in range(row_num):
            # Cut the image
            left = i * subimage_width
            upper = j * subimage_height
            right = left + subimage_width
            lower = upper + subimage_height
            subimage = image.crop((left, upper, right, lower))

            # Apply random transformations
            if random.random() < 0.5:
                subimage = ImageOps.mirror(subimage)
            if random.random() < 0.5:
                subimage = ImageOps.flip(subimage)
            if random.random() < 0.5:
                subimage = subimage.rotate(90)

            # Save the subimage with a random name
            subimage_file_name = f"{prefix_output_filename}_{random.randint(0, 10000)}.png"
            subimage.save(subimage_file_name)

def main():
    parser = argparse.ArgumentParser(description="Cut an image into MxN pieces and apply random transformations.")
    parser.add_argument("image_file_name", help="The name of the image file.")
    parser.add_argument("column_num", type=int, help="The number of columns to cut the image into.")
    parser.add_argument("row_num", type=int, help="The number of rows to cut the image into.")
    parser.add_argument("prefix_output_filename", help="The prefix for the output file names.")

    args = parser.parse_args()

    cut_image(args.image_file_name, args.column_num, args.row_num, args.prefix_output_filename)

if __name__ == "__main__":
    main()