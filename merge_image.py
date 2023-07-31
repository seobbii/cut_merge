import os
from PIL import Image, ImageFilter
import itertools
import numpy as np
import cv2
import argparse


# 채널 유사도 계산 함수
def calculate_channel_similarity(image1, image2):
    # 이미지를 넘파이 배열로 변환
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)

    # 채널 크기 맞추기
    pixels1 = np.resize(pixels1, pixels2.shape)

    # 채널 별로 픽셀 간의 차이 계산
    channel_diff = np.abs(pixels1 - pixels2)

    # 채널 별 차이를 평균하여 유사도 계산
    channel_similarity = 1 - np.mean(channel_diff) / 255

    return channel_similarity

# Canny 알고리즘을 사용하여 엣지 추출
def canny_edge_detection(image):
    # 그레이스케일 변환
    grayscale_image = image.convert("L")
    
    # 엣지 추출
    edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
    
    return edge_image

# 텍스처 특징 추출
def extract_texture_features(image):
    # OpenCV로 이미지 로드
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # ORB 특징 추출기 생성
    orb = cv2.ORB_create()
    
    # 특징 디스크립터 계산
    _, descriptors = orb.detectAndCompute(gray_image, None)
    
    return descriptors

# 색상 특징 추출
def extract_color_features(image):
    # 이미지를 넘파이 배열로 변환
    pixels = np.array(image)

    # RGB 각 채널의 평균값 계산
    color_features = np.mean(pixels, axis=(0, 1))

    return color_features

def merge_images(prefix, column_num, row_num, output_filename):
    # 파일 경로
    file_list = os.listdir()

    # 원본 이미지 경로
    imgs = []
    for img in file_list:
        if img.startswith(prefix):
            imgs.append(img)

    # 이미지 조각들을 담을 리스트
    image_pieces = []
    for image_name in file_list:
        if image_name.startswith(prefix):
            image_path = os.path.join('.', image_name)
            image = Image.open(image_path)
            image_pieces.append(image)

    # 원본 이미지 로드
    original_image_name = imgs[0]
    original_image_path = f'./{original_image_name}'
    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size
    target_ratio = original_width / original_height

    # 원본 비율이 1:1인 경우 원본 조각 이미지와 270도 회전한 이미지 추가
    if target_ratio == 1:
        for i in range(len(image_pieces)):
            original_piece = image_pieces[i]
            original_piece_rotated = original_piece.transpose(Image.ROTATE_270)
            image_pieces.append(original_piece)
            image_pieces.append(original_piece_rotated)

    # 이미지 조각들을 원본 비율로 맞추기
    for i in range(len(image_pieces)):
        piece_width, piece_height = image_pieces[i].size
        piece_ratio = piece_width / piece_height

        # 비율이 다른 경우 이미지 조각 조정
        if piece_ratio != target_ratio:
            transform = Image.ROTATE_270
            rotated_piece = image_pieces[i].transpose(transform)
            image_pieces.append(rotated_piece)

        # 이미지 조각 크기 조정
        resized_piece = image_pieces[i].resize((original_width // column_num, original_height // row_num))
        image_pieces[i] = resized_piece

        # 추가적인 변환 조합
        image_pieces.append(resized_piece)
        image_pieces.append(resized_piece.transpose(Image.FLIP_LEFT_RIGHT))
        image_pieces.append(resized_piece.transpose(Image.FLIP_TOP_BOTTOM))
        image_pieces.append(resized_piece.transpose(Image.FLIP_LEFT_RIGHT | Image.FLIP_TOP_BOTTOM))

    # 모든 경우의 수로 이미지 조합 생성 및 유사도 비교
    best_merged_image = None
    best_similarity = 0

    # 이미지 특징 추출
    original_edge_image = canny_edge_detection(original_image)
    original_texture_features = extract_texture_features(original_image)
    original_color_features = extract_color_features(original_image)

    total_combinations = len(list(itertools.permutations(range(len(image_pieces)), column_num * row_num)))
    combination_count = 0

    for combination in itertools.permutations(range(len(image_pieces)), column_num * row_num):
        combination_count += 1
        print(f"Progress: {100 * combination_count / total_combinations:.2f}%")

        merged_width = column_num * image_pieces[0].width
        merged_height = row_num * image_pieces[0].height
        merged_image = Image.new("RGB", (merged_width, merged_height))

        for row in range(row_num):
            for col in range(column_num):
                piece = image_pieces[combination[row * column_num + col]]
                merged_image.paste(piece, (col * piece.width, row * piece.height))

        # 이미지 크기 조정
        merged_image = merged_image.resize((original_width, original_height))

        # 특징 추출
        merged_edge_image = canny_edge_detection(merged_image)
        merged_texture_features = extract_texture_features(merged_image)
        merged_color_features = extract_color_features(merged_image)

        # 엣지 유사도 비교
        edge_similarity = calculate_channel_similarity(merged_edge_image, original_edge_image)
        # 텍스처 유사도 비교
        texture_similarity = calculate_channel_similarity(merged_texture_features, original_texture_features)
        # 색상 유사도 비교
        color_similarity = calculate_channel_similarity(merged_color_features, original_color_features)

        # 유사도 종합
        overall_similarity = (edge_similarity + texture_similarity + color_similarity) / 3
        print(overall_similarity)

        # 가장 높은 유사도 업데이트
        if overall_similarity > best_similarity:
            best_similarity = overall_similarity
            best_merged_image = merged_image

        # 이미 최고 유사도를 넘어설 수 없는 경우 종료
        if overall_similarity >= 1.0:
            break

    # 결과 이미지 저장
    if best_merged_image is not None:
        best_merged_image.save(output_filename)
        print(f"{output_filename}로 이미지 병합 완료")
    else:
        print("이미지 병합 실패")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge image pieces to reconstruct the original image.')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix of the image piece files')
    parser.add_argument('--column_num', type=int, required=True, help='Number of columns of the original image')
    parser.add_argument('--row_num', type=int, required=True, help='Number of rows of the original image')
    parser.add_argument('--output', type=str, required=True, help='File name of the output image')

    args = parser.parse_args()
    merge_images(args.prefix, args.column_num, args.row_num, args.output)
