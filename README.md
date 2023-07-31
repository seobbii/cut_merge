# cut_merge

# cut_image.py
python cut_image.py {img_filename} {col_num} {row_num} {output_name}
1. 이미지는 원하는 n m의 수로 나눌수있도록 하였고, 변환은 주어진 조건을 50%의 확률로 진행되도록 하였습니다.
2. 파일 이름으로 특정할수 없도록 _0000에서 랜덤한 숫자로 설정하였습니다.

# merge_image.py
python merge_image.py --prefix {prefix_name} --column {col_num} --row {row_num} --output {output_name}
1. 사실 실패했다고 보고있습니다.
   - 우선 처음 문제를 보고 든 생각은 각 이미지들의 가장자리 rgb값을 비교하여 같거나, 비슷하다면 merge
     다르다면 변환을 진행시켜 원본 이미지를 만드려 했습니다.
   - 하지만 코드 설계가 잘 진행되지 않아 중단하였습니다.

2. 우연찮게 같은 문제를 확인하여 merge_image.py 파일을 완성하였습니다.
   - 코드 내부에 파일 이름 구성이 달라 해당 부분을 수정하는 작업을 거쳤습니다.
   - 다만 제가 완성하지 못한 코드라 평가는 감수하겠습니다.
   - 코드에 대한 스터디가 필요할 것 같습니다.
