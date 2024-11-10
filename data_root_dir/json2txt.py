import json
import os
import argparse
from tqdm import tqdm

def json2txt(json_folder, txt_folder):
    # txt 폴더가 없으면 생성
    os.makedirs(txt_folder, exist_ok=True)

    # json 파일 목록 가져오기
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Converting JSON to TXT"):
        # json 파일 경로와 txt 파일 경로 정의
        json_path = os.path.join(json_folder, json_file)
        txt_path = os.path.join(txt_folder, os.path.splitext(json_file)[0] + '.txt')

        # json 파일 열기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # bbox 정보가 존재하는지 확인 후 txt로 변환
        if 'bbox' in data:
            with open(txt_path, 'w', encoding='utf-8') as f:
                for item in data['bbox']:
                    x_coords = item['x']
                    y_coords = item['y']
                    
                    # x, y의 최소 및 최대값을 계산
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # 좌표를 (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max) 순서로 정렬
                    coords = f"{x_min},{y_min},{x_max},{y_min},{x_max},{y_max},{x_min},{y_max}"
                    
                    # 텍스트 추가
                    f.write(f"{coords},{item['data']}\n")
        else:
            print(f"'bbox' key not found in {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to TXT format.")
    parser.add_argument("json_folder", type=str, help="Folder containing JSON files.")
    parser.add_argument("txt_folder", type=str, help="Folder to save converted TXT files.")
    args = parser.parse_args()

    json2txt(args.json_folder, args.txt_folder)
