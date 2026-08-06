import os
import shutil

RESULTS_DIR = r"E:\AIVLE_10team\results"
BACKUP_DIR = os.path.join(RESULTS_DIR, "legacy_backup")

# 보존할 핵심 파일 목록
KEEP_FILES = {
    "bestYOLOm5080model.pt",
    "csrnet_ultimate_epoch_8.pth",
    "mangwon_label_summary.csv",
    "mangwon_label_pedestrians.csv"
}

def organize():
    if not os.path.exists(RESULTS_DIR):
        print(f"[ERROR] '{RESULTS_DIR}' 폴더가 존재하지 않습니다.")
        return

    # 백업 폴더 생성
    os.makedirs(BACKUP_DIR, exist_ok=True)
    print(f"[INFO] 백업 디렉토리 생성 완료: {BACKUP_DIR}")

    # results 폴더 내 모든 파일/폴더 목록 순회
    for item in os.listdir(RESULTS_DIR):
        # 백업 폴더 자체이거나, 보존할 파일인 경우 스킵
        if item == "legacy_backup" or item in KEEP_FILES:
            continue

        source_path = os.path.join(RESULTS_DIR, item)
        target_path = os.path.join(BACKUP_DIR, item)

        try:
            if os.path.isdir(source_path):
                # 디렉토리인 경우 shutil.move 사용
                # 대상 경로에 이미 존재하면 삭제 후 덮어쓰기
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.move(source_path, BACKUP_DIR)
                print(f"[MOVE] 폴더 이동 완료: {item} -> legacy_backup/")
            else:
                # 파일인 경우 shutil.move 사용
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.move(source_path, target_path)
                print(f"[MOVE] 파일 이동 완료: {item} -> legacy_backup/")
        except Exception as e:
            print(f"[ERROR] {item} 이동 실패: {e}")

if __name__ == "__main__":
    organize()
