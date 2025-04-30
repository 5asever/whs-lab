import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import platform
from collections import defaultdict

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 저장 폴더
os.makedirs("heatmaps", exist_ok=True)

# 댓글 수집
total_comments = []
for post_id in tqdm(range(1, 101)):
    url = f'http://board.nyan101.com/comments/{post_id}'
    response = requests.get(url)
    comments = response.json()
    total_comments.extend(comments)

print(f"[+] 총 {len(total_comments)}개의 댓글 수집 완료")

# 사용자별 댓글 정리
user_comments = {}
for comment in total_comments:
    author_id = comment['author_id']
    if author_id not in user_comments:
        user_comments[author_id] = []
    user_comments[author_id].append(comment)

print(f"[+] 총 {len(user_comments)}명의 사용자 발견")

# 봇 의심 계정 탐지 함수 (조건 1, 2, 3)
def detect_suspected_bots(total_comments, min_comments=50):
    user_timestamps = defaultdict(list)
    for c in total_comments:
        uid = c["author_id"]
        dt = datetime.strptime(c["created_at"], "%Y-%m-%d %H:%M")
        user_timestamps[uid].append(dt)

    results = {}

    for uid, timestamps in user_timestamps.items():
        timestamps.sort()
        info = {}

        # 조건1: 한 시간 동안 50개 이상 댓글
        for i in range(len(timestamps)):
            start = timestamps[i]
            end = start + timedelta(hours=1)
            count = sum(start <= t < end for t in timestamps)
            if count >= min_comments:
                info["50_in_1hour"] = (start, count)
                break

        # 조건2: 24시간 중 매 시간마다 활동 기록 존재
        hours = pd.Series([t.hour for t in timestamps])
        if len(hours.unique()) == 24:
            info["24hour_activity"] = True

        # 조건3: 시간대별 댓글 수의 편차가 거의 없음
        hour_counts = hours.value_counts()
        if len(hour_counts) > 0 and hour_counts.std() < 0.5:
            info["uniform_hourly_activity"] = True

        if info:
            results[uid] = info

    # 댓글 수 기준 상위 2명 선택
    sorted_candidates = sorted(results.items(), key=lambda x: len(user_timestamps[x[0]]), reverse=True)[:2]
    return sorted_candidates

#  봇 활동 시기 및 평균 간격 분석 출력
def print_bot_activity_details(suspected_users, user_comments):
    print("\n 실행결과")
    for i, (uid, _) in enumerate(suspected_users, 1):
        name = user_comments[uid][0]['author_name']
        timestamps = sorted([
            datetime.strptime(c['created_at'], '%Y-%m-%d %H:%M')
            for c in user_comments[uid]
        ])

        start = timestamps[0]
        end = timestamps[-1]

        # 평균 간격 계산
        if len(timestamps) > 1:
            deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 for i in range(len(timestamps)-1)]
            avg_gap = sum(deltas) / len(deltas)
        else:
            avg_gap = 0.0

        # 출력
        print(f"[{i}] 봇 계정 {uid}({name})의 가동시기 분석")
        print(f"[+] 가동 시작: {start:%m월%d일 %H시%M분}")
        print(f"[+] 가동 종료: {end:%m월%d일 %H시%M분}")
        print(f"[+] 평균 댓글 작성 간격: {avg_gap:.2f}분\n")

# 분석 실행
suspected_top2 = detect_suspected_bots(total_comments)

# 결과 출력
print("\n[+] 상위 봇 의심 계정 2명 (조건 1, 2, 3)")
for uid, info in suspected_top2:
    name = user_comments[uid][0]['author_name']
    print(f"\n[🚨] 사용자 #{uid} ({name})")
    if '50_in_1hour' in info:
        t, cnt = info['50_in_1hour']
        print(f" - [조건1] 한 시간 동안 50개 이상 댓글 작성: 시작 시각 {t} / 총 {cnt}개")
    if '24hour_activity' in info:
        print(f" - [조건2] 24시간 전체 시간대에 활동 기록 존재")
    if 'uniform_hourly_activity' in info:
        print(f" - [조건3] 시간대별 활동 편차가 거의 없음 (std < 0.5)")

# 봇 가동 시기 분석
print_bot_activity_details(suspected_top2, user_comments)

# Mission1: heatmap 생성 (상위 25명)
top_25_users = sorted(user_comments.items(), key=lambda x: len(x[1]), reverse=True)[:25]

for user_id, comments in top_25_users:
    user_name = comments[0]['author_name']
    activity_matrix = np.zeros((24, 7), dtype=int)

    for comment in comments:
        dt = datetime.strptime(comment['created_at'], '%Y-%m-%d %H:%M')
        weekday = dt.weekday()
        hour = dt.hour
        activity_matrix[hour, weekday] += 1

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['월', '화', '수', '목', '금', '토', '일'])
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{i:02d}' for i in range(24)])
    ax.set_title(f'User     #{user_id}({user_name})의 활동 패턴')
    fig.colorbar(im, ax=ax, label='활동 횟수')
    plt.tight_layout()
    filename = f"heatmaps/{user_id}_{user_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("\n[+] heatmaps 폴더에 25명 heatmap 저장 완료")
