# run 함수 내 루프 부분 수정 제안
while True:
    search = temp_grid & skipping_mask
    dist_map = distance_transform_edt(search)
    max_val = dist_map.max()
    
    if max_val < 0.5: 
        print("\n[알림] 더 이상 가공 가능한 공간이 없습니다. 종료합니다.")
        break
    
    seed = np.unravel_index(np.argmax(dist_map), search.shape)
    z1, z2, y1, y2, x1, x2 = grow_box_refined(temp_grid, *seed)
    
    # 현재 시도 중인 블록의 가상 크기
    attempt_size = np.array([x2-x1+1, y2-y1+1, z2-z1+1]) * self.pitch
    
    if np.all(attempt_size >= self.min_side):
        # ... (기존 블록 생성 로직) ...
        if len(self.cutters) % 10 == 0:
            tqdm.write(f"  [진행] 블록 #{len(self.cutters)} 생성 중... (크기: {attempt_size.max():.1f}mm)")
    else:
        # 이 부분이 78%에서 반복될 가능성이 높음
        skipping_mask[seed] = False
        # 너무 잦은 출력을 피하기 위해 가끔씩만 상태 보고
        if np.random.rand() < 0.05: 
            tqdm.write(f"  [분석] 작은 틈새 발견 (크기: {attempt_size.max():.1f}mm) - min_side 미달로 스킵 중...")
