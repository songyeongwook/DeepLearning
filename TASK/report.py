# 신호등 'blinking' 상황 반복 시뮬레이션
import time

def blinking_crosswalk():
    print("신호등이 깜빡입니다. 4초 카운트 시작. '.'을 입력하면 종료합니다.")
    while True:
        user = input("계속하려면 Enter, 종료하려면 '.' 입력: ")
        if user.strip() == '.':
            print("종료합니다.")
            break
        for sec in range(1, 5):
            print(f"{sec}초 경과...")
            time.sleep(1)
            if sec == 2:
                print("2초가 지나서 못 건넌다!")

def traffic_light_simulation():
    import time
    while True:
        # Green light
        print("신호등: 초록불! 건너라!")
        for sec in range(10, 0, -1):
            print(f"초록불 {sec}초 남음")
            time.sleep(1)
        # Yellow light
        print("신호등: 노란불! 조심!")
        for sec in range(5, 0, -1):
            print(f"노란불 {sec}초 남음")
            if sec == 2:
                print("2초! 못 건넌다!")
            time.sleep(1)
        # Red light
        print("신호등: 빨간불! 기다려라!")
        for sec in range(5, 0, -1):
            print(f"빨간불 {sec}초 남음")
            time.sleep(1)
        print("신호등이 다시 초록불로 바뀝니다.")
        # 반복

if __name__ == "__main__":
    # traffic_light_simulation()  # 주석 해제 시 실행
    pass
def crosswalk_action(signal, knows_blink_duration=False):
    """
    signal: 'green', 'red', 'blinking'
    knows_blink_duration: 깜빡임 지속시간을 경험적으로 아는지 여부 (True/False)
    """
    if signal == 'green':
        return "길을 건넌다"
    elif signal == 'red':
        return "기다린다"
    elif signal == 'blinking':
        if knows_blink_duration:
            return "뛰어간다"
        else:
            return "초록색 신호로 바뀔 때까지 기다린다"
    else:
        return "멈춘다"  # 신호등이 없거나 알 수 없는 경우

# 사용 예시
signal = "blinking"
knows_blink_duration = True
print(crosswalk_action(signal, knows_blink_duration))