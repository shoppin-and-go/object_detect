import cv2
import numpy as np
import os
from dataclasses import dataclass


@dataclass
class Image:
    name: str
    image: np.ndarray
    save_path: str


def get_width_height_area(x, y, w, h):
    width = w - x
    height = h - y
    area = width * height
    return width, height, area


def write_image(img: Image):
    if not os.path.exists(img.save_path):
        os.makedirs(img.save_path)
    img_name = os.path.join(img.save_path, img.name)
    cv2.imwrite(img_name, img.image)


def denoise_thresholded_image(kernel, img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    fg_threshold = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_threshold = cv2.dilate(fg_threshold, kernel, iterations=2)

    return fg_threshold


def send_image(img: Image):
    # TODO: Send image to server
    later = True


def send_all_images(images: list[Image]):
    for img in images:
        send_image(img)
        write_image(img)


def cut_roi(frame, roi):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = roi
    output = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return output


def read_video(config):
    print("Reading video...")

    # 비디오 파일 읽기
    video = config["video"]
    capture = cv2.VideoCapture(video)

    # 배경 구하기
    backSub = cv2.createBackgroundSubtractorMOG2(history=config["backHist"], varThreshold=config["backThresh"],
                                                 detectShadows=True)

    # 전체 프레임 수
    print("Total frames: ", int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 움직임 감지할 ROI 설정
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    D_roi_top_left_x = int(frame_width * config["detect_ROI"][0])
    D_roi_top_left_y = int(frame_height * config["detect_ROI"][1])
    D_roi_bottom_right_x = int(frame_width * config["detect_ROI"][2])
    D_roi_bottom_right_y = int(frame_height * config["detect_ROI"][3])
    D_roi_w, D_roi_h, D_roi_area = get_width_height_area(D_roi_top_left_x, D_roi_top_left_y, D_roi_bottom_right_x,
                                                         D_roi_bottom_right_y)

    # 캡쳐할 ROI 설정
    C_roi_top_left_x = int(frame_width * config["capture_ROI"][0])
    C_roi_top_left_y = int(frame_height * config["capture_ROI"][1])
    C_roi_bottom_right_x = int(frame_width * config["capture_ROI"][2])
    C_roi_bottom_right_y = int(frame_height * config["capture_ROI"][3])
    C_roi_w, C_roi_h, C_roi_area = get_width_height_area(C_roi_top_left_x, C_roi_top_left_y, C_roi_bottom_right_x,
                                                         C_roi_bottom_right_y)
    capture_ROI = (C_roi_top_left_x, C_roi_top_left_y, C_roi_bottom_right_x, C_roi_bottom_right_y)

    # 각종 변수 초기화
    image_counter = 0  # 물체가 인지된 이미지의 수 카운트
    frame_counter = 0  # 현재 프레임이 몇 번째 프레임인지 카운트
    process_counter = 1  # 넣고 빼는 과정이 몇 번 있었는지 카운트
    before_process_frame = 0  # 이전에 물체인지가 몇번째 프레임에서 발생했는지 카운트
    start_process_frame = 0  # 물체인지가 시작된 프레임
    sending = False  # 지난 프로세스가 넣고 빼는 과정의 마지막 프레임인지 여부
    send_Mode = False  # 물체가 인지되어 캡쳐할 상태인지 여부
    send_Frames = []  # 캡쳐할 프레임들을 저장할 리스트
    prev_frame = None  # 이전 프레임을 저장할 변수
    prev_prev_frame = None  # 이전 이전 프레임을 저장할 변수
    last_frames = []  # 마지막 프레임들을 저장할 리스트
    total_process_frame = 0  # 총 프로세스된 프레임 수
    is_proL_calculated = False  # 프로세스된 프레임 수가 계산되었는지 여부

    while True:
        ret, frame = capture.read()
        if not ret:  # 읽기 실패 혹은 더이상 읽을 비디오가 없을 때
            print("Reached the end of the video.")
            if send_Mode:
                send_all_images(send_Frames)
            break

        original_frame = frame.copy()

        frame_counter += 1

        if send_Mode and config["sendIMG_Nums"] == len(send_Frames):
            send_all_images(send_Frames)
            send_Frames = []
            send_Mode = False

        elif send_Mode:
            send_Frames.append(
                Image(name=f"{process_counter}. {frame_counter}.jpg", image=cut_roi(original_frame, capture_ROI),
                      save_path=config["sendIMG"]))

        if not sending and len(last_frames) != 0:
            print(f'last_frames: {len(last_frames)}')
            sendig_Frames = []
            for i, img in enumerate(last_frames):
                sendig_Frames.append(
                    Image(name=f"{process_counter}-2. {before_process_frame + i -2}.jpg", image=cut_roi(img, capture_ROI),
                          save_path=config["sendIMG"]))

            print(f"Sending last frames: {len(sendig_Frames)}")
            send_all_images(sendig_Frames)
            prev_frame = None
            prev_prev_frame = None
            last_frames = []

        # 프레임 스킵
        if capture.get(cv2.CAP_PROP_POS_FRAMES) % config["frames_skip"] != 0:
            if sending:
                if prev_frame is not None:
                    prev_prev_frame = prev_frame
                prev_frame = original_frame
            continue

        # ROI 추출 -> D_roi: 움직임 감지할 ROI, C_roi: 캡쳐할 ROI
        D_roi = frame[D_roi_top_left_y:D_roi_bottom_right_y, D_roi_top_left_x:D_roi_bottom_right_x]
        C_roi = frame[C_roi_top_left_y:C_roi_bottom_right_y, C_roi_top_left_x:C_roi_bottom_right_x]

        # 배경 추출 및 전경 마스크 생성
        foreground_mask = backSub.apply(D_roi)
        _, fg_threshold = cv2.threshold(foreground_mask, 254, 255, cv2.THRESH_BINARY)

        # 노이즈 제거를 위한 모폴로지 연산
        kernel = (3, 3)
        fg_threshold = denoise_thresholded_image(kernel, fg_threshold)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(fg_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 물체 감지
        if len(contours) > 0:
            # 면적이 가장 큰 것만 선택
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > config["detectPercents"] * D_roi_area:  # 물체기 ROI 면적의 일정 퍼센트 이상이면
                sending = True
                is_proL_calculated = False

                x, y, w, h = cv2.boundingRect(largest_contour)
                x_global = x + D_roi_top_left_x
                y_global = y + D_roi_top_left_y
                cv2.rectangle(frame, (x_global, y_global), (x_global + w, y_global + h), (0, 255, 0), 2)

                # 현재 프로세스가 이전 프레임과 연속적인지 확인
                if before_process_frame == 0:  # 처음으로 물체가 인지된 경우
                    before_process_frame = frame_counter
                    start_process_frame = frame_counter
                    send_Mode = True
                    send_Frames.append(Image(name=f"{process_counter}. {frame_counter}.jpg",
                                             image=cut_roi(original_frame, capture_ROI), save_path=config["sendIMG"]))

                elif frame_counter - before_process_frame > 2 * config["frames_skip"] + 1:
                    # 연속으로 감지가 되지 않을경우 -> 새로운 프로세스로 취급
                    process_counter += 1
                    before_process_frame = frame_counter
                    start_process_frame = frame_counter
                    send_Mode = True
                    send_Frames.append(Image(name=f"{process_counter}. {frame_counter}.jpg",
                                             image=cut_roi(original_frame, capture_ROI), save_path=config["sendIMG"]))

                else:  # 연속으로 감지가 되는 경우
                    before_process_frame = frame_counter
                    last_frames = []
                    if prev_prev_frame is not None:
                        last_frames.append(prev_prev_frame)
                    if prev_frame is not None:
                        last_frames.append(prev_frame)
                    last_frames.append(original_frame)

                # 물체로 인식된 영역의 이미지 저장
                object_img = frame[y_global:y_global + h, x_global:x_global + w]
                image_filename = f'process{process_counter}. frame{frame_counter}.jpg'
                detected_img = Image(image_filename, object_img, config["outputIMG"])
                write_image(detected_img)
                image_counter += 1

            else:
                sending = False
                # print(f'frame {frame_counter} : object detected but not enough')

        else:
            sending = False
            # print(f'frame {frame_counter} : no object detected')

        # ROI 영역을 시각적으로 보여줌
        cv2.rectangle(frame, (D_roi_top_left_x, D_roi_top_left_y), (D_roi_bottom_right_x, D_roi_bottom_right_y),
                      (255, 0, 0), 2)
        cv2.rectangle(frame, (C_roi_top_left_x, C_roi_top_left_y), (C_roi_bottom_right_x, C_roi_bottom_right_y),
                      (0, 0, 255), 2)

        # 프로세스가 끝난 경우 -> 프로세스 시간 출력
        if before_process_frame != 0 and not sending and not is_proL_calculated:
            total_process_frame = before_process_frame - start_process_frame
            print(f"process {process_counter} : {total_process_frame} frames")
            is_proL_calculated = True

        # 프로세스가 너무 짧은 경우 -> 무시
        if total_process_frame != 0 and total_process_frame < config["min_Process_Frames"]:
            print(f"process {process_counter} : process is too short")
            process_counter -= 1
            total_process_frame = 0
            send_Mode = False
            send_Frames = []
            last_frames = []
            prev_frame = None
            prev_prev_frame = None

        # 시각적으로 보여줄 화면들(새창으로)
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", fg_threshold)

        if sending:
            if prev_frame is not None:
                prev_prev_frame = prev_frame
            prev_frame = original_frame

        if cv2.waitKey(30) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

    print(f"Detected {frame_counter} frames.")
    print(f"Detected {process_counter} objects.")


if __name__ == '__main__':
    config = {
        "video": "video.mp4",  # 입력할 비디오 파일명
        "backHist": 50,  # 배경을 이전 몇 프레임까지 고려할지
        "backThresh": 15,  # 배경과 전경을 구분할 임계값 0 ~ 16(평균) ~ 255
        "frames_skip": 5,  # 몇 프레임마다 처리할지
        "detect_ROI": (0.01, 0.01, 0.7, 0.2),  # 움직임 감지할 ROI 크기 설정
        "capture_ROI": (0.01, 0.01, 0.9, 0.3),  # 화면 캡쳐할 ROI 크기 설정
        "detectPercents": 0.05,  # ROI 중 몇 퍼센트 이상의 객체를 검출할지
        "outputIMG": "./output",  # 이미지 저장할 경로
        "sendIMG": "./send",  # 전송할 이미지를 저장할 경로
        "sendIMG_Nums": 3,  # 전송할 이미지의 개수
        "min_Process_Frames": 6,  # 프로세스로 취급할 최소 프레임 수
    }

    read_video(config)
