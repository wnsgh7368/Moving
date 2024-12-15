import torch
import cv2
import time
import numpy as np
from app.database.dbConnect import DbConnect
from ultralytics import YOLO
import logging
import signal
import threading

# 로깅 비활성화
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class PersonDetector:
    def __init__(self, model_path, video_path, db_config, frame_skip=3):
        try:
            print(f"비디오 경로: {video_path}")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 중인 디바이스: {self.device}")
            
            # 메모리 사용량 제한
            torch.set_num_threads(2)  # CPU 스레드 수 제한
            
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise Exception(f"비디오 파일을 열 수 없습니다: {video_path}")
            
            self.db_manager = DbConnect(**db_config)
            
            self.active_ids = {}
            self.inactive_time_limit = 2.5
            
            # 10초 평균 계산을 위한 변수들
            self.count_history = []
            self.last_save_time = time.time()

            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.roi_polygon = np.array([
                [700, 500], [1000, 500], [1000, 570], [700, 620]
            ], np.int32)

            self.frame_skip = frame_skip

            # 성능 최적화를 위한 설정 추가
            self.model.conf = 0.6  # confidence threshold 증가
            self.model.iou = 0.5
            self.model.max_det = 50  # detection 수 더 제한
            
            # 이미지 크기 감소
            self.img_size = 640  # 832에서 640으로 감소
            
            # 배치 크기 감소
            self.batch_size = 2  # 4에서 2로 감소
            self.frames_buffer = []

            self.running = True
            self._stop_event = threading.Event()
            
        except Exception as e:
            print(f"초기화 중 에러 발생: {str(e)}")
            raise e

    def detect(self, frame):
        """단일 이미지에 대한 detection 수행"""
        try:
            frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                results = self.model(img)
            
            return results[0]  # 첫 번째 결과만 반환
        except Exception as e:
            print(f"Detection 에러: {str(e)}")
            raise e

    def stop(self):
        """안전하게 처리를 중지"""
        print("검출 프로세스 중지 중...")
        self._stop_event.set()
        self.running = False

    def cleanup(self):
        """리소스 정리"""
        try:
            print("리소스 정리 시작...")
            
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                    print("비디오 캡처 해제 완료")
                except Exception as e:
                    print(f"비디오 캡처 해제 중 오류: {str(e)}")
            
            if hasattr(self, 'db_manager'):
                try:
                    if hasattr(self.db_manager, 'close'):
                        self.db_manager.close()
                    print("DB 연결 종료 완료")
                except Exception as e:
                    print(f"DB 연결 종료 중 오류: {str(e)}")
            
            # CUDA 메모리 정리
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    print("CUDA 메모리 정리 완료")
                except Exception as e:
                    print(f"CUDA 메모리 정리 중 오류: {str(e)}")
            
            # 모델 정리
            if hasattr(self, 'model'):
                try:
                    del self.model
                    print("모델 정리 완료")
                except Exception as e:
                    print(f"모델 정리 중 오류: {str(e)}")
            
            print("모든 정리 작업이 완료되었습니다.")
        except Exception as e:
            print(f"정리 작업 중 오류 발생: {str(e)}")
        finally:
            # 마지막으로 모든 참조 제거
            for attr in list(self.__dict__.keys()):
                try:
                    delattr(self, attr)
                except:
                    pass

    def process_video(self):
        frame_count = 0
        
        while self.cap.isOpened() and self.running and not self._stop_event.is_set():
            frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip != 0:
                continue
            
            # 이미지 전처리 최적화
            frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 배치 처리를 위해 프레임 버퍼에 추가
            self.frames_buffer.append(img)
            
            # 배치 크기만큼 모였을 때 처리
            if len(self.frames_buffer) >= self.batch_size:
                batch_tensor = torch.stack([
                    torch.from_numpy(f).float() / 255.0 
                    for f in self.frames_buffer
                ]).to(self.device).permute(0, 3, 1, 2)
                
                with torch.no_grad():
                    results = self.model(batch_tensor)
                    
                # 배치의 각 프레임에 대해 처리
                for i, result in enumerate(results):
                    current_ids = self.process_results(result, self.frames_buffer[i],
                                                    scale_x=frame.shape[1]/self.img_size,
                                                    scale_y=frame.shape[0]/self.img_size)
                    self.update_active_ids(current_ids)
                
                self.frames_buffer.clear()

            current_count = len(self.active_ids)
            self.count_history.append(current_count)

            current_time = time.time()
            if current_time - self.last_save_time >= 5:  # 10초마다
                avg_count = int(sum(self.count_history) / len(self.count_history))
                self.db_manager.insert_count(avg_count)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DB 저장: 5초 평균 인원 수 = {avg_count}명")
                print("-" * 50)
                
                # 초기화
                self.count_history.clear()
                self.last_save_time = current_time

            if self._stop_event.is_set():
                break

        print("비디오 처리를 종료합니다.")
        self.cleanup()

    def process_results(self, results, frame, scale_x, scale_y):
        current_ids = set()
        
        if len(results) > 0:
            boxes = results.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy() * [scale_x, scale_y, scale_x, scale_y])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf > 0.5:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if cv2.pointPolygonTest(self.roi_polygon, (center_x, center_y), False) >= 0:
                        track_id = i
                        current_ids.add(track_id)
                        self.active_ids[track_id] = time.time()

        return current_ids

    def update_active_ids(self, current_ids):
        current_time = time.time()
        for track_id in list(self.active_ids.keys()):
            if track_id not in current_ids and current_time - self.active_ids[track_id] > self.inactive_time_limit:
                del self.active_ids[track_id]